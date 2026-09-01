"""
Cross-validated training of a 4-channel CNN classifier.

Trains an ImageNet-pretrained convolutional backbone on RGB tiles concatenated
with a binary foreground mask as a fourth input channel. Dataset partitioning is
performed at the slide level using class-stratified group k-fold splitting so
that all tiles from a slide fall in the same fold.

Usage:
    python src/training/cv_train_mask.py \
        --data-path  data/metadata/dataset_trainval.csv \
        --imgs-path  data/tiles \
        --model      resnet34 \
        --lr         0.003 \
        --epochs     50 \
        --cv         5 \
        --output-dir outputs/models
"""

import datetime
import os
import time

import pickle
import copy
import shutil
from pathlib import Path
import tempfile
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import utils
from PIL import Image
import mlflow
import mlflow.pytorch
import tempfile
import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
import pandas as pd
from PIL import Image
import warnings

import mlflow
import mlflow.pytorch
import argparse
from mlflow.entities.run_info import RunInfo
from mlflow.tracking.client import MlflowClient
from mlflow.entities import  RunStatus
import traceback
import numpy as np
from sklearn.metrics import confusion_matrix
import gc
import torch
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import string
import random

class DatasetError(RuntimeError):
    pass

warnings.filterwarnings("ignore", module="mlflow")


class MlflowTracker(object):
    
    active_run: mlflow.ActiveRun = None
    tracking_uri = None    
    experiment_name = ""
    run_uuid = ""
    run_name = ""
    
    
       
    @staticmethod
    def initialize(experiment_name, run_name="", run_uuid=None, tracking_uri=None):
        if tracking_uri is None:
            raise RuntimeError(
                "MLflow tracking URI must be provided via --mlflow-uri"
            )
    
        MlflowTracker.tracking_uri = str(tracking_uri)
        MlflowTracker.experiment_name = experiment_name
        MlflowTracker.run_uuid = run_uuid
        MlflowTracker.run_name = run_name
    
        print("Initialized mlflow:")
        print(f" tracking_uri = {MlflowTracker.tracking_uri}")
        print(f" run_uuid     = {MlflowTracker.run_uuid}")
        print(f" experiment   = {MlflowTracker.experiment_name}")


    @staticmethod
    def connect():
        if MlflowTracker.tracking_uri is None:
            raise RuntimeError(
                "Mlflow tracking is not configured with a tracking URI. Cannot connect."
            )
    
        mlflow.set_tracking_uri(MlflowTracker.tracking_uri)
        mlflow.set_experiment(MlflowTracker.experiment_name)
    
        if MlflowTracker.run_uuid is not None:
            run = mlflow.start_run(run_id=MlflowTracker.run_uuid)
        else:
            run = mlflow.start_run(run_name=MlflowTracker.run_name)
    
        if run is None:
            run = mlflow.active_run()
            if run is None:
                raise RuntimeError("MLflow failed to start or retrieve an active run")
    
        MlflowTracker.active_run = run
        MlflowTracker.run_uuid = run.info.run_id
    
        print(f"MLflow run started: {MlflowTracker.run_uuid}")

           
        
    @staticmethod
    def reconnect():
        print("Reconnecting to mlflow")
        print(f"  {MlflowTracker.tracking_uri}")
        print(f"  {MlflowTracker.run_uuid}")
        print(f"  {MlflowTracker.experiment_name}")
        mlflow.end_run()
        print("  Ended previous run")
        mlflow.set_tracking_uri(MlflowTracker.tracking_uri)
        mlflow.set_experiment(MlflowTracker.experiment_name)
        MlflowTracker.active_run = mlflow.start_run(run_id=MlflowTracker.run_uuid)
        print("  Reconnected")
        
    @staticmethod
    def finish():
        print("Finish mlflow run")
        mlflow.end_run(RunStatus.to_string(RunStatus.FINISHED))

    @staticmethod
    def fail():
        mlflow.end_run(RunStatus.to_string(RunStatus.FAILED))

def load_classification_model(checkpoint_path, modelname, class_names=[0,1], device=torch.device('cpu')):
    args = argparse.Namespace()
    args.__dict__ = { 
                    "model": modelname,
                    "class_names": class_names
                    }
    model = torchvision.models.__dict__[args.model]()
    model = reshape_classification_head(model, args, class_names)
    model_without_ddp = model
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model_without_ddp.load_state_dict(checkpoint['model'])
    model_without_ddp.to(device)
    return model_without_ddp

def patch_first_conv_in_channels(model, in_ch=4):
    """
    Patch common torchvision models to accept in_ch input channels.
    Initializes new channel weights as mean of RGB weights.
    """
    if hasattr(model, "conv1") and isinstance(model.conv1, nn.Conv2d):
        old = model.conv1
        new = nn.Conv2d(in_ch, old.out_channels,
                        kernel_size=old.kernel_size,
                        stride=old.stride,
                        padding=old.padding,
                        bias=(old.bias is not None))
        with torch.no_grad():
            new.weight[:, :3, :, :] = old.weight
            new.weight[:, 3:4, :, :] = old.weight.mean(dim=1, keepdim=True)
            if old.bias is not None:
                new.bias.copy_(old.bias)
        model.conv1 = new
        return model

    # DenseNet torchvision: model.features.conv0
    if hasattr(model, "features") and hasattr(model.features, "conv0"):
        old = model.features.conv0
        new = nn.Conv2d(in_ch, old.out_channels,
                        kernel_size=old.kernel_size,
                        stride=old.stride,
                        padding=old.padding,
                        bias=(old.bias is not None))
        with torch.no_grad():
            new.weight[:, :3, :, :] = old.weight
            new.weight[:, 3:4, :, :] = old.weight.mean(dim=1, keepdim=True)
            if old.bias is not None:
                new.bias.copy_(old.bias)
        model.features.conv0 = new
        return model

    raise ValueError("Don't know how to patch first conv for this model.")


def calc_log_ROC(model, data_loader, device, classes, output_dir, cv_str, run_name):
    """
    Calculate ROC, save ROC curve & CSV, log both to MLflow.
    Saves inside:
        <output_dir>/<run_name>/<cv_str>/
            - ROC.png
            - ROC.csv
    """

    print(f"[ROC] Starting ROC calculation for run={run_name}, cv={cv_str}")

    try:
        model.eval()
        all_scores = []
        all_targets = []

        with torch.no_grad():
            print("[ROC] Collecting predictions...")
            for batch_idx, (image, target, _) in enumerate(data_loader):
                if batch_idx % 10 == 0:
                    print(f"[ROC]   Processing batch {batch_idx}/{len(data_loader)}")

                image = image.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                output = model(image)
                probs = torch.softmax(output, dim=1)

                all_scores.append(probs.detach().cpu())
                all_targets.append(target.detach().cpu())

        print("[ROC] Concatenating tensors...")
        scores = torch.cat(all_scores, dim=0)
        targets = torch.cat(all_targets, dim=0)

        print(f"[ROC] Data shapes → scores={scores.shape}, targets={targets.shape}")

        # For binary classification: positive class = last class
        class_of_interest = len(classes) - 1
        print(f"[ROC] Positive class index = {class_of_interest}")

        y_true = targets.numpy()
        y_score = scores[:, class_of_interest].numpy()

        print("[ROC] Computing ROC curve...")
        fpr, tpr, thresholds = roc_curve(
            y_true, y_score, pos_label=class_of_interest)
        roc_auc = auc(fpr, tpr)
        print(f"[ROC] AUC = {roc_auc:.4f}")

        # ------------------------------------------------------------------
        # Save folder path
        # ------------------------------------------------------------------
        
        save_dir = os.path.join(output_dir, cv_str)


        print(f"[ROC] Ensuring save directory exists: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # Save PNG
        # ------------------------------------------------------------------
        roc_png_path = os.path.join(save_dir, f"{cv_str}_ROC.png")
        print(f"[ROC] Saving ROC figure → {roc_png_path}")

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange',
                 lw=2, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve ({run_name} - {cv_str})')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(roc_png_path, dpi=400)
        plt.close()

        # ------------------------------------------------------------------
        # Save CSV
        # ------------------------------------------------------------------
        roc_csv_path = os.path.join(save_dir, f"{cv_str}_ROC.csv")
        print(f"[ROC] Saving ROC CSV → {roc_csv_path}")

        roc_df = pd.DataFrame({
            "FPR": fpr,
            "TPR": tpr,
            "Threshold": thresholds
        })
        roc_df.to_csv(roc_csv_path, index=False)

        # ------------------------------------------------------------------
        # Log to MLflow
        # ------------------------------------------------------------------
        print("[ROC] Logging artifacts to MLflow...")
        retry(lambda: mlflow.log_artifact(roc_png_path, artifact_path=f"{cv_str}_ROC"), 
              5, "Could not log ROC PNG")
        retry(lambda: mlflow.log_artifact(roc_csv_path, artifact_path=f"{cv_str}_ROC"), 
              5, "Could not log ROC CSV")

        print(f"[ROC] DONE for cv={cv_str}, run={run_name}")

    except Exception as e:
        print(f"[ROC] ERROR while calculating ROC for cv={cv_str}: {e}")
        traceback.print_exc()
        raise e  # rethrow so outer handler can see it


def resolve_mask_path(img_path: str) -> str:
    """
    Convert:
    .../3000_tiles/tiles/<slide>/<file>.png
    to:
    .../3000_tiles/masks/<slide>/<file>_mask.png
    """
    parts = img_path.replace("\\", "/").split("/")

    if "tiles" not in parts:
        raise RuntimeError(f"'tiles' not found in image path: {img_path}")

    idx = parts.index("tiles")
    parts[idx] = "masks"

    mask_path = "/".join(parts)

    # add _mask before .png
    if mask_path.endswith(".png"):
        mask_path = mask_path[:-4] + "_mask.png"

    return mask_path


        
def reshape_classification_head(model, args, class_names):
    num_classes = len(class_names)

    # ResNet, ResNeXt, WideResNet, ShuffleNet
    if args.model in [
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'resnext50_32x4d', 'resnext101_32x8d',
        'wide_resnet50_2', 'wide_resnet101_2',
        'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0'
    ]:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    # SqueezeNet
    elif args.model == 'squeezenet1_1':
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model.num_classes = num_classes

    # DenseNet
    elif args.model in ['densenet121', 'densenet161', 'densenet169', 'densenet201']:
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

    # MobileNetV2 and EfficientNet (v1)
    elif args.model.startswith("efficientnet") or args.model == "mobilenet_v2":
        if isinstance(model.classifier, nn.Sequential):
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
        else:
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)

    # ConvNeXt & RegNet
    elif args.model.startswith("convnext") or args.model.startswith("regnet"):
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

    else:
        raise ValueError(f'Changing classification head for model "{args.model}" is not supported.')

    return model

import random
import torchvision.transforms.functional as TF

def extract_slide_id_from_path(img_rel_path: str) -> str:
    """
    Assumes something like: tiles/<slide_id>/<roi>.png or rois/<slide_id>/<roi>.png
    Adjust this if your structure differs.
    """
    parts = img_rel_path.replace("\\", "/").split("/")
    return parts[-2] if len(parts) >= 2 else "unknown_slide"

class PairedForegroundAwareRandomResizedCrop:
    """
    Like RandomResizedCrop, but retries until crop has >= min_fg foreground fraction in mask.
    """
    def __init__(self, size, scale=(0.5, 1.0), ratio=(0.75, 1.33),
                 min_fg=0.05, max_tries=20, interpolation=TF.InterpolationMode.BILINEAR):
        self.size = size if isinstance(size, (tuple, list)) else (size, size)
        self.scale = scale
        self.ratio = ratio
        self.min_fg = min_fg
        self.max_tries = max_tries
        self.interpolation = interpolation

    def __call__(self, img, mask):
        # img: PIL RGB, mask: PIL L (0/255)
        w, h = img.size
        for _ in range(self.max_tries):
            i, j, th, tw = transforms.RandomResizedCrop.get_params(
                img, scale=self.scale, ratio=self.ratio
            )
            mask_crop = TF.crop(mask, i, j, th, tw)
            mask_np = np.array(mask_crop)
            fg_frac = (mask_np > 127).mean()  # foreground proportion
            if fg_frac >= self.min_fg:
                img = TF.resized_crop(img, i, j, th, tw, self.size, interpolation=self.interpolation)
                # mask must use NEAREST to preserve binary nature
                mask = TF.resized_crop(mask, i, j, th, tw, self.size, interpolation=TF.InterpolationMode.NEAREST)
                return img, mask

        # fallback: center crop-resize
        img = TF.resize(img, self.size, interpolation=self.interpolation)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)
        return img, mask


class PairedRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        return img, mask


class MaskDropout:
    """
    Light regularization so model doesn't treat mask as perfect truth.
    Randomly drops a small fraction of foreground pixels.
    """
    def __init__(self, drop_prob=0.15):
        self.drop_prob = drop_prob

    def __call__(self, mask_pil):
        m = np.array(mask_pil).astype(np.uint8)
        fg = m > 127
        if fg.any():
            drop = np.random.rand(*m.shape) < self.drop_prob
            m[fg & drop] = 0
        return Image.fromarray(m)


class RGBMaskTo4CHTensor:
    def __init__(self, normalize_rgb=True):
        self.normalize_rgb = normalize_rgb
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

    def __call__(self, img_pil, mask_pil):
        # RGB -> tensor
        rgb = TF.to_tensor(img_pil)  # [3,H,W] float 0..1
        if self.normalize_rgb:
            rgb = self.normalize(rgb)

        # mask -> {0,1}
        mask_np = (np.array(mask_pil) > 127).astype(np.float32)  # [H,W]
        mask = torch.from_numpy(mask_np)[None, :, :]  # [1,H,W]

        x = torch.cat([rgb, mask], dim=0)  # [4,H,W]
        return x

import time
import random
from PIL import Image

MAX_IO_RETRIES = 3
IO_RETRY_SLEEP = 0.1


def safe_open_image(path, mode="RGB"):
    """
    Robust image open for HPC / multiprocess I/O.
    Returns None if file cannot be opened after retries.
    """
    for _ in range(MAX_IO_RETRIES):
        try:
            img = Image.open(path)
            img.load()  # force actual disk read
            return img.convert(mode)
        except Exception:
            time.sleep(IO_RETRY_SLEEP)
    return None


class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, root, path_to_csv,
                 paired_crop=None,
                 paired_flip=None,
                 mask_dropout=None,
                 to_tensor=None,
                 dataset_type='binary',
                 class_names=[]):
        self.root = root
        df = pd.read_csv(path_to_csv)

        self.imgs = list(df.iloc[:, 0])
        self.labels = list(df.iloc[:, 1])

        if "slide_id" in df.columns:
            self.slide_ids = list(df["slide_id"])
        else:
            self.slide_ids = [extract_slide_id_from_path(p) for p in self.imgs]

        self.paired_crop = paired_crop
        self.paired_flip = paired_flip
        self.mask_dropout = mask_dropout
        self.to_tensor = to_tensor

        if dataset_type == 'binary':
            self.classes = class_names if class_names else ['class0', 'class1']
        elif dataset_type == 'multi_class':
            self.classes = class_names if class_names else sorted(set(self.labels))
        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_rel_path = self.imgs[idx]
        img_path = os.path.join(self.root, img_rel_path)
        label = int(self.labels[idx])
        slide_id = self.slide_ids[idx]

       
        img = safe_open_image(img_path, mode="RGB")
        mask_path = resolve_mask_path(img_path)
        mask = safe_open_image(mask_path, mode="L")

        # ---------------- SAFE FALLBACK ----------------
        if img is None:
            raise DatasetError(f"[IMG FAILED] {img_path}")

        if mask is None:
            raise DatasetError(f"[MASK FAILED] {mask_path}")



        # ------------------------------------------------

        # paired geometric transforms
        if self.paired_crop is not None:
            img, mask = self.paired_crop(img, mask)
        if self.paired_flip is not None:
            img, mask = self.paired_flip(img, mask)

        # mask regularization
        if self.mask_dropout is not None:
            mask = self.mask_dropout(mask)

        # to 4ch tensor
        x = self.to_tensor(img, mask) if self.to_tensor is not None else (img, mask)

        return x, label, slide_id


    
def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = f'Epoch: [{epoch}]'

    for image, target, slide_id in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()

        image = image.to(device)
        target = target.long().to(device)

        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = output.argmax(dim=1)

        acc1 = utils.accuracy(output, target, topk=(1,))[0]
        batch_size = image.size(0)

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg / 100


def make_probe_batch(x4, mode: str):
    rgb = x4[:, 0:3, :, :]
    m = x4[:, 3:4, :, :]

    if mode == "normal":
        return x4

    if mode == "fg_only":
        rgb2 = rgb * m
        return torch.cat([rgb2, m], dim=1)

    if mode == "bg_only":
        rgb2 = rgb * (1.0 - m)
        return torch.cat([rgb2, m], dim=1)

    if mode == "mask_shuffle":
        idx = torch.randperm(m.shape[0], device=m.device)
        m2 = m[idx]
        return torch.cat([rgb, m2], dim=1)

    raise ValueError(mode)


import torch
import utils
import tempfile

def evaluate(model, criterion, data_loader, device, class_names, cv_str=""):
    model.eval()
    num_classes = len(class_names)
    metric_logger = utils.MetricLogger(delimiter="  ")

    confusion = torch.zeros(num_classes, num_classes)

    probe_correct = {
        "normal": 0,
        "fg_only": 0,
        "bg_only": 0,
        "mask_shuffle": 0
    }
    probe_total = 0

    with torch.no_grad():
        for image, target, slide_id in data_loader:
            image = image.to(device)
            target = target.long().to(device)

            for mode in probe_correct.keys():
                x_probe = make_probe_batch(image, mode)
                output = model(x_probe)
                preds = output.argmax(dim=1)

                if mode == "normal":
                    loss = criterion(output, target)
                    acc1 = utils.accuracy(output, target, topk=(1,))[0]

                    metric_logger.update(loss=loss.item())
                    metric_logger.meters['acc1'].update(acc1.item(), n=image.size(0))

                    for t, p in zip(target, preds):
                        confusion[t, p] += 1

                probe_correct[mode] += (preds == target).sum().item()

            probe_total += image.size(0)

    # compute per-class stats
    recall = confusion.diag() / (confusion.sum(1) + 1e-9)
    precision = confusion.diag() / (confusion.sum(0) + 1e-9)
    # per-class F1
    f1_per_class = (2 * precision * recall) / (precision + recall + 1e-9)
    
    macro_precision = precision.mean().item()
    macro_recall = recall.mean().item()
    macro_f1 = f1_per_class.mean().item()
    print(f"Macro Precision: {macro_precision:.4f}, Macro Recall: {macro_recall:.4f}, Macro F1: {macro_f1:.4f}")
    
    tp = confusion.diag().sum()
    fp = confusion.sum(0).sum() - tp
    fn = confusion.sum(1).sum() - tp

    micro_precision = (tp / (tp + fp + 1e-9)).item()
    micro_recall = (tp / (tp + fn + 1e-9)).item()
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-9)

    print(f"Micro Precision: {micro_precision:.4f}, Micro Recall: {micro_recall:.4f}, Micro F1: {micro_f1:.4f}")

    recall_dict = {f"{cv_str}Recall_{c}": recall[i].item() for i, c in enumerate(class_names)}
    precision_dict = {f"{cv_str}Precision_{c}": precision[i].item() for i, c in enumerate(class_names)}

    probe_metrics = {
        f"{cv_str}ProbeAcc_{k}": probe_correct[k] / probe_total
        for k in probe_correct
    }

    return (
        metric_logger.loss.global_avg,
        metric_logger.acc1.global_avg / 100,
        micro_f1,
        {
            "macro_f1": macro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
        },
        recall_dict,
        precision_dict,
        probe_metrics
    )


def retry(cb, max_retries=5, err_msg=None):
    i = 0
    exception = None
    while(i<max_retries):
        i += 1
        try:
            return cb()
        except Exception as e:
            exception = e
            print(traceback.format_exc())
            if err_msg is None:
                print(e)
            else:
                print(err_msg.format(e))
            print("Restart mlflow")
            try:
                MlflowTracker.reconnect()
            except Exception as me:
                print(f"Mlflw restart failed: {me}")
            time.sleep(10)
    raise exception
        

def load_data(args, cv):
    print("Loading ONE-CSV dataset with StratifiedGroupKFold")

    df = pd.read_csv(args.data_path)

    y = df["label"].values
    groups = df["slide_id"].values

    sgkf = StratifiedGroupKFold(
        n_splits=args.nr_cv,
        shuffle=True,
        random_state=42
    )

    splits = list(sgkf.split(df, y, groups))
    train_idx, val_idx = splits[cv]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val   = df.iloc[val_idx].reset_index(drop=True)

    fold_dir = output_dir(args, cv)
    os.makedirs(fold_dir, exist_ok=True)

    train_csv = os.path.join(fold_dir, "train.csv")
    val_csv   = os.path.join(fold_dir, "val.csv")

    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)

    # ------------------ transforms ------------------
    paired_crop_train = PairedForegroundAwareRandomResizedCrop(
        size=tuple(args.input_size),
        scale=(0.5, 1.0),
        ratio=(0.75, 1.33),
        min_fg=0.05
    )

    paired_flip_train = PairedRandomHorizontalFlip(p=0.5)
    mask_dropout_train = MaskDropout(drop_prob=0.10)
    to_tensor = RGBMaskTo4CHTensor(normalize_rgb=True)

    dataset_train = CSVDataset(
        args.imgs_path,
        train_csv,
        paired_crop=paired_crop_train,
        paired_flip=paired_flip_train,
        mask_dropout=mask_dropout_train,
        to_tensor=to_tensor,
        dataset_type=args.dataset_type,
        class_names=args.class_names
    )

    dataset_val = CSVDataset(
        args.imgs_path,
        val_csv,
        paired_crop=lambda img, mask: (
            TF.resize(img, args.input_size),
            TF.resize(mask, args.input_size, interpolation=TF.InterpolationMode.NEAREST)
        ),
        paired_flip=None,
        mask_dropout=None,
        to_tensor=to_tensor,
        dataset_type=args.dataset_type,
        class_names=args.class_names
    )

    return dataset_train, dataset_val, dataset_train.classes


def output_dir(args, cv):
    return os.path.join(args.output_dir, args.run_name, f"cv{cv}")

def cv_str(cv):
    return f"cv{cv}_"

def inner_main(args, cv, resume=False):
    if utils.is_main_process():
        if args.output_dir:
            if Path(output_dir(args, cv)).exists():
                shutil.rmtree(output_dir(args, cv))
            utils.mkdir(output_dir(args, cv))

#   utils.init_distributed_mode(args)
    print(args)

    print("Loading data")
    
    dataset_train, dataset_val, class_names = load_data(args, cv)
    
    data_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=RandomSampler(dataset_train),
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,     # IMPORTANT for HPC
        prefetch_factor=2            # reduce I/O bursts
    )


    data_loader_test = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        sampler=SequentialSampler(dataset_val),
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True
    )


    
    print("Creating model")
    model = torchvision.models.__dict__[args.model](pretrained=args.pretrained)
    model = patch_first_conv_in_channels(model, in_ch=4)
    model = reshape_classification_head(model, args, class_names)

    device_arg = args.device
    
    if args.device == "cuda" and not torch.cuda.is_available():
        device_arg = "cpu"
        print(f"Device: {device_arg}")
    device = torch.device(device_arg)
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    labels = np.array(dataset_train.labels)
    class_counts = np.bincount(labels, minlength=len(class_names))
    class_weights = class_counts.sum() / (len(class_counts) * np.maximum(class_counts, 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.lr_schedule == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    model_without_ddp = model

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # args.start_epoch = checkpoint['epoch'] + 1

    
    print("Start training")
    start_time = time.time()
    best_model_performance = -1.0
    best_model_epoch = -1
    best_model = None
    best_model_optimizer = None
    best_model_lr_scheduler = None
    final_epoch = None   # to track last epoch that actually ran

    
    if utils.is_main_process() and not resume:
        # Log parameters on first execution 
        for key, value in vars(args).items():
            mlflow.log_param(key, value)
            
    print(f"Total epochs {args.epochs}")    
    for epoch in range(args.start_epoch, args.epochs):
        final_epoch = epoch
        print(f"Epoch {epoch}")
        
        train_loss, train_acc = train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        loss, acc, micro_f1, macro_metrics, recall_per_class, precision_per_class, probe_metrics = evaluate(model, criterion, data_loader_test, device=device, class_names=class_names, cv_str=cv_str(cv))
        

        if utils.is_main_process():
            # log metrics
            retry(lambda: mlflow.log_metric(f"{cv_str(cv)}Train_Acc", train_acc, step=epoch), 5)
            retry(lambda: mlflow.log_metric(f"{cv_str(cv)}Train_Loss", train_loss, step=epoch), 5)
            retry(lambda: mlflow.log_metric(f"{cv_str(cv)}Accuracy_Normal_Val", acc, step=epoch), 5)
            retry(lambda: mlflow.log_metric(f"{cv_str(cv)}Micro_F1_Val", micro_f1, step=epoch), 5)

            retry(lambda: mlflow.log_metric(f"{cv_str(cv)}Macro_F1_Val", macro_metrics["macro_f1"], step=epoch), 5)
            retry(lambda: mlflow.log_metric(f"{cv_str(cv)}Macro_Precision_Val", macro_metrics["macro_precision"], step=epoch), 5)
            retry(lambda: mlflow.log_metric(f"{cv_str(cv)}Macro_Recall_Val", macro_metrics["macro_recall"], step=epoch), 5)

            retry(lambda: mlflow.log_metrics(recall_per_class, step=epoch), 5)
            retry(lambda: mlflow.log_metrics(precision_per_class, step=epoch), 5)
            retry(lambda: mlflow.log_metrics(probe_metrics, step=epoch), 5)


            #track best model
            if acc > best_model_performance:

                best_model = copy.deepcopy(model_without_ddp)
                best_model_optimizer = copy.deepcopy(optimizer)
                best_model_lr_scheduler = copy.deepcopy(lr_scheduler)
                best_model_performance = acc
                best_model_epoch = epoch

                #save best model locally
                # Log every best model to prevent wasting progress
                if args.output_dir:
                    checkpoint = {
                        'model': best_model.state_dict(),
                        'optimizer': best_model_optimizer.state_dict(),
                        'lr_scheduler': best_model_lr_scheduler.state_dict(),
                        'last_epoch': best_model_epoch,
                        'args': args}
                    utils.save_on_master(
                        checkpoint,
                        os.path.join(output_dir(args, cv), 'model_best.pth'))
                    with open(os.path.join(output_dir(args, cv), 'best_model_epoch.log'), "a") as epoch_log:
                        epoch_log.write(f"Logged model at epoch {best_model_epoch}\n")
                        
    # ---------------- SAFETY GUARD ----------------
    # In case acc never improved (e.g. degenerate training),
    # fall back to last model so downstream code does not crash
    if best_model is None:
        print("[WARN] best_model was never set. Using last epoch model.")
        best_model = model_without_ddp
        best_model_performance = float("nan")
    # ---------------------------------------------
        
    # log model
    if utils.is_main_process():
        # also save best model performance as last step performance
        retry(lambda: mlflow.log_metric(f'{cv_str(cv)}Val_Acc_Best', best_model_performance), 5) # "Could not log final accuracy {e}"

        print("\nSaving FINAL model (last epoch) ...")

        checkpoint_final = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'last_epoch': final_epoch,
            'args': args
        }
        print("Final epoch:", final_epoch)
        utils.save_on_master(
            checkpoint_final,
            os.path.join(output_dir(args, cv), 'model_last.pth')
        )

        print("Logging FINAL model (last epoch) to MLflow ...")
        mlflow.log_artifact(
        os.path.join(output_dir(args, cv), 'model_last.pth'),
        artifact_path=f"{cv_str(cv)}_pytorch_last")


        if args.log_roc:
            print("\nCalc and log ROC")
            calc_log_ROC(best_model, data_loader_test, device=device, classes=class_names, output_dir=output_dir(args, cv), cv_str=cv_str(cv), run_name=args.run_name)
        
        if args.log_model:
            print("\nSaving BEST model ...")
            retry(lambda: mlflow.log_artifact(os.path.join(output_dir(args, cv), 'model_best.pth'), artifact_path=f"{cv_str(cv)}pytorch-best-model"), 5) # "Could not log final model {e}"
            # retry(lambda: mlflow.pytorch.log_model(best_model, artifact_path=f"{args.cv_str}pytorch-model", pickle_module=pickle), 5) # "Could not log final model {e}"
            print("\nThe best model is logged at:\n%s" % os.path.join(mlflow.get_artifact_uri(), cv_str(cv), "pytorch-model"))
        



    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    result = best_model_performance
    
    # Free GPU memory
    del model
    del best_model
    del optimizer
    del criterion
    del lr_scheduler
    torch.cuda.empty_cache()
    gc.collect()

    full_metrics = {
        "val_loss": loss,
        "val_acc": acc,
        "micro_f1": micro_f1,
        **macro_metrics,        # macro_f1, macro_precision, macro_recall, etc.
        **recall_per_class,     # Recall_class0, Recall_class1, ...
        **precision_per_class,  # Precision_class0, Precision_class1, ...
        **probe_metrics,        # ProbeAcc_fg_only, ProbeAcc_bg_only, ...
    }

    return macro_metrics["macro_f1"], full_metrics



def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def main(args, cv:int, run_name:str,  tags:dict | None = None, run_uuid: str | None = None):
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    args.run_name = run_name

    MlflowTracker.initialize(experiment_name=args.experiment_name, run_name=run_name, run_uuid=run_uuid, tracking_uri=args.mlflow_uri)
    MlflowTracker.connect()
    resume = run_uuid is not None  # single source of truth

    with MlflowTracker.active_run:
        if tags:
            for k, v in tags.items():
                mlflow.set_tag(k, v)
        return inner_main(args, cv, resume=resume)

class StoreDictKeyPair(argparse.Action):
     def __call__(self, parser, namespace, values, option_string=None):
         my_dict = {}
         for kv in values.split(","):
             k,v = kv.split("=")
             my_dict[k] = v
         setattr(namespace, self.dest, my_dict)

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--data-path', help='path to csv containing dataset')
    parser.add_argument('--imgs-path', default='/', help='Root folder containing all the images with relative paths in data-path(s).')
    parser.add_argument('--run_name', default='default', help='name of the training run')
    parser.add_argument('--experiment-name', help='name of the experiment')
    parser.add_argument('--input-size', type=int, nargs=2, default=[150,150], help='shape to which images are resized for training: (h, w)')
    parser.add_argument('--model', default='resnet50', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.003, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr_step_size', default=10, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr_schedule', default='cosine', choices=['cosine','step'],
                        help='LR schedule: cosine (default, recommended) or step')
    parser.add_argument('--lr_gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--run-uuid', dest="run_uuid", default =None, help='')

    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )

    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        #action="store_true",
        default=True
    )
    # distributed training parameters
    parser.add_argument('--distributed', default=False)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--log_model', action='store_true', dest="log_model", help='tore final model in mlflow')
    parser.add_argument('--no_log_model', action='store_false', dest="log_model", help='do not store final model in mlflow')
    parser.add_argument('--log_roc', action='store_true', dest="log_roc", help='store final roc in mlflow')
    parser.add_argument('--no_log_roc', action='store_false', dest="log_roc", help='do not store final roc in mlflow')
    parser.set_defaults(log_model=True)
    parser.set_defaults(log_roc=True)
    parser.add_argument('--dataset-type', dest="dataset_type", default="binary", help='Dataset classification type. Can be "binary", "multi_class" or "multi_label"')
    parser.add_argument('--class_names', dest="class_names", type=str, nargs="+", default=[], help='Class names in order. I.e. --class_names label0 label1 label2')
    parser.add_argument('--cv','--cross-validation', dest="nr_cv", default=1, type=int, help='Number of cross validation runs')
    parser.add_argument('--cv-start', dest="cv_start", default=0, type=int, help='Number of cross validation runs')
    parser.add_argument("--user_attr", dest="user_attr", action=StoreDictKeyPair, default={}, metavar="KEY1=VAL1,KEY2=VAL2...")
    parser.add_argument("--system_attr", dest="system_attr", action=StoreDictKeyPair, default={}, metavar="KEY1=VAL1,KEY2=VAL2...")
    parser.add_argument("--mlflow_uri", type=str, default=None, help="MLflow tracking URI (file path or http(s)://)")

    
    
    args = parser.parse_args()
    print(f"Log Model: {args.log_model}")
    print(f"Log Roc: {args.log_roc}")
    return args


if __name__ == "__main__":
    args = parse_args()
    run_uuid = args.run_uuid
       
    metric = 0
    user_attr = {}
    system_attr = {}
    for i in range(args.cv_start, args.nr_cv): 
        print(f"START CV {i}")
        run_name = f"{args.run_name}_cv{i}"
        metric += main(args=args, cv=i, run_name=run_name, tags=None, run_uuid=args.run_uuid)
        print(f"END CV {i}")
         
    print(metric / (args.nr_cv - args.cv_start))