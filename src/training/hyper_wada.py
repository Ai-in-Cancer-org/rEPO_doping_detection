"""
Hyperparameter optimisation with Optuna.

Searches over backbone architecture and SGD hyperparameters using a
tree-structured Parzen estimator sampler. Trials are evaluated on a single
cross-validation fold to maximise the number of configurations explored within
a fixed compute budget. Multiple workers may run concurrently against a shared
Optuna JournalStorage file.

Usage:
    python src/training/hyper_wada.py \
        --data-path  data/metadata/dataset_trainval.csv \
        --imgs-path  data/tiles \
        --test_csv   data/metadata/dataset_test.csv \
        --out_dir    outputs/hpo \
        --worker_id  0 \
        --n_trials_per_worker 13
"""


import argparse
import copy
import gc
import os
import threading
from datetime import datetime
from pathlib import Path
from time import sleep

import mlflow
import numpy as np
import optuna
import torch
import yaml
from optuna.samplers import TPESampler
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import pandas as pd

from cv_train_mask import main as train_main
from cv_train_mask import parse_args as train_parse_args
from cv_train_mask import DatasetError

# ─────────────────────────────────────────────────────────────────────────────
# YAML LOGGER  (thread-safe, all workers append to same file)
# ─────────────────────────────────────────────────────────────────────────────
_YAML_LOCK = threading.Lock()


def append_trial_to_yaml(out_file, trial, args, metrics, fold_used, start_time, end_time):
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "trial_number":  trial.number,
        "worker_id":     int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)),
        "fold_used":     fold_used,
        "start_time":    start_time.isoformat(),
        "end_time":      end_time.isoformat(),
        "duration_sec":  (end_time - start_time).total_seconds(),
        "objective":     metrics.get("composite", None),
        "optuna_params": trial.params,
        "metrics":       metrics,
    }
    with _YAML_LOCK:
        data = yaml.safe_load(out_file.read_text()) if out_file.exists() else []
        data = data or []
        data.append(record)
        out_file.write_text(yaml.safe_dump(data, sort_keys=False))


# ─────────────────────────────────────────────────────────────────────────────
# GPU MEMORY GUARD
# ─────────────────────────────────────────────────────────────────────────────
def get_free_mem(gpu_idx=0):
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(gpu_idx)
    return nvmlDeviceGetMemoryInfo(h).free


def wait_for_memory(min_bytes, gpu_idx=0):
    while get_free_mem(gpu_idx) < min_bytes:
        print(f"[MEM] GPU{gpu_idx} waiting — free={get_free_mem(gpu_idx)/1e9:.2f}GB")
        sleep(30)


# ─────────────────────────────────────────────────────────────────────────────
# PRUNER
# ─────────────────────────────────────────────────────────────────────────────
def make_pruner():
    return optuna.pruners.MedianPruner(
        n_startup_trials=10,  # never prune first 10 trials
        n_warmup_steps=2,     # never prune before epoch 2
        interval_steps=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SEARCH SPACE
# ─────────────────────────────────────────────────────────────────────────────
MODELS = [
    "resnet18", "resnet34", "resnet50", "resnet101",
    "densenet121", "densenet169",
    "resnext50_32x4d", "wide_resnet50_2",
]


MODELS_FULL  = ["resnet34","resnet50","resnet101","resnet152",
                "densenet121","densenet169","resnext50_32x4d","wide_resnet50_2"]
MODELS_SMALL = ["resnet34","resnet50","densenet121","resnet152"]


def sample_hparams(trial, run_name="default"):
    """
    Search space:
      lr            [1e-4, 1e-2] log-uniform
      momentum      [0.88, 0.98] uniform
      weight_decay  [1e-5, 5e-4] log-uniform
      lr_step_size / lr_gamma REMOVED (cosine schedule ignores them)
    """
    # All workers must use identical distributions when sharing a JournalStorage study.
    return dict(
        model        = trial.suggest_categorical("model", MODELS_FULL),
        lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        momentum     = trial.suggest_float("momentum", 0.88, 0.98),
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 5e-4, log=True),
    )


# ─────────────────────────────────────────────────────────────────────────────
# OBJECTIVE
# ─────────────────────────────────────────────────────────────────────────────
def objective(trial, base_args, gpu_idx, yaml_out):
    args = copy.deepcopy(base_args)

    # HPO uses fold 0 only — full 5-fold CV happens in the final training step
    args.nr_cv = 5
    fold_used  = 0
    args.output_dir = str(Path(base_args.output_dir) / "trials" / f"trial_{trial.number}")

    # apply sampled hparams
    hp = sample_hparams(trial, run_name=base_args.experiment_name)
    for k, v in hp.items():
        setattr(args, k, v)
    args.pretrained   = True
    args.lr_schedule  = "cosine"
    args.lr_step_size = 10
    args.lr_gamma     = 0.1

    run_name = f"trial{trial.number}_gpu{gpu_idx}"
    tags = {
        "optuna_trial":        str(trial.number),
        "gpu_idx":             str(gpu_idx),
        "slurm_job_id":        os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "hostname":            os.uname().nodename,
        **{k: str(v) for k, v in hp.items()},
    }

    print(f"\n[HPO] Trial {trial.number} | GPU {gpu_idx} | {hp['model']} | lr={hp['lr']:.5f} | mom={hp['momentum']:.3f}")
    start = datetime.now()

    try:
        _, full_metrics = train_main(
            args=args, cv=fold_used,
            run_name=run_name, tags=tags, run_uuid=None,
        )
    except DatasetError as e:
        print(f"[HPO] DatasetError in trial {trial.number}: {e} — pruning")
        raise optuna.TrialPruned()
    except RuntimeError as e:
        print(f"[HPO] RuntimeError in trial {trial.number}: {e}")
        raise

    end = datetime.now()

    # ── Composite objective ───────────────────────────────────────────────────
    # Score = 0.40×AUROC + 0.30×AP + 0.20×macro_F1 + 0.10×(1-ECE)
    # This prevents gaming: a model with high AUROC but low AP or bad calibration
    # will score lower than a balanced model.
    auroc    = full_metrics.get("val_auroc",    full_metrics.get("val_acc", 0.5))
    ap       = full_metrics.get("val_ap",       full_metrics.get("macro_f1", 0.5))
    macro_f1 = full_metrics.get("macro_f1",     0.5)
    ece      = full_metrics.get("val_ece",       0.1)

    # EPO class (positive class = index 1) F1 — guard against ignoring positives
    # Key names from cv_train_mask: cv0_Recall_repo, cv0_Precision_repo
    cv_pfx   = f"cv{fold_used}_"
    epo_rec  = full_metrics.get(f"{cv_pfx}Recall_repo",    full_metrics.get("macro_recall", 0.5))
    epo_prec = full_metrics.get(f"{cv_pfx}Precision_repo", full_metrics.get("macro_precision", 0.5))
    epo_f1   = 2 * epo_prec * epo_rec / (epo_prec + epo_rec + 1e-9)

    bg_probe  = full_metrics.get(f"{cv_pfx}ProbeAcc_bg_only",    0.5)

    composite = (0.40 * auroc +
                 0.30 * ap   +
                 0.20 * macro_f1 +
                 0.10 * (1.0 - ece))

    print(f"[HPO] Trial {trial.number} composite={composite:.4f} "
          f"(AUROC={auroc:.3f} AP={ap:.3f} F1={macro_f1:.3f} ECE={ece:.3f} "
          f"EPO_F1={epo_f1:.3f} bg_probe={bg_probe:.3f})")

    # ── Hard reject conditions ────────────────────────────────────────────────
    # Stage 1: loose absolute floor for hard rejects (genuinely dead trials)
    reject_reason = None
    if auroc    <  0.65:  reject_reason = f"AUROC={auroc:.3f} < 0.65 (near-random at epoch 50)"
    if bg_probe >= 0.70:  reject_reason = f"bg_probe={bg_probe:.3f} >= 0.70 (staining artefact)"
    if ap       <  0.55:  reject_reason = f"AP={ap:.3f} < 0.55 (collapses at high recall)"
    if epo_f1   <  0.45:  reject_reason = f"EPO F1={epo_f1:.3f} < 0.45 (misses too many positives)"

    # Stage 2: soft flags — logged but NOT hard-rejected
    # Post-HPO analysis uses these to rank trials
    flags = []
    if auroc    >= 0.97:  flags.append("AUROC >= 0.97 (possible overfit to fold 0)")
    if bg_probe >= 0.60:  flags.append(f"bg_probe={bg_probe:.3f} >= 0.60 (borderline artefact)")
    if ap       <  0.65:  flags.append(f"AP={ap:.3f} < 0.65 (precision-recall weakness)")
    if epo_f1   <  0.55:  flags.append(f"EPO F1={epo_f1:.3f} < 0.55 (positive class weakness)")
    if flags:
        print(f"[HPO] Trial {trial.number} FLAGS: {chr(124).join(flags)}")
        full_metrics["flags"] = flags

    if reject_reason:
        print(f"[HPO] Trial {trial.number} HARD REJECT: {reject_reason}")
        full_metrics["reject_reason"] = reject_reason
        full_metrics["composite"]     = float(composite)
        append_trial_to_yaml(yaml_out, trial, args, full_metrics, fold_used, start, end)
        raise optuna.TrialPruned()

    full_metrics["composite"]  = float(composite)
    full_metrics["val_auroc"]  = float(auroc)
    full_metrics["val_ap"]     = float(ap)
    full_metrics["val_ece"]    = float(ece)
    full_metrics["epo_f1"]     = float(epo_f1)
    full_metrics["bg_probe"]   = float(bg_probe)

    append_trial_to_yaml(yaml_out, trial, args, full_metrics, fold_used, start, end)

    trial.report(composite, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    torch.cuda.empty_cache()
    gc.collect()
    return composite


# ─────────────────────────────────────────────────────────────────────────────
# TEST SET EVALUATION  (runs only on worker 0 after all trials complete)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_best_on_test(study, base_args, test_csv, out_dir, device):
    """
    Load the best trial's checkpoint and evaluate on the held-out test set.
    Saves:
      hpo_best_roc.csv         — FPR, TPR, threshold
      hpo_best_pr.csv          — Precision, Recall, threshold
      hpo_results.log          — human-readable + machine-parseable summary
    """
    import torchvision
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, SequentialSampler
    import torchvision.transforms.functional as TF
    from cv_train_mask import (
        CSVDataset, RGBMaskTo4CHTensor,
        reshape_classification_head, patch_first_conv_in_channels,
    )
    from sklearn.metrics import precision_recall_curve

    best = study.best_trial
    hp   = best.params
    print(f"\n[TEST] Best trial: #{best.number}  objective={best.value:.4f}")
    print(f"[TEST] Params: {hp}")

    # Find checkpoint — it lives in the best trial's output dir
    ckpt_dir = Path(base_args.output_dir) / "trials" / f"trial_{best.number}"
    # cv_train_mask saves inside run_name/cv0/
    ckpt_candidates = list(ckpt_dir.rglob("model_best.pth"))
    if not ckpt_candidates:
        print("[TEST] WARNING: no model_best.pth found — skipping test evaluation")
        return {}

    ckpt_path = ckpt_candidates[0]
    print(f"[TEST] Checkpoint: {ckpt_path}")

    # Build model
    model = torchvision.models.__dict__[hp["model"]](pretrained=False)
    model = patch_first_conv_in_channels(model, in_ch=4)
    dummy = argparse.Namespace(model=hp["model"])
    model = reshape_classification_head(model, dummy, ["ctrl", "repo"])
    ckpt  = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()

    # Test loader
    to_tensor = RGBMaskTo4CHTensor(normalize_rgb=True)
    ds = CSVDataset(
        root=base_args.imgs_path,
        path_to_csv=test_csv,
        paired_crop=lambda img, mask: (
            TF.resize(img, base_args.input_size),
            TF.resize(mask, base_args.input_size,
                      interpolation=TF.InterpolationMode.NEAREST),
        ),
        paired_flip=None, mask_dropout=None,
        to_tensor=to_tensor, dataset_type="binary",
        class_names=["ctrl", "repo"],
    )
    loader = DataLoader(ds, batch_size=16, sampler=SequentialSampler(ds),
                        num_workers=4, pin_memory=True)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            p = F.softmax(model(x), dim=1)
            all_probs.append(p.cpu().numpy())
            all_labels.append(y.numpy())

    probs  = np.concatenate(all_probs,  axis=0)
    labels = np.concatenate(all_labels, axis=0)
    scores = probs[:, 1]  # prob of rEPO class

    auroc = roc_auc_score(labels, scores)
    ap    = average_precision_score(labels, scores)
    acc   = float((probs.argmax(axis=1) == labels).mean())
    fpr, tpr, thresh_roc = roc_curve(labels, scores)
    prec, rec, thresh_pr = precision_recall_curve(labels, scores)

    print(f"[TEST] AUROC={auroc:.4f}  AP={ap:.4f}  ACC={acc:.4f}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ROC CSV
    pd.DataFrame({"fpr": fpr, "tpr": tpr,
                  "threshold": np.append(thresh_roc, np.nan)}).to_csv(
        out / "hpo_best_roc.csv", index=False)

    # PR CSV
    pd.DataFrame({"precision": prec, "recall": rec,
                  "threshold": np.append(thresh_pr, np.nan)}).to_csv(
        out / "hpo_best_pr.csv", index=False)

    # results log
    results = {
        "run":            base_args.experiment_name,
        "best_trial":     best.number,
        "hpo_objective":  round(float(best.value), 6),
        "test_auroc":     round(auroc, 6),
        "test_ap":        round(ap, 6),
        "test_acc":       round(acc, 6),
        "n_test_tiles":   len(labels),
        "n_pos":          int(labels.sum()),
        "n_neg":          int((labels == 0).sum()),
        "best_params":    hp,
        "timestamp":      datetime.now().isoformat(),
        "ckpt_path":      str(ckpt_path),
    }

    log_path = out / "hpo_results.log"
    with open(log_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"HPO RESULTS — {base_args.experiment_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Best trial     : #{best.number}\n")
        f.write(f"HPO objective  : {best.value:.4f}  (fold-0 val accuracy)\n")
        f.write(f"Test AUROC     : {auroc:.4f}\n")
        f.write(f"Test AP        : {ap:.4f}\n")
        f.write(f"Test Accuracy  : {acc:.4f}\n")
        f.write(f"Test tiles     : {len(labels)}  (pos={int(labels.sum())}, neg={int((labels==0).sum())})\n\n")
        f.write("Best hyperparameters:\n")
        for k, v in hp.items():
            f.write(f"  {k:<16} {v}\n")
        f.write(f"\nCheckpoint: {ckpt_path}\n")
        f.write(f"ROC curve : {out}/hpo_best_roc.csv\n")
        f.write(f"PR curve  : {out}/hpo_best_pr.csv\n")
        f.write("\n[MACHINE-READABLE]\n")
        f.write(yaml.dump(results, sort_keys=False))

    print(f"[TEST] Results written → {log_path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    # Start from the training arg parser so all training args are inherited
    base = train_parse_args.__wrapped__ if hasattr(train_parse_args, '__wrapped__') else None

    ap = argparse.ArgumentParser(
        description="WADA HPO — parallel Optuna on 8 GPUs via SLURM array",
        parents=[],
    )
    # Training args (subset needed for HPO)
    ap.add_argument("--data-path",        required=True)
    ap.add_argument("--imgs-path",        default="/")
    ap.add_argument("--input-size",       type=int, nargs=2, default=[512, 512])
    ap.add_argument("--epochs",           type=int, default=5)
    ap.add_argument("-b", "--batch-size", type=int, default=8)
    ap.add_argument("-j", "--workers",    type=int, default=8)
    ap.add_argument("--device",           default="cuda")
    ap.add_argument("--output-dir",       required=True)
    ap.add_argument("--experiment-name",  required=True)
    ap.add_argument("--mlflow_uri",       required=True)
    ap.add_argument("--cv",               dest="nr_cv", type=int, default=5)
    ap.add_argument("--class_names",      nargs="+", default=["ctrl", "repo"])
    ap.add_argument("--dataset-type",     dest="dataset_type", default="binary")
    ap.add_argument("--pretrained",       default=True)
    ap.add_argument("--distributed",      default=False)
    ap.add_argument("--sync-bn",          dest="sync_bn", action="store_true")
    ap.add_argument("--log_model",        action="store_true")
    ap.add_argument("--log_roc",          action="store_true")
    ap.add_argument("--run_name",         default="hpo")
    ap.add_argument("--run_uuid",         default=None)
    ap.add_argument("--start-epoch",      type=int, default=0, dest="start_epoch")
    ap.add_argument("--resume",           default="")
    ap.add_argument("--print-freq",       type=int, default=100)
    # HPO-specific
    ap.add_argument("--n_trials_per_worker", type=int, default=13,
                    help="Trials per GPU worker. 8 workers x 13 = 104 total.")
    ap.add_argument("--test_csv",         default=None,
                    help="Path to held-out test set CSV. "
                         "If provided, best trial is evaluated on this after HPO.")
    ap.add_argument("--optuna_log",       default=None,
                    help="Path to shared Optuna JournalStorage file. "
                         "Defaults to <output_dir>/optuna.log")
    return ap.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    # Each SLURM array task gets a unique GPU
    worker_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)
    print(f"\n[HPO] Worker {worker_id} | GPU {worker_id} | "
          f"node {os.uname().nodename}")

    # Shared Optuna storage — all workers read/write the same journal file
    optuna_log = args.optuna_log or str(Path(args.output_dir) / "optuna.log")
    Path(optuna_log).parent.mkdir(parents=True, exist_ok=True)

    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(optuna_log)
    )

    study = optuna.create_study(
        study_name=args.experiment_name,
        direction="maximize",
        storage=storage,
        sampler=TPESampler(n_startup_trials=10, seed=42),
        load_if_exists=True,
        pruner=make_pruner(),
    )

    yaml_out = Path(args.output_dir) / "hpo_trials.yaml"

    # Estimate minimum memory needed
    min_mem = 5694 * args.input_size[0] * args.input_size[1] * (args.batch_size ** 0.25)
    print(f"[MEM] Required: {min_mem/1e9:.2f} GB | "
          f"Available: {get_free_mem(worker_id)/1e9:.2f} GB")
    wait_for_memory(min_mem, gpu_idx=worker_id)

    print(f"[HPO] Starting {args.n_trials_per_worker} trials on GPU {worker_id}")

    study.optimize(
        lambda trial: objective(trial, args, worker_id, yaml_out),
        n_trials=args.n_trials_per_worker,
        gc_after_trial=True,
        show_progress_bar=(worker_id == 0),
    )

    # Only worker 0 does the final test set evaluation and summary
    if worker_id == 0:
        print(f"\n[HPO] All trials done. Best: #{study.best_trial.number} "
              f"objective={study.best_value:.4f}")
        print(f"[HPO] Best params: {study.best_trial.params}")

        # Safety net: warn if best composite is suspiciously low
        if study.best_value < 0.75:
            print()
            print("=" * 60)
            print("  WARNING: Best composite score < 0.75.")
            print("  Recommended actions:")
            print("  1. Check hpo_trials.yaml — look at auroc/ap distributions")
            print("  2. Widen LR range: try [5e-5, 2e-2]")
            print("  3. Increase epochs or reduce model complexity")
            print("=" * 60)

        # Top-5 completed trials summary for manual inspection
        print("\n[HPO] Top 5 trials by composite score:")
        try:
            import yaml as _yaml
            _data = _yaml.safe_load(open(str(yaml_out)).read()) if Path(str(yaml_out)).exists() else []
            _done = [t for t in (_data or []) if t.get("objective") is not None
                     and "reject_reason" not in t.get("metrics", {})]
            _top5 = sorted(_done, key=lambda t: t.get("objective", 0), reverse=True)[:5]
            for i, t in enumerate(_top5):
                m = t.get("metrics", {}); p = t.get("optuna_params", {})
                f = m.get("flags", [])
                print(f"  #{i+1} trial={t['trial_number']:>3}  "
                      f"composite={t['objective']:.4f}  "
                      f"auroc={m.get('val_auroc',0):.3f}  "
                      f"ap={m.get('val_ap',0):.3f}  "
                      f"bg={m.get('bg_probe',0):.3f}  "
                      f"model={p.get('model','?')}"
                      + (f"  [FLAGS: {f}]" if f else ""))
        except Exception as _e:
            print(f"  (could not load YAML: {_e})")

        if args.test_csv and Path(args.test_csv).exists():
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            evaluate_best_on_test(
                study=study,
                base_args=args,
                test_csv=args.test_csv,
                out_dir=args.output_dir,
                device=device,
            )
        else:
            if args.test_csv:
                print(f"[TEST] test_csv not found: {args.test_csv} — skipping")
            else:
                print("[TEST] No --test_csv provided — skipping test evaluation")

        # Print top 5 trials
        print("\n[HPO] Top 5 trials:")
        trials_df = study.trials_dataframe()
        if not trials_df.empty:
            top = trials_df.dropna(subset=["value"]).nlargest(5, "value")
            print(top[["number","value"] + [c for c in top.columns
                                            if c.startswith("params_")]].to_string())
