# rEPO Doping Detection from Peripheral Blood Smear WSIs

Code accompanying the manuscript *Image-Based Detection of Recombinant
Erythropoietin Use via Weakly Supervised Deep Learning* (under review).

---

## Overview

A deep learning pipeline for detection of recombinant erythropoietin (rEPO)
exposure from whole slide images (WSIs) of peripheral blood smears.

Tiles are extracted from quality-controlled monolayer regions of each slide and
paired with a binary foreground mask, which is concatenated as a fourth input
channel so the network is directed towards cellular content rather than
background. An ImageNet-pretrained convolutional backbone is trained for binary
classification under slide-level stratified group k-fold cross-validation.
Evaluation covers tile- and slide-level discrimination, subgroup stratification
and robustness to image degradation.

---

## Pipeline

```
WSI slides
    |
    v  (1) Monolayer quality control - external, see Haemorasis
    |
    v  (2) Tile extraction from QC output
    |      src/preprocessing/extract_tiles_from_qc.py
    |
    v  (3) Foreground mask generation
    |      src/preprocessing/gen_masks.py
    |
    v  (4) Dataset CSV and stratified group k-fold splits
    |      src/preprocessing/build_dataset_csv.py
    |      src/preprocessing/make_splits.py
    |
    v  (5) Hyperparameter optimisation
    |      src/training/hyper_wada.py
    |      src/training/extract_best_hpo.py
    |
    v  (6) Cross-validated training
    |      src/training/cv_train_mask.py
    |
    v  (7) Evaluation and subgroup analysis
    |      src/evaluation/inference.py
    |      src/evaluation/subgroup_metrics.py
    |
    v  (8) Figures
           src/visualization/publication_figures.py
           src/visualization/noise_robustness.py
           src/visualization/plot_noise_heatmap.py
```

---

## Installation

```bash
git clone https://github.com/Ai-in-Cancer-org/rEPO_doping_detection.git
cd rEPO_doping_detection
pip install -e .
```

Requires Python >= 3.10 and a CUDA-capable GPU for training. OpenSlide system
libraries must be installed separately (`libopenslide-dev` on Debian/Ubuntu).

---

## Step 1 - Monolayer quality control (external)

Quality-controlled monolayer region identification uses an adapted version of
[Haemorasis](https://github.com/josegcpa/haemorasis). Follow its documentation
to install the container and run the quality-control stage over a slide
directory. Full white cell and red cell segmentation is **not** required - only
the quality-control output is consumed here.

The stage produces per-slide score files under `_quality_control/`, in which each
line records a candidate window position and its monolayer probability. These
files are the input to step 2.

---

## Step 2 - Tile extraction

```bash
python src/preprocessing/extract_tiles_from_qc.py \
    --qc_dir         data/qc_output/_quality_control \
    --slides_dir     data/slides \
    --out_root       data/tiles \
    --prob_threshold 1.0 \
    --tile_size      1500
```

Each accepted window position is used to extract a larger region centred on that
position directly from the slide. Tiles are written to
`data/tiles/<slide_id>/tile_NNNNN_probP.PP.png`.

---

## Step 3 - Foreground masks

```bash
python src/preprocessing/gen_masks.py \
    --tile_root    data/tiles \
    --mask_root    data/masks \
    --overlay_root data/overlays
```

Add `--slide <slide_id>` to process a single slide.

---

## Step 4 - Dataset CSV and splits

`configs/slide_labels.csv` is a two-column file you provide:

```
slide_id,label
slide001,0
slide002,1
```

```bash
python src/preprocessing/build_dataset_csv.py \
    --tiles_dir data/tiles \
    --label_csv configs/slide_labels.csv \
    --out_csv   data/metadata/dataset.csv

python src/preprocessing/make_splits.py \
    --csv       data/metadata/dataset.csv \
    --out_dir   data/metadata \
    --prefix    dataset \
    --n_folds   5 \
    --test_fold 4 \
    --seed      42
```

Splits are grouped by `slide_id`, so all tiles from a slide stay within a single
fold.

---

## Step 5 - Hyperparameter optimisation

Workers share an Optuna JournalStorage file and may be launched concurrently,
one per GPU.

```bash
python src/training/hyper_wada.py \
    --data-path   data/metadata/dataset_trainval.csv \
    --imgs-path   data/tiles \
    --test_csv    data/metadata/dataset_test.csv \
    --out_dir     outputs/hpo \
    --worker_id   0 \
    --n_trials_per_worker 13 \
    --input-size  512 512 \
    --epochs      5 \
    -b 8 -j 8 \
    --device cuda
```

Then patch the best configuration into a training script:

```bash
python src/training/extract_best_hpo.py \
    --yaml   outputs/hpo/hpo_trials.yaml \
    --script train.sh
```

---

## Step 6 - Training

```bash
python src/training/cv_train_mask.py \
    --data-path  data/metadata/dataset_trainval.csv \
    --imgs-path  data/tiles \
    --model      resnet34 \
    --lr         0.003 \
    --epochs     50 \
    --cv         5 \
    --output-dir outputs/models \
    --mlflow_uri outputs/mlflow
```

---

## Step 7 - Evaluation

```bash
python src/evaluation/inference.py \
    --ckpt      outputs/models/cv0/model_best.pth \
    --model     resnet34 \
    --csv       data/metadata/dataset_test.csv \
    --imgs_root data/tiles \
    --masks_dir data/masks \
    --out_dir   outputs/results

python src/evaluation/subgroup_metrics.py \
    --preds       outputs/results/tile_predictions.csv \
    --groups_yaml configs/subgroups_example.yaml \
    --out_dir     outputs/results/subgroups
```

---

## Step 8 - Figures

```bash
python src/visualization/publication_figures.py \
    --results_dir outputs/results \
    --noise_csv   outputs/noise/noise_results_all.csv \
    --out_dir     outputs/figures

python src/visualization/noise_robustness.py \
    --ckpt      outputs/models/cv0/model_best.pth \
    --model     resnet34 \
    --csv       all=data/metadata/dataset_test.csv \
    --csv       male=data/metadata/dataset_test_male.csv \
    --csv       female=data/metadata/dataset_test_female.csv \
    --imgs_root data/tiles \
    --masks_dir data/masks \
    --out_dir   outputs/noise

python src/visualization/plot_noise_heatmap.py \
    --csv     outputs/noise/noise_results_all.csv \
    --out_dir outputs/figures
```

`--csv` takes `NAME=PATH` and may be repeated once per cohort.

Run `noise_robustness.py` before `publication_figures.py`, since the latter reads
`noise_results_all.csv` to draw the robustness panel.

---

## Configuration

`configs/config.yaml` holds default paths and hyperparameters.
`configs/subgroups_example.yaml` shows how to declare analysis subgroups by
slide identifier or by identifier prefix.

---

## Notes

- Statistical analysis of haematological parameters reported in the manuscript
  was performed separately and is not part of this repository.
- Job submission scripts are site-specific and are not included; the commands
  above can be wrapped in whatever scheduler your cluster uses.

---

## Citation

Please cite the manuscript once published, and also cite Haemorasis:

```bibtex
@software{haemorasis,
  author = {Almeida, Jose Guilherme Pereira de and others},
  title  = {Haemorasis},
  url    = {https://github.com/josegcpa/haemorasis}
}
```

---

## License

MIT - see `LICENSE`.
