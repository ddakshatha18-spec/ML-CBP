# ML-CBP: Wound Tissue Segmentation

Pixel-level segmentation of wound photographs into tissue types using U-Net models, with a controlled comparison of a plain U-Net baseline, class-weighted loss, and a pretrained ResNet34 encoder.

> **Research prototype only.** This is not a clinically validated system. Its output must not be used to make diagnostic or treatment decisions.

## Task

Input: one wound photograph. Output: a mask assigning every pixel to one of six classes.

| ID | Class |
|----|-------|
| 0 | Background |
| 1 | Epithelial |
| 2 | Slough |
| 3 | Granulation |
| 4 | Necrotic |
| 5 | Other |

Pixels labelled `255` are ignored in the loss and in evaluation (none occurred in the training masks).

Multi-visit healing-trajectory analysis is a separate, later stage and is **not** part of this repository's results.

## Dataset

LUTSeg (Hugging Face: `ksanchez84/LUTSeg`), stored locally in `data/LUTSeg/` with the supplied patient-level split:

- 111 training images (`train.txt`), 30 validation images (`val.txt`)
- Strong class imbalance in the training pixels: Background 88.72%, Epithelial 4.28%, Slough 2.38%, Granulation 4.17%, Necrotic 0.42%, Other 0.03%

Check the dataset's license and terms before redistributing any of it. The dataset is not included in this repository.

## Models and experiments

All runs use the same training setup unless stated: 384 x 384 inputs, batch size 2, 30 epochs, Adam, learning rate 0.001, Cross-Entropy + Dice loss, flip / rotation / brightness-contrast augmentation, best checkpoint chosen by validation Dice.

| Name | Model | Class weights | Training script |
|------|-------|---------------|-----------------|
| Baseline | custom U-Net (`src/model.py`) | none | `src/train.py` |
| Weighted CE | custom U-Net | yes | `src/train_weighted.py` |
| ResNet34 | `smp.Unet`, ImageNet-pretrained ResNet34 | none | `src/train_resnet34.py` |
| ResNet34 + weights | `smp.Unet`, ImageNet-pretrained ResNet34 | yes | `src/train_resnet34_weighted.py` |

Class weights apply only to the Cross-Entropy term (capped inverse-square-root frequency, hand-chosen, not tuned):
`[0.0815, 0.3712, 0.4980, 0.3763, 1.1845, 3.4885]` for Background, Epithelial, Slough, Granulation, Necrotic, Other.

## Evaluation protocol

Every checkpoint is evaluated by a copy of `src/final_evaluation.py` with the same settings: the 30 validation images at 256 x 256, a global confusion matrix, per-class Dice and IoU. Numbers from the training-time validation printout (384 x 384, batch-averaged) are **not** comparable and are not reported below.

"Tissue-only mean Dice" is the mean over the five non-background classes (Epithelial, Slough, Granulation, Necrotic, Other). `Other` is 0 for every model, which lowers this mean.

## Results

### Main comparison: baseline vs ResNet34 + class weights (3 runs each)

Mean ± sample standard deviation over 3 seeds, Dice score.

| Class | Baseline | ResNet34 + weights |
|-------|----------|--------------------|
| Background | 0.9527 ± 0.0028 | 0.9565 ± 0.0025 |
| Epithelial | 0.2849 ± 0.0608 | 0.3572 ± 0.0265 |
| Slough | 0.2621 ± 0.0246 | 0.3185 ± 0.0248 |
| Granulation | 0.5970 ± 0.0082 | 0.6715 ± 0.0183 |
| Necrotic | 0.1850 ± 0.1427 | 0.3281 ± 0.1242 |
| Other | 0.0000 | 0.0000 |
| **Tissue-only mean** | **0.2658 ± 0.0247** | **0.3351 ± 0.0283** |
| **Six-class mean** | **0.3803 ± 0.0204** | **0.4386 ± 0.0232** |

Tissue-only mean Dice per run: Baseline 0.291 / 0.265 / 0.241; ResNet34 + weights 0.354 / 0.303 / 0.349. Every ResNet34 + weights run scored higher than every baseline run.

Reading the table:
- The gains on Granulation, Slough and Epithelial are consistent across seeds.
- Necrotic improves on average, but varies a lot between runs (0.19 to 0.42 for ResNet34 + weights), so treat that class as unstable.
- `Other` is not learned by any model.

### Single-run exploratory comparison (seed 42, 256 x 256)

These configurations were run once. Differences of a point or two are within noise.

| Config | Wound Dice (binary) | Six-class mean Dice | Tissue-only mean Dice | Six-class mean IoU |
|--------|--------------------|---------------------|-----------------------|--------------------|
| Baseline | 0.7385 | 0.4011 | 0.2908 | 0.3063 |
| Weighted CE (U-Net) | 0.6991 | 0.3634 | 0.2475 | 0.2751 |
| ResNet34 | 0.7449 | 0.3854 | 0.2711 | 0.3074 |
| ResNet34 + weights | 0.7672 | 0.4539 | 0.3540 | 0.3497 |

Per-class Dice (seed 42):

| Class | Baseline | Weighted CE | ResNet34 | ResNet34 + weights |
|-------|----------|-------------|----------|--------------------|
| Background | 0.9528 | 0.9433 | 0.9571 | 0.9537 |
| Epithelial | 0.2891 | 0.2947 | 0.2937 | 0.3312 |
| Slough | 0.2903 | 0.1275 | 0.3722 | 0.3307 |
| Granulation | 0.6050 | 0.4789 | 0.6894 | 0.6882 |
| Necrotic | 0.2695 | 0.3362 | 0.0001 | 0.4197 |
| Other | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Observations from these single runs:
- Class weights on the plain U-Net traded Slough and Granulation accuracy for some Necrotic gain, with a net loss.
- ResNet34 alone improved Granulation and Slough but never predicted Necrotic.
- Combining the pretrained encoder and class weights gave the best results. Because each of the other configurations has only one run, the contribution of each component on its own is not established.

### Qualitative results

`outputs/figures/overlays_best_worst.png` shows the four best and four worst validation images, ranked by the ResNet34 + weights model's per-image mean Dice over the classes present in the ground truth. Each row shows the image, the ground truth, and the predictions of all four models.

- Best cases are large wounds on the leg or ankle, with a distinct red wound bed. Even these reach only about 0.52 to 0.57 mean Dice.
- Worst cases are small wounds on the sole of the foot with yellowish skin. The baseline and plain ResNet34 often predict nothing, and the weighted models locate the wound but often label it Slough where the ground truth is Epithelial or Granulation. Likely contributors are the small wound size, callused skin resembling Slough, and unusual lighting or backgrounds; this has not been tested.
- The ResNet34 models tend to draw an Epithelial halo that extends beyond the annotated wound.

## Limitations

- **Small validation set:** 30 images. All results are preliminary.
- **No separate test set:** the best checkpoint is selected on the same 30 images the results are reported on, which is mildly optimistic.
- **Few seeds:** 3 runs for the two main configurations; Weighted CE and ResNet34 alone have 1 run each.
- **Two changes combined:** the ResNet34 + weights gain mixes encoder and loss changes; it was not separated with multi-seed ablations.
- **`Other` class:** about 4,900 training pixels and 462 validation pixels; it is never predicted, so it is a data limitation rather than a tuned result.
- **Necrotic class:** high run-to-run variability.
- **Hand-chosen class weights:** not tuned.
- **Mixed training setups:** the baseline and U-Net runs were trained on CPU, the ResNet34 runs on Apple MPS; the original baseline used `src/train.py`, while baseline seeds 1 and 2 used a copy of the weighted script with weights disabled. Both use the same configuration, but MPS and CPU results can differ slightly.
- **Resolution:** models trained at 384 x 384, evaluated at 256 x 256.
- **Not clinical:** no clinical validation; do not use for treatment decisions.

## Reproducing the results

Environment: Python 3.13, PyTorch, `segmentation-models-pytorch`, `pandas`, `matplotlib`, `pillow`, `numpy`, `tqdm`. Approximate run times on a MacBook M4 Pro (24 GB): about 35 to 40 minutes per 30-epoch U-Net run on CPU, 5 to 10 minutes per ResNet34 run on MPS.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision segmentation-models-pytorch pandas matplotlib pillow numpy tqdm
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Place the dataset in `data/LUTSeg/` with `train.txt` and `val.txt`.

Train the four configurations (seed 42):

```bash
python src/train.py                      # Baseline  -> checkpoints/best_model.pth
python src/train_weighted.py             # Weighted CE
python src/train_resnet34.py             # ResNet34
python src/train_resnet34_weighted.py    # ResNet34 + weights
```

Extra seeds (1 and 2) used copies of the training scripts with only `SEED` and the checkpoint name changed (`train_base_s{1,2}.py`, `train_r34w_s{1,2}.py`).

Evaluate each checkpoint with the same settings (each script differs only in checkpoint path, output folder and model definition):

```bash
python src/final_evaluation.py
python src/final_evaluation_weighted.py
python src/final_evaluation_resnet34.py
python src/final_evaluation_resnet34_weighted.py
python src/eval_base_s1.py && python src/eval_base_s2.py
python src/eval_r34w_s1.py && python src/eval_r34w_s2.py
```

Build the comparison tables and figures:

```bash
python src/compare_models.py     # single-run comparison -> outputs/comparison/
python src/seed_compare.py       # 3-run comparison     -> outputs/comparison/seed_compare.csv
python src/make_overlays.py      # figure               -> outputs/figures/overlays_best_worst.png
```

## Repository notes

Do not commit the dataset, `.venv/`, or the roughly 89 MB checkpoint files unless Git LFS or another storage option is set up. A suggested `.gitignore`:

```
.venv/
data/
checkpoints/*.pth
logs/
__pycache__/
```

## Possible future work

- A held-out test set or external dataset for unbiased evaluation
- Multi-seed ablations separating the encoder from the class weights
- Other loss or weighting strategies, and a more targeted approach for `Other`
- Healing-trajectory analysis across visits, validated separately from segmentation
- An upload-and-segment demo (for example Gradio), with research-only warnings
