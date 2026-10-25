# Wound Tissue Segmentation Model: Review Package

**Research prototype only. Not a medical device. Not for treatment decisions.**

This package contains the trained model for wound tissue segmentation and everything needed to run and check it.

## What the model does

Input: one wound photograph. Output: a mask that assigns each pixel to one of six classes.

| ID | Class |
|----|-------|
| 0 | Background |
| 1 | Epithelial |
| 2 | Slough |
| 3 | Granulation |
| 4 | Necrotic |
| 5 | Other |

Architecture: U-Net with a ResNet34 encoder (`segmentation_models_pytorch`), 6 output classes. The encoder was initialised from ImageNet weights before training; no download is needed to run the model.

Training: LUTSeg dataset, patient-level split (111 training images, 30 validation images). Input 384 x 384, batch size 2, 30 epochs, Adam, learning rate 0.001, Cross-Entropy + Dice loss. Class weights apply to the Cross-Entropy term only: `[0.0815, 0.3712, 0.4980, 0.3763, 1.1845, 3.4885]` for Background, Epithelial, Slough, Granulation, Necrotic, Other. The checkpoint with the best validation Dice was kept.

The weights in this package come from the seed-42 run. Of three seeds, it is the strongest one on tissue-only Dice, so use the 3-seed averages below as the representative figures.

## Files

```
weights/resnet34_unet_weighted_seed42.pth   model weights (tensors only)
run_inference.py                            runs the model on an image or a folder
requirements.txt                            Python packages
SHA256SUMS.txt                              checksum of the weights file
results/                                    result tables and figures
README_FACULTY.md                           this file
```

Full training and evaluation code, with the exact scripts used for every number below: branch `p3` of `https://github.com/ddakshatha18-spec/ML-CBP` (see `README_segmentation.md` there). The repository is not public unless you have been given access.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_inference.py --input path/to/image_or_folder --output out
```

For each image the script writes `<name>_mask.png` (pixel values are the class ids 0 to 5), `<name>_overlay.png` (colour overlay) and one `summary.csv` with the wound area and the share of each tissue.

Colours in the overlay: Epithelial blue, Slough yellow, Granulation green, Necrotic red, Other purple.

Wound area is a percentage of the image, and tissue shares are percentages of the predicted wound pixels. There is no physical scale, so these are not cm^2 values.

The script uses the input size 256 x 256 that was used for all reported results, and it upsamples the mask back to the original image size. It does not apply EXIF rotation, to match how the dataset masks were annotated. Check that the weights file is intact:

```bash
shasum -a 256 -c SHA256SUMS.txt
```

The weights load with `torch.load(..., weights_only=True)`, so the file holds tensors only.

## Results (30 validation images, 256 x 256)

Mean +/- standard deviation over 3 training runs (different seeds), Dice score.

| Class | Baseline U-Net | This model |
|---|---|---|
| Background | 0.9527 +/- 0.0028 | 0.9565 +/- 0.0025 |
| Epithelial | 0.2849 +/- 0.0608 | 0.3572 +/- 0.0265 |
| Slough | 0.2621 +/- 0.0246 | 0.3185 +/- 0.0248 |
| Granulation | 0.5970 +/- 0.0082 | 0.6715 +/- 0.0183 |
| Necrotic | 0.1850 +/- 0.1427 | 0.3281 +/- 0.1242 |
| Other | 0.0000 | 0.0000 |
| **Tissue-only mean Dice** (5 non-background classes) | 0.2658 +/- 0.0247 | **0.3351 +/- 0.0283** |
| Six-class mean Dice | 0.3803 +/- 0.0204 | 0.4386 +/- 0.0232 |
| Six-class mean IoU | 0.2933 +/- 0.0119 | 0.3383 +/- 0.0167 |

Binary wound Dice (finding the wound region at all), seed 42 only: baseline 0.7385, this model 0.7672. The seed-42 weights in this package give a six-class mean Dice of 0.4539 and a mean IoU of 0.3497.

## Limitations

- The validation set has only 30 images from 5 patients, so the results are preliminary. There is no separate test set. The checkpoint was chosen on the same 30 images that are reported, which is mildly optimistic.
- Necrotic varies a lot between runs (Dice 0.19 to 0.42). The class `Other` is never predicted (about 4,900 training pixels).
- The gain over the baseline comes from two changes together (pretrained encoder and class weights). They were not separated by multi-seed ablation.
- No scale calibration, so no real wound sizes.
- Not clinically validated. Do not use for diagnosis or treatment decisions.

## Data and attribution

Trained on LUTSeg (Hugging Face: `ksanchez84/LUTSeg`, licence CC BY 4.0). The dataset is not included in this package; please download it from the source and cite the dataset authors.
