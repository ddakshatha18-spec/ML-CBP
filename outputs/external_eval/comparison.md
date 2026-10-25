# Comparison with the base paper (TiSage)

Research prototype. Small test sets (16 and 30 images); all numbers preliminary.

## DFUTissue (4 classes: background, fibrin, granulation, callus), 16 test images

| Model | mIoU | Dice |
|---|---|---|
| DeepLabV3+-R50 (paper) | 70.02 | 81.12 |
| DINOv2-DPT (paper) | 68.71 | 80.21 |
| FixMatch (paper, +600 unlabeled) | 68.91 | 80.19 |
| UniMatch-V2 (paper, +600 unlabeled) | 69.94 | 80.96 |
| TiSage (paper, +600 unlabeled) | 72.36 | 83.05 |
| Ours, clean protocol (train 78, epoch chosen on val 16, tested on test 16) - best epoch | 57.60 ± 1.65 (n=3) | 69.35 ± 1.42 (n=3) |
| Ours, clean protocol (train 78, epoch chosen on val 16, tested on test 16) - last epoch (not selected) | 55.21 ± 5.34 (n=3) | 67.02 ± 4.81 (n=3) |
| Ours, paper protocol (train 94, epoch chosen ON TEST: optimistic) - best epoch | 60.97 ± 2.10 (n=3) | 72.95 ± 1.95 (n=3) |
| Ours, paper protocol (train 94, epoch chosen ON TEST: optimistic) - last epoch (not selected) | 55.55 ± 8.39 (n=3) | 67.33 ± 8.20 (n=3) |

## LUTSeg (6 classes incl. background), 30 validation images

| Model | mIoU | Dice |
|---|---|---|
| DeepLabV3+-R50 (paper) | 22.06 | 25.61 |
| DINOv2-DPT (paper) | 31.37 | 39.19 |
| Ours ResNet34 + weights, scored at 256 x 256 | 33.83 ± 1.67 (n=3) | 43.86 ± 2.32 (n=3) |
| Ours ResNet34 + weights, scored at native resolution (TiSage style) | 28.30 ± 1.43 (n=3) | 35.74 ± 2.43 (n=3) |

## Zero-shot (LUTSeg model, no retraining) - NOT comparable with the paper

| Dataset | variant | mIoU | Dice | binary wound Dice |
|---|---|---|---|---|
| cwdb_expert4_yellow2-fibrinoid | default|strict | 35.34 | 43.57 | 62.01 |
| cwdb_expert4_yellow2-fibrinoid | default|lenient | 35.54 | 43.67 | 64.16 |
| cwdb_expert4_yellow2-ignore | default|strict | 35.64 | 43.93 | 61.07 |
| cwdb_expert4_yellow2-ignore | default|lenient | 35.84 | 44.04 | 63.44 |
| dfutissue_test_Original | callus_ignore|strict | 66.38 | 77.59 | 82.86 |
| dfutissue_test_Original | callus_ignore|lenient | 66.81 | 77.82 | 83.84 |
| dfutissue_test_Original | callus_background|strict | 60.83 | 71.87 | 74.32 |
| dfutissue_test_Original | callus_background|lenient | 62.07 | 72.55 | 79.89 |

## What is and is not comparable

- Paper models are trained on all labelled training images and the best epoch is picked on the same split that is reported. Only our 'paper protocol' rows copy that; the 'clean' rows do not, so expect them to be lower.
- Paper and ours both average over all classes including background; classes absent from the data count as 0.
- Our model sees 256 x 256 inputs; the paper's sees native resolution.
- Zero-shot numbers depend on the class mapping (Slough = fibrin/fibrinoid is an assumption) and on ComplexWoundDB's coarse, partly undocumented expert-4 labels.
- Not clinically validated.
