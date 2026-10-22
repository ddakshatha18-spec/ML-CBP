from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import segmentation_models_pytorch as smp
from model import UNet

ROOT = Path("data/LUTSeg")
SIZE = (256, 256)
NAMES = ["Background", "Epithelial", "Slough", "Granulation", "Necrotic", "Other"]
COLORS = np.array([[0,0,0],[80,160,255],[255,220,50],[60,200,90],[230,50,50],[200,80,220]], dtype=np.uint8)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RANK_BY = "ResNet34 + weights"

MODELS = {
    "Baseline": ("unet", "checkpoints/best_model.pth"),
    "Weighted CE": ("unet", "checkpoints/best_model_weighted.pth"),
    "ResNet34": ("smp", "checkpoints/best_model_resnet34.pth"),
    "ResNet34 + weights": ("smp", "checkpoints/best_model_resnet34_weighted.pth"),
}

def load_model(kind, path):
    if kind == "unet":
        m = UNet(in_channels=3, num_classes=6)
    else:
        m = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=6)
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    m.load_state_dict(state)
    return m.eval()

models = {}
for name, (kind, path) in MODELS.items():
    try:
        models[name] = load_model(kind, path)
        print("loaded:", name)
    except Exception as e:
        print("SKIPPED:", name, "->", type(e).__name__)
assert RANK_BY in models, "ranking model failed to load"

samples = []
for line in (ROOT / "val.txt").read_text().splitlines():
    if line.strip():
        ip, mp = line.split()
        samples.append((ROOT / ip, ROOT / mp))

def overlay(img, mask):
    out = img.astype(np.float32).copy()
    fg = (mask > 0) & (mask < 6)
    out[fg] = 0.5 * out[fg] + 0.5 * COLORS[mask[fg]]
    return out.astype(np.uint8)

def case_score(gt, pred):
    scores = []
    for c in range(1, 6):
        g, p = gt == c, pred == c
        if g.sum() == 0:
            continue
        scores.append(2 * (g & p).sum() / (g.sum() + p.sum()))
    return float(np.mean(scores)) if scores else 1.0

records = []
with torch.no_grad():
    for ip, mp in samples:
        img = Image.open(ip).convert("RGB").resize(SIZE, Image.BILINEAR)
        gt = np.array(Image.open(mp).convert("L").resize(SIZE, Image.NEAREST))
        arr = np.array(img)
        x = torch.from_numpy(((arr / 255.0 - MEAN) / STD).transpose(2, 0, 1).astype(np.float32))[None]
        preds = {n: m(x).argmax(1)[0].numpy() for n, m in models.items()}
        records.append((arr, gt, preds, case_score(gt, preds[RANK_BY])))

records.sort(key=lambda r: r[3])
chosen = records[-4:][::-1] + records[:4]
labels = [f"BEST {i+1}" for i in range(4)] + [f"WORST {i+1}" for i in range(4)]

cols = 2 + len(models)
fig, axes = plt.subplots(len(chosen), cols, figsize=(3 * cols, 3 * len(chosen)))
for r, ((arr, gt, preds, sc), lab) in enumerate(zip(chosen, labels)):
    panels = [("Image", arr), ("Ground truth", overlay(arr, gt))]
    panels += [(n, overlay(arr, p)) for n, p in preds.items()]
    for c, (title, pic) in enumerate(panels):
        ax = axes[r, c]
        ax.imshow(pic)
        ax.axis("off")
        if r == 0:
            ax.set_title(title, fontsize=10)
    axes[r, 0].text(4, 14, f"{lab} ({sc:.2f})", color="white", fontsize=9, backgroundcolor="black")

fig.legend(handles=[Patch(color=COLORS[i] / 255, label=NAMES[i]) for i in range(1, 6)],
           loc="lower center", ncol=5, fontsize=10)
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig("outputs/figures/overlays_best_worst.png", dpi=150)
print("Saved outputs/figures/overlays_best_worst.png")
