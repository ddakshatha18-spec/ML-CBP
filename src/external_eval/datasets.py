"""Loaders for DFUTissue and ComplexWoundDB (torch-free; PIL + numpy only).

DFUTissue (uwm-bigdata/DFUTissueSegNet, ``DFUTissue/``)
  * ``Labeled/Original`` is the native image size (small square crops, 58 distinct sizes in the
    110 labelled images). ``Labeled/Padded`` is the same crop pasted onto a black 256 x 256 canvas,
    which raises background from ~78% to ~92% of pixels. TiSage's split files use ``Original``,
    so ``Original`` is the default here.
  * Masks are RGB PNGs whose three channels hold the same class id: 0 background, 1 fibrin,
    2 granulation, 3 callus (``palette_colorCode.txt``: Red - Fibrin, Green - Granulation, Blue - Callus).
  * Splits: ``labeled_train_names.txt`` (78), ``labeled_val_names.txt`` (16) live in ``TrainVal/``;
    ``test_names.txt`` (16) lives in ``Test/``. The files use CRLF line endings.

ComplexWoundDB (recogna-lab/datasets, ``ComplexWoundDB/``)
  * 27 images (PNG), masks per expert in ``annotations/masks/expert_N``. Expert 4 is the one the
    dataset authors used as ground truth; experts 1 to 3 contain thousands of interpolated colours.
  * Colour code (expert 4): white = non-wound, red = granulation, yellow = fibrinoid (slough),
    black = dry necrosis, purple = hematoma. Expert 4 also uses a second yellow, (253, 231, 36), in 11 of
    27 masks. Its meaning is NOT documented; by default it is treated as fibrinoid (nearest colour)
    and ``yellow2="ignore"`` excludes it, so the effect can be measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from . import IGNORE

# ----------------------------------------------------------------------------------------------
# DFUTissue
# ----------------------------------------------------------------------------------------------
DFU_CLASSES = ["background", "fibrin", "granulation", "callus"]  # native ids 0..3 (TiSage util/classes.py)


@dataclass(frozen=True)
class Item:
    """One labelled image. ``mask_path`` is None when the loader builds the mask itself."""

    id: str
    image_path: Path
    mask_path: Path
    group: str = ""  # e.g. "train" / "val" / "test" / expert name


def _read_ids(path: Path) -> List[str]:
    # The split files are CRLF; splitlines() + strip() handles both.
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def dfutissue_items(root, split: str, version: str = "Original") -> List[Item]:
    """``split`` is train (78), val (16), test (16), trainval (94) or all (110)."""
    root = Path(root)
    labeled = root / "Labeled"
    if not labeled.is_dir():
        raise FileNotFoundError(f"DFUTissue 'Labeled' folder not found under {root}")
    if version not in ("Original", "Padded"):
        raise ValueError("version must be 'Original' or 'Padded'")

    names = {
        "train": _read_ids(labeled / "labeled_train_names.txt"),
        "val": _read_ids(labeled / "labeled_val_names.txt"),
        "test": _read_ids(labeled / "test_names.txt"),
    }

    def make(ids: List[str], folder: str, group: str) -> List[Item]:
        out = []
        for i in ids:
            img = labeled / version / "Images" / folder / f"{i}.png"
            msk = labeled / version / "Annotations" / folder / f"{i}.png"
            if not img.is_file() or not msk.is_file():
                raise FileNotFoundError(f"Missing DFUTissue file for id {i}: {img} / {msk}")
            out.append(Item(i, img, msk, group))
        return out

    parts = {
        "train": lambda: make(names["train"], "TrainVal", "train"),
        "val": lambda: make(names["val"], "TrainVal", "val"),
        "test": lambda: make(names["test"], "Test", "test"),
    }
    if split in parts:
        return parts[split]()
    if split == "trainval":
        return parts["train"]() + parts["val"]()
    if split == "all":
        return parts["train"]() + parts["val"]() + parts["test"]()
    raise ValueError(f"Unknown DFUTissue split '{split}'")


def load_dfutissue_mask(path) -> np.ndarray:
    """Class-id mask (H, W) uint8 from the RGB annotation PNG (channels must agree)."""
    a = np.array(Image.open(path))
    if a.ndim == 3:
        if not (np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 1], a[..., 2])):
            raise ValueError(f"DFUTissue mask channels differ (not a class-id mask): {path}")
        a = a[..., 0]
    return a.astype(np.uint8)


# ----------------------------------------------------------------------------------------------
# ComplexWoundDB
# ----------------------------------------------------------------------------------------------
CWDB_CLASSES = ["background", "granulation", "fibrinoid", "necrosis", "hematoma"]  # native ids 0..4

# (name, RGB, native id). yellow2 is the undocumented second yellow of expert 4.
CWDB_PALETTE = [
    ("white", (255, 255, 255), 0),
    ("red", (237, 28, 36), 1),
    ("yellow", (255, 242, 0), 2),
    ("yellow2", (253, 231, 36), 2),
    ("black", (0, 0, 0), 3),
    ("purple", (163, 35, 142), 4),
]


def snap_to_palette(rgb: np.ndarray, palette: List[Tuple[str, Tuple[int, int, int], int]],
                    max_dist: Optional[float] = None) -> np.ndarray:
    """Assign every pixel the class id of the nearest palette colour (Euclidean RGB distance).

    Works on the unique colours only, so it is fast even for blended masks. With ``max_dist`` set,
    pixels farther than that from every palette colour become IGNORE (255).
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape (H, W, 3)")
    flat = rgb.reshape(-1, 3)
    uniq, inverse = np.unique(flat, axis=0, return_inverse=True)
    pal = np.array([c for _, c, _ in palette], dtype=np.float64)
    ids = np.array([i for _, _, i in palette], dtype=np.int64)
    d = np.linalg.norm(uniq[:, None, :].astype(np.float64) - pal[None, :, :], axis=2)
    nearest = d.argmin(axis=1)
    out_ids = ids[nearest].astype(np.int64)
    if max_dist is not None:
        out_ids[d.min(axis=1) > max_dist] = IGNORE
    return out_ids[inverse.reshape(-1)].reshape(rgb.shape[:2]).astype(np.uint8)


def cwdb_items(root, expert: int = 4) -> List[Item]:
    root = Path(root)
    img_dir = root / "images"
    mask_dir = root / "annotations" / "masks" / f"expert_{expert}"
    if not img_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"ComplexWoundDB folders not found under {root} (expert {expert})")
    items = []
    for m in sorted(mask_dir.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else 10**9):
        img = img_dir / m.name
        if not img.is_file():
            raise FileNotFoundError(f"Missing ComplexWoundDB image for mask {m}")
        items.append(Item(m.stem, img, m, f"expert_{expert}"))
    if not items:
        raise ValueError(f"No masks in {mask_dir}")
    return items


def load_cwdb_mask(path, yellow2: str = "fibrinoid", max_dist: Optional[float] = 40.0) -> np.ndarray:
    """Native-id mask (0 bg, 1 granulation, 2 fibrinoid, 3 necrosis, 4 hematoma, 255 ignore).

    ``yellow2``: "fibrinoid" (default, nearest-colour result) or "ignore".
    PNG masks are RGBA; only RGB is used (alpha is 255 everywhere in expert 4).
    """
    if yellow2 not in ("fibrinoid", "ignore"):
        raise ValueError("yellow2 must be 'fibrinoid' or 'ignore'")
    rgb = np.array(Image.open(path).convert("RGB"))
    palette = [(n, c, (IGNORE if (n == "yellow2" and yellow2 == "ignore") else i)) for n, c, i in CWDB_PALETTE]
    mask = snap_to_palette(rgb, palette, max_dist=max_dist)
    return mask


def load_image_rgb(path) -> Image.Image:
    """RGB image on the raw pixel grid. No EXIF rotation (masks are annotated on the raw grid)."""
    img = Image.open(path)
    img.load()
    return img.convert("RGB")
