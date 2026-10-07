from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
import random


class LUTSegDataset(Dataset):

    def __init__(
        self,
        root_dir,
        split="train",
        image_size=(256, 256),
        augment=False
    ):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.augment = augment

        split_file = self.root_dir / f"{split}.txt"

        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}"
            )

        self.samples = []

        for line in split_file.read_text().splitlines():

            line = line.strip()

            if not line:
                continue

            image_path, mask_path = line.split()

            self.samples.append(
                (
                    self.root_dir / image_path,
                    self.root_dir / mask_path,
                )
            )

        if not self.samples:
            raise ValueError(
                f"No samples found in {split_file}"
            )

        print(f"{split}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, mask_path = self.samples[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        mask = Image.open(
            mask_path
        ).convert("L")

        # -----------------------------
        # Resize
        # -----------------------------

        image = TF.resize(
            image,
            self.image_size,
            interpolation=InterpolationMode.BILINEAR
        )

        mask = TF.resize(
            mask,
            self.image_size,
            interpolation=InterpolationMode.NEAREST
        )

        # -----------------------------
        # Data augmentation
        # -----------------------------

        if self.augment:

            # Horizontal flip
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Vertical flip
            if random.random() < 0.2:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # Small rotation
            if random.random() < 0.5:

                angle = random.uniform(-15, 15)

                image = TF.rotate(
                    image,
                    angle,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0
                )

                mask = TF.rotate(
                    mask,
                    angle,
                    interpolation=InterpolationMode.NEAREST,
                    fill=255
                )

            # Brightness/contrast changes
            if random.random() < 0.5:

                brightness_factor = random.uniform(
                    0.9,
                    1.1
                )

                contrast_factor = random.uniform(
                    0.9,
                    1.1
                )

                image = TF.adjust_brightness(
                    image,
                    brightness_factor
                )

                image = TF.adjust_contrast(
                    image,
                    contrast_factor
                )

        # -----------------------------
        # Convert image
        # -----------------------------

        image = TF.to_tensor(image)

        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # -----------------------------
        # Convert mask
        # -----------------------------

        mask = torch.from_numpy(
            np.array(mask)
        ).long()

        return image, mask