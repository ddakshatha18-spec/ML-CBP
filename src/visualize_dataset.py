import matplotlib.pyplot as plt

from dataset import LUTSegDataset


dataset = LUTSegDataset(
    "data/LUTSeg",
    split="train"
)

image, mask = dataset[0]

# Undo normalization so the image displays correctly
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

image = image.permute(1, 2, 0).numpy()

for i in range(3):
    image[:, :, i] = image[:, :, i] * std[i] + mean[i]

image = image.clip(0, 1)

mask = mask.numpy()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Wound Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask)
plt.title("Segmentation Mask")
plt.axis("off")

plt.tight_layout()
plt.show()