import torch

from model import UNet


# Create model
model = UNet(
    in_channels=3,
    num_classes=6
)

print("Model created successfully!")

# Create a fake batch of 2 images
x = torch.randn(2, 3, 256, 256)

# Run through model
y = model(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)
