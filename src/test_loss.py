import torch

from model import UNet
from losses import CombinedLoss
from metrics import dice_score, iou_score


# Create model
model = UNet(
    in_channels=3,
    num_classes=6
)

# Fake input
images = torch.randn(2, 3, 256, 256)

# Fake segmentation masks
masks = torch.randint(
    0,
    6,
    (2, 256, 256)
)

# Forward pass
outputs = model(images)

# Loss
loss_function = CombinedLoss(num_classes=6)

loss = loss_function(outputs, masks)

# Metrics
dice = dice_score(outputs, masks)
iou = iou_score(outputs, masks)

print("Loss:", loss.item())
print("Dice:", dice)
print("IoU :", iou)