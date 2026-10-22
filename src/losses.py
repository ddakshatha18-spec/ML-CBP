import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):

    def __init__(
        self,
        num_classes=6,
        ignore_index=255
    ):
        super().__init__()

        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, logits, targets):

        probs = F.softmax(
            logits,
            dim=1
        )

        valid = targets != self.ignore_index

        safe_targets = targets.clone()
        safe_targets[~valid] = 0

        one_hot = F.one_hot(
            safe_targets,
            num_classes=self.num_classes
        )

        one_hot = one_hot.permute(
            0, 3, 1, 2
        ).float()

        valid = valid.unsqueeze(1)

        probs = probs * valid
        one_hot = one_hot * valid

        intersection = (
            probs * one_hot
        ).sum(dim=(0, 2, 3))

        denominator = (
            probs.sum(dim=(0, 2, 3))
            + one_hot.sum(dim=(0, 2, 3))
        )

        dice = (
            2 * intersection + 1e-6
        ) / (
            denominator + 1e-6
        )

        return 1 - dice.mean()


class CombinedLoss(nn.Module):

    def __init__(
        self,
        num_classes=6,
        ignore_index=255,
        class_weights=None
    ):
        super().__init__()

        self.cross_entropy = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index
        )

        self.dice = DiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index
        )

    def forward(self, logits, targets):

        ce_loss = self.cross_entropy(
            logits,
            targets
        )

        dice_loss = self.dice(
            logits,
            targets
        )

        return ce_loss + dice_loss