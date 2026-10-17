import torch


def dice_score(logits, targets, num_classes=6, ignore_index=255):
    predictions = torch.argmax(logits, dim=1)

    scores = []

    for class_id in range(num_classes):

        valid = targets != ignore_index

        pred_class = (predictions == class_id) & valid
        target_class = (targets == class_id) & valid

        intersection = (pred_class & target_class).sum().float()

        denominator = (
            pred_class.sum().float()
            + target_class.sum().float()
        )

        if denominator == 0:
            continue

        dice = (2 * intersection) / (denominator + 1e-6)

        scores.append(dice.item())

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def iou_score(logits, targets, num_classes=6, ignore_index=255):
    predictions = torch.argmax(logits, dim=1)

    scores = []

    for class_id in range(num_classes):

        valid = targets != ignore_index

        pred_class = (predictions == class_id) & valid
        target_class = (targets == class_id) & valid

        intersection = (pred_class & target_class).sum().float()

        union = (
            pred_class.sum().float()
            + target_class.sum().float()
            - intersection
        )

        if union == 0:
            continue

        iou = intersection / (union + 1e-6)

        scores.append(iou.item())

    if not scores:
        return 0.0

    return sum(scores) / len(scores)