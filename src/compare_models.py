import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs = {
    "Baseline": "outputs/final_evaluation/class_metrics.csv",
    "Weighted CE": "outputs/final_evaluation_weighted/class_metrics.csv",
    "ResNet34": "outputs/final_evaluation_resnet34/class_metrics.csv",
    "ResNet34 + weights": "outputs/final_evaluation_resnet34_weighted/class_metrics.csv",
}

dice, iou = {}, {}
for name, path in runs.items():
    df = pd.read_csv(path).set_index("Class")
    dice[name] = df["Dice"]
    iou[name] = df["IoU"]

dice = pd.DataFrame(dice)
iou = pd.DataFrame(iou)

summary = pd.DataFrame({
    "Mean Dice (6 classes)": dice.mean(),
    "Mean Dice (tissue only, no Background)": dice.drop("Background").mean(),
    "Mean IoU (6 classes)": iou.mean(),
    "Mean IoU (tissue only, no Background)": iou.drop("Background").mean(),
}).T

dice.round(4).to_csv("outputs/comparison/per_class_dice.csv")
iou.round(4).to_csv("outputs/comparison/per_class_iou.csv")
summary.round(4).to_csv("outputs/comparison/summary.csv")

print("\nPER-CLASS DICE\n", dice.round(4))
print("\nPER-CLASS IoU\n", iou.round(4))
print("\nSUMMARY\n", summary.round(4))

ax = dice.plot(kind="bar", figsize=(10, 5), width=0.8)
ax.set_ylabel("Dice")
ax.set_title("Per-class Dice: model comparison (30 validation images)")
ax.set_ylim(0, 1)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("outputs/comparison/per_class_dice.png", dpi=200)
print("\nSaved to outputs/comparison/")
