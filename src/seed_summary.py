import pandas as pd
paths = {
    "seed42": "outputs/final_evaluation_resnet34_weighted/class_metrics.csv",
    "seed1": "outputs/final_evaluation_r34w_s1/class_metrics.csv",
    "seed2": "outputs/final_evaluation_r34w_s2/class_metrics.csv",
}
d = pd.DataFrame({k: pd.read_csv(v).set_index("Class")["Dice"] for k, v in paths.items()})
d["mean"] = d[list(paths)].mean(axis=1)
d["std"] = d[list(paths)].std(axis=1)
print(d.round(4))
t = d.drop("Background")[list(paths)].mean()
print("\nTissue-only mean Dice per seed:\n", t.round(4))
print(f"\nTissue-only mean Dice: {t.mean():.4f} +/- {t.std():.4f}")
a = d[list(paths)].mean()
print(f"Six-class mean Dice:   {a.mean():.4f} +/- {a.std():.4f}")
d.round(4).to_csv("outputs/comparison/seed_summary.csv")
