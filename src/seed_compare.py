import pandas as pd
groups = {
    "Baseline": ["outputs/final_evaluation/class_metrics.csv",
                 "outputs/final_evaluation_base_s1/class_metrics.csv",
                 "outputs/final_evaluation_base_s2/class_metrics.csv"],
    "ResNet34+weights": ["outputs/final_evaluation_resnet34_weighted/class_metrics.csv",
                         "outputs/final_evaluation_r34w_s1/class_metrics.csv",
                         "outputs/final_evaluation_r34w_s2/class_metrics.csv"],
}
rows, tissue, six = {}, {}, {}
for g, paths in groups.items():
    d = pd.concat([pd.read_csv(p).set_index("Class")["Dice"] for p in paths], axis=1)
    rows[g + " mean"] = d.mean(axis=1)
    rows[g + " std"] = d.std(axis=1)
    tissue[g] = d.drop("Background").mean()
    six[g] = d.mean()
print(pd.DataFrame(rows).round(4))
for g in groups:
    print(f"\n{g}: tissue-only {tissue[g].mean():.4f} +/- {tissue[g].std():.4f} | six-class {six[g].mean():.4f} +/- {six[g].std():.4f}")
    print("  per run tissue-only:", tissue[g].round(4).tolist())
pd.DataFrame(rows).round(4).to_csv("outputs/comparison/seed_compare.csv")
