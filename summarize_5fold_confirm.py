from pathlib import Path
import json
import pandas as pd

project_root = Path("D:/FGDN_Project")
base = project_root / "outputs" / "ablations" / "AAL_5fold_confirm"

rows = []

for candidate_dir in sorted(base.iterdir()):
    if not candidate_dir.is_dir():
        continue

    candidate = candidate_dir.name

    for fold_dir in sorted(candidate_dir.glob("fold_*")):
        if not fold_dir.is_dir():
            continue

        json_files = list(
            fold_dir.glob("tables/fgdn/AAL/5_fold/fold_*/best_evaluation.json")
        )

        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            rows.append({
                "candidate": candidate,
                "fold": data.get("fold"),
                "accuracy": data.get("accuracy", data.get("acc")),
                "auc": data.get("auc"),
                "best_monitor_auc_from_training": data.get("best_monitor_auc_from_training"),
                "training_epoch_saved": data.get("training_epoch_saved"),
                "confusion_matrix": data.get("confusion_matrix"),
                "metric_file": str(jf),
            })

df = pd.DataFrame(rows)

out_dir = project_root / "outputs" / "ablations" / "summaries"
out_dir.mkdir(parents=True, exist_ok=True)

detail_path = out_dir / "aal_5fold_confirm_details.csv"
summary_path = out_dir / "aal_5fold_confirm_summary.csv"
md_path = out_dir / "aal_5fold_confirm_summary.md"

df = df.sort_values(["candidate", "fold"])
df.to_csv(detail_path, index=False)

summary = (
    df.groupby("candidate")
    .agg(
        n_folds=("fold", "count"),
        acc_mean=("accuracy", "mean"),
        acc_std=("accuracy", "std"),
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
        acc_min=("accuracy", "min"),
        acc_max=("accuracy", "max"),
        auc_min=("auc", "min"),
        auc_max=("auc", "max"),
    )
    .reset_index()
    .sort_values("auc_mean", ascending=False)
)

summary.to_csv(summary_path, index=False)
summary.to_markdown(md_path, index=False)

print(f"Wrote: {detail_path}")
print(f"Wrote: {summary_path}")
print(f"Wrote: {md_path}")
print()
print(summary.to_string(index=False))