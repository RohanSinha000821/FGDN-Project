import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize weighted FGDN cross-validation results."
    )
    parser.add_argument("--project-root", type=str, default="D:/FGDN_Project")
    parser.add_argument("--atlas", type=str, choices=["AAL", "HarvardOxford"], required=True)
    parser.add_argument("--num-folds", type=int, choices=[5, 10], required=True)
    parser.add_argument("--checkpoint-type", type=str, choices=["best", "last"], default="best")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(args.project_root)

    fold_rows = []

    for fold in range(1, args.num_folds + 1):
        eval_path = (
            project_root
            / "outputs"
            / "tables"
            / "fgdn_weighted"
            / args.atlas
            / f"{args.num_folds}_fold"
            / f"fold_{fold}"
            / f"{args.checkpoint_type}_evaluation.json"
        )

        if not eval_path.exists():
            print(f"[WARNING] Missing file for fold {fold}: {eval_path}")
            continue

        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        fold_rows.append(
            {
                "fold": fold,
                "accuracy": float(data["accuracy"]),
                "auc": float(data["auc"]),
            }
        )

    if not fold_rows:
        raise RuntimeError("No weighted evaluation files found.")

    accs = np.array([row["accuracy"] for row in fold_rows], dtype=np.float64)
    aucs = np.array([row["auc"] for row in fold_rows], dtype=np.float64)

    summary = {
        "atlas": args.atlas,
        "num_folds_requested": args.num_folds,
        "num_folds_found": len(fold_rows),
        "checkpoint_type": args.checkpoint_type,
        "model_variant": "fgdn_weighted",
        "fold_results": fold_rows,
        "accuracy_mean": float(accs.mean()),
        "accuracy_std": float(accs.std(ddof=0)),
        "auc_mean": float(aucs.mean()),
        "auc_std": float(aucs.std(ddof=0)),
    }

    out_dir = (
        project_root
        / "outputs"
        / "tables"
        / "fgdn_weighted"
        / args.atlas
        / f"{args.num_folds}_fold"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{args.checkpoint_type}_cv_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("Weighted FGDN CV Summary")
    print("=" * 72)
    print(f"Atlas          : {args.atlas}")
    print(f"Folds found    : {len(fold_rows)}/{args.num_folds}")
    print(f"Accuracy mean  : {summary['accuracy_mean']:.6f}")
    print(f"Accuracy std   : {summary['accuracy_std']:.6f}")
    print(f"AUC mean       : {summary['auc_mean']:.6f}")
    print(f"AUC std        : {summary['auc_std']:.6f}")
    print(f"Saved summary  : {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()