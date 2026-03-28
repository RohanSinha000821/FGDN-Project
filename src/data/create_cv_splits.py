import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold


def parse_args():
    parser = argparse.ArgumentParser(description="Create stratified CV splits for FGDN.")
    parser.add_argument(
        "--project-root",
        type=str,
        default="D:/FGDN_Project",
        help="Path to project root",
    )
    parser.add_argument(
        "--atlas",
        type=str,
        choices=["AAL", "HarvardOxford", "all"],
        default="all",
        help="Atlas to process",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[5, 10],
        help="List of fold counts to generate, e.g. --folds 5 10",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def load_labels(project_root: Path, atlas_name: str):
    base_dir = (
        project_root
        / "data"
        / "interim"
        / "connectivity_matrices"
        / atlas_name
        / "tangent"
    )

    labels_path = base_dir / "labels.npy"
    subject_ids_path = base_dir / "subject_ids.npy"

    if not labels_path.exists():
        raise FileNotFoundError(f"labels.npy not found: {labels_path}")
    if not subject_ids_path.exists():
        raise FileNotFoundError(f"subject_ids.npy not found: {subject_ids_path}")

    labels = np.load(labels_path, allow_pickle=True)
    subject_ids = np.load(subject_ids_path, allow_pickle=True)

    if len(labels) != len(subject_ids):
        raise ValueError(f"Mismatch: {len(labels)} labels vs {len(subject_ids)} subject IDs")

    return subject_ids, labels


def save_fold_files(
    out_dir: Path,
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    subject_ids: np.ndarray,
    labels: np.ndarray,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"fold_{fold_idx}_train_idx.npy", train_idx)
    np.save(out_dir / f"fold_{fold_idx}_test_idx.npy", test_idx)

    summary = {
        "fold": int(fold_idx),
        "num_train": int(len(train_idx)),
        "num_test": int(len(test_idx)),
        "train_label_distribution": {
            "ASD_1": int((labels[train_idx] == 1).sum()),
            "HC_0": int((labels[train_idx] == 0).sum()),
        },
        "test_label_distribution": {
            "ASD_1": int((labels[test_idx] == 1).sum()),
            "HC_0": int((labels[test_idx] == 0).sum()),
        },
        "train_subject_ids_sample": [str(x) for x in subject_ids[train_idx][:5]],
        "test_subject_ids_sample": [str(x) for x in subject_ids[test_idx][:5]],
    }

    with open(out_dir / f"fold_{fold_idx}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def create_splits_for_atlas(project_root: Path, atlas_name: str, folds_list, seed: int):
    print("=" * 72)
    print(f"Creating CV splits for atlas: {atlas_name}")
    print("=" * 72)

    subject_ids, labels = load_labels(project_root, atlas_name)
    indices = np.arange(len(labels))

    print(f"[{atlas_name}] Number of subjects: {len(labels)}")
    print(f"[{atlas_name}] ASD_1 count       : {(labels == 1).sum()}")
    print(f"[{atlas_name}] HC_0 count        : {(labels == 0).sum()}")

    for n_splits in folds_list:
        print("-" * 72)
        print(f"[{atlas_name}] Creating {n_splits}-fold stratified splits")

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )

        out_dir = (
            project_root
            / "data"
            / "interim"
            / "cv_splits"
            / atlas_name
            / f"{n_splits}_fold"
        )

        all_fold_summaries = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(indices, labels), start=1):
            save_fold_files(
                out_dir=out_dir,
                fold_idx=fold_idx,
                train_idx=train_idx,
                test_idx=test_idx,
                subject_ids=subject_ids,
                labels=labels,
            )

            fold_info = {
                "fold": fold_idx,
                "num_train": int(len(train_idx)),
                "num_test": int(len(test_idx)),
                "train_asd": int((labels[train_idx] == 1).sum()),
                "train_hc": int((labels[train_idx] == 0).sum()),
                "test_asd": int((labels[test_idx] == 1).sum()),
                "test_hc": int((labels[test_idx] == 0).sum()),
            }
            all_fold_summaries.append(fold_info)

            print(
                f"  Fold {fold_idx}: "
                f"train={len(train_idx)}, test={len(test_idx)}, "
                f"train ASD/HC=({fold_info['train_asd']}/{fold_info['train_hc']}), "
                f"test ASD/HC=({fold_info['test_asd']}/{fold_info['test_hc']})"
            )

        with open(out_dir / "all_folds_summary.json", "w", encoding="utf-8") as f:
            json.dump(all_fold_summaries, f, indent=2)

        print(f"[{atlas_name}] Saved {n_splits}-fold splits to: {out_dir}")


def main():
    args = parse_args()
    project_root = Path(args.project_root)

    if args.atlas == "all":
        atlases = ["AAL", "HarvardOxford"]
    else:
        atlases = [args.atlas]

    for atlas_name in atlases:
        create_splits_for_atlas(
            project_root=project_root,
            atlas_name=atlas_name,
            folds_list=args.folds,
            seed=args.seed,
        )

    print("=" * 72)
    print("Finished creating CV splits.")
    print("=" * 72)


if __name__ == "__main__":
    main()