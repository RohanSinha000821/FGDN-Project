import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from nilearn.connectome import ConnectivityMeasure
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors


ATLAS_NAMES = ["AAL", "HarvardOxford"]
SUPPORTED_KINDS = ["tangent", "correlation", "partial correlation", "covariance", "precision"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build class-specific FGDN graph templates without tangent leakage."
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default="D:/FGDN_Project",
        help="Path to project root",
    )
    parser.add_argument(
        "--atlas",
        type=str,
        choices=ATLAS_NAMES + ["all"],
        default="all",
        help="Atlas to process",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[5, 10],
        help="Fold settings to process, e.g. --folds 5 10",
    )
    parser.add_argument(
        "--kind",
        type=str,
        choices=SUPPORTED_KINDS,
        default="tangent",
        help="Connectivity feature kind",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Number of nearest neighbors for graph template construction",
    )
    return parser.parse_args()


def load_split_indices(project_root: Path, atlas_name: str, n_folds: int, fold_idx: int):
    split_dir = (
        project_root
        / "data"
        / "interim"
        / "cv_splits"
        / atlas_name
        / f"{n_folds}_fold"
    )

    train_idx = np.load(split_dir / f"fold_{fold_idx}_train_idx.npy")
    test_idx = np.load(split_dir / f"fold_{fold_idx}_test_idx.npy")
    return train_idx.astype(np.int64), test_idx.astype(np.int64)


def load_subject_bundle(project_root: Path, atlas_name: str):
    """
    Leakage-free canonical bundle saved by the corrected build_connectivity.py.
    """
    base_dir = (
        project_root
        / "data"
        / "interim"
        / "subject_timeseries"
        / atlas_name
    )

    timeseries = np.load(base_dir / "timeseries.npy", allow_pickle=True)
    labels = np.load(base_dir / "labels.npy", allow_pickle=True).astype(np.int64)
    subject_ids = np.load(base_dir / "subject_ids.npy", allow_pickle=True)

    if not (len(timeseries) == len(labels) == len(subject_ids)):
        raise ValueError(f"Mismatch in subject bundle for atlas {atlas_name}")

    return timeseries, labels, subject_ids


def load_precomputed_connectivity(project_root: Path, atlas_name: str, kind: str):
    """
    Safe for non-tangent kinds only.
    """
    base_dir = (
        project_root
        / "data"
        / "interim"
        / "connectivity_matrices"
        / atlas_name
        / kind
    )

    connectivity = np.load(base_dir / "connectivity.npy").astype(np.float32)
    labels = np.load(base_dir / "labels.npy", allow_pickle=True).astype(np.int64)
    subject_ids = np.load(base_dir / "subject_ids.npy", allow_pickle=True)

    if not (len(connectivity) == len(labels) == len(subject_ids)):
        raise ValueError(f"Mismatch in connectivity bundle for atlas {atlas_name}, kind={kind}")

    return connectivity, labels, subject_ids


def compute_train_connectivity_for_fold(
    project_root: Path,
    atlas_name: str,
    kind: str,
    train_idx: np.ndarray,
):
    """
    For tangent:
      fit the estimator on outer-train subjects only and return outer-train tangent matrices.
    For non-tangent kinds:
      just subset the precomputed per-subject matrices.
    """
    if kind == "tangent":
        timeseries, labels, subject_ids = load_subject_bundle(project_root, atlas_name)

        train_ts: List[np.ndarray] = [
            np.asarray(timeseries[int(i)], dtype=np.float64) for i in train_idx
        ]

        estimator = ConnectivityMeasure(
            kind="tangent",
            cov_estimator=LedoitWolf(store_precision=False),
            vectorize=False,
            discard_diagonal=False,
        )

        train_conn = estimator.fit_transform(train_ts).astype(np.float32)
        train_labels = labels[train_idx]
        train_subject_ids = subject_ids[train_idx]
        fit_scope = "outer_train_only_tangent_fit"

    else:
        connectivity, labels, subject_ids = load_precomputed_connectivity(
            project_root=project_root,
            atlas_name=atlas_name,
            kind=kind,
        )

        train_conn = connectivity[train_idx].astype(np.float32)
        train_labels = labels[train_idx]
        train_subject_ids = subject_ids[train_idx]
        fit_scope = "precomputed_per_subject_connectivity"

    if len(train_conn) != len(train_labels):
        raise ValueError("Mismatch between train connectivity and labels.")

    return train_conn, train_labels, train_subject_ids, fit_scope


def build_knn_adjacency(mean_fc: np.ndarray, k: int) -> np.ndarray:
    """
    mean_fc shape: [num_rois, num_rois]
    Each node = one ROI
    Node feature for KNN template building = row of the mean FC matrix
    """
    num_rois = mean_fc.shape[0]
    effective_k = min(k, num_rois - 1)

    nbrs = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean")
    nbrs.fit(mean_fc)
    _, indices = nbrs.kneighbors(mean_fc)

    adjacency = np.zeros((num_rois, num_rois), dtype=np.int64)

    for i in range(num_rois):
        for j in indices[i]:
            if i == j:
                continue
            adjacency[i, j] = 1
            adjacency[j, i] = 1

    np.fill_diagonal(adjacency, 0)
    return adjacency


def adjacency_to_edge_index(adjacency: np.ndarray) -> np.ndarray:
    rows, cols = np.where(adjacency > 0)
    return np.vstack([rows, cols]).astype(np.int64)


def save_template(
    out_dir: Path,
    class_name: str,
    mean_fc: np.ndarray,
    adjacency: np.ndarray,
    edge_index: np.ndarray,
    metadata: dict,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"{class_name}_mean_fc.npy", mean_fc.astype(np.float32))
    np.save(out_dir / f"{class_name}_adjacency.npy", adjacency.astype(np.int64))
    np.save(out_dir / f"{class_name}_edge_index.npy", edge_index.astype(np.int64))

    with open(out_dir / f"{class_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def process_fold(project_root: Path, atlas_name: str, n_folds: int, fold_idx: int, kind: str, k: int):
    train_idx, test_idx = load_split_indices(project_root, atlas_name, n_folds, fold_idx)

    train_conn, train_labels, train_subject_ids, fit_scope = compute_train_connectivity_for_fold(
        project_root=project_root,
        atlas_name=atlas_name,
        kind=kind,
        train_idx=train_idx,
    )

    asd_conn = train_conn[train_labels == 1]
    hc_conn = train_conn[train_labels == 0]

    if len(asd_conn) == 0 or len(hc_conn) == 0:
        raise ValueError(
            f"{atlas_name} {n_folds}-fold fold {fold_idx}: one class is empty in training set."
        )

    asd_mean_fc = asd_conn.mean(axis=0)
    hc_mean_fc = hc_conn.mean(axis=0)

    asd_adj = build_knn_adjacency(asd_mean_fc, k=k)
    hc_adj = build_knn_adjacency(hc_mean_fc, k=k)

    asd_edge_index = adjacency_to_edge_index(asd_adj)
    hc_edge_index = adjacency_to_edge_index(hc_adj)

    out_dir = (
        project_root
        / "data"
        / "interim"
        / "graph_templates"
        / atlas_name
        / kind
        / f"{n_folds}_fold"
        / f"fold_{fold_idx}"
    )

    common_meta = {
        "atlas": atlas_name,
        "kind": kind,
        "connectivity_fit_scope": fit_scope,
        "num_folds": n_folds,
        "fold": fold_idx,
        "num_train_subjects_total": int(len(train_idx)),
        "num_test_subjects": int(len(test_idx)),
    }

    save_template(
        out_dir=out_dir,
        class_name="ASD",
        mean_fc=asd_mean_fc,
        adjacency=asd_adj,
        edge_index=asd_edge_index,
        metadata={
            **common_meta,
            "class_name": "ASD",
            "num_train_subjects_class": int(len(asd_conn)),
            "num_rois": int(asd_mean_fc.shape[0]),
            "k": int(min(k, asd_mean_fc.shape[0] - 1)),
            "num_edges_directed": int(asd_edge_index.shape[1]),
            "train_subject_ids_sample": [
                str(x) for x in train_subject_ids[train_labels == 1][:5]
            ],
        },
    )

    save_template(
        out_dir=out_dir,
        class_name="HC",
        mean_fc=hc_mean_fc,
        adjacency=hc_adj,
        edge_index=hc_edge_index,
        metadata={
            **common_meta,
            "class_name": "HC",
            "num_train_subjects_class": int(len(hc_conn)),
            "num_rois": int(hc_mean_fc.shape[0]),
            "k": int(min(k, hc_mean_fc.shape[0] - 1)),
            "num_edges_directed": int(hc_edge_index.shape[1]),
            "train_subject_ids_sample": [
                str(x) for x in train_subject_ids[train_labels == 0][:5]
            ],
        },
    )

    print(
        f"[{atlas_name}] kind={kind} {n_folds}-fold fold {fold_idx}: "
        f"ASD train={len(asd_conn)}, HC train={len(hc_conn)}, "
        f"ASD edges={asd_edge_index.shape[1]}, HC edges={hc_edge_index.shape[1]}"
    )


def process_atlas(project_root: Path, atlas_name: str, folds_list, kind: str, k: int):
    print("=" * 72)
    print(f"Building graph templates for atlas={atlas_name}, kind={kind}")
    print("=" * 72)

    for n_folds in folds_list:
        for fold_idx in range(1, n_folds + 1):
            process_fold(
                project_root=project_root,
                atlas_name=atlas_name,
                n_folds=n_folds,
                fold_idx=fold_idx,
                kind=kind,
                k=k,
            )

    print(f"[{atlas_name}] Finished all requested folds for kind={kind}.")


def main():
    args = parse_args()
    project_root = Path(args.project_root)

    atlases = ATLAS_NAMES if args.atlas == "all" else [args.atlas]

    for atlas_name in atlases:
        process_atlas(
            project_root=project_root,
            atlas_name=atlas_name,
            folds_list=args.folds,
            kind=args.kind,
            k=args.k,
        )

    print("=" * 72)
    print("Finished building graph templates.")
    print("=" * 72)


if __name__ == "__main__":
    main()