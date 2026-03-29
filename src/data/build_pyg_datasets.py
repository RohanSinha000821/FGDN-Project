import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from nilearn.connectome import ConnectivityMeasure
from sklearn.covariance import LedoitWolf
from torch_geometric.data import Data


ATLAS_NAMES = ["AAL", "HarvardOxford"]
SUPPORTED_KINDS = ["tangent", "correlation", "partial correlation", "covariance", "precision"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build PyTorch Geometric datasets for FGDN without tangent leakage."
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
    return parser.parse_args()


def load_subject_bundle(project_root: Path, atlas_name: str):
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


def load_graph_templates(project_root: Path, atlas_name: str, kind: str, n_folds: int, fold_idx: int):
    template_dir = (
        project_root
        / "data"
        / "interim"
        / "graph_templates"
        / atlas_name
        / kind
        / f"{n_folds}_fold"
        / f"fold_{fold_idx}"
    )

    asd_edge_index = np.load(template_dir / "ASD_edge_index.npy").astype(np.int64)
    hc_edge_index = np.load(template_dir / "HC_edge_index.npy").astype(np.int64)

    return asd_edge_index, hc_edge_index


def compute_fold_connectivity(
    project_root: Path,
    atlas_name: str,
    kind: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
):
    """
    For tangent:
      fit on outer-train only, then transform outer-test.
    For non-tangent kinds:
      subset precomputed per-subject matrices.
    """
    if kind == "tangent":
        timeseries, labels, subject_ids = load_subject_bundle(project_root, atlas_name)

        train_ts = [np.asarray(timeseries[int(i)], dtype=np.float64) for i in train_idx]
        test_ts = [np.asarray(timeseries[int(i)], dtype=np.float64) for i in test_idx]

        estimator = ConnectivityMeasure(
            kind="tangent",
            cov_estimator=LedoitWolf(store_precision=False),
            vectorize=False,
            discard_diagonal=False,
        )

        train_conn = estimator.fit_transform(train_ts).astype(np.float32)
        test_conn = estimator.transform(test_ts).astype(np.float32)

        fit_scope = "outer_train_only_tangent_fit"

    else:
        connectivity, labels, subject_ids = load_precomputed_connectivity(
            project_root=project_root,
            atlas_name=atlas_name,
            kind=kind,
        )

        train_conn = connectivity[train_idx].astype(np.float32)
        test_conn = connectivity[test_idx].astype(np.float32)

        fit_scope = "precomputed_per_subject_connectivity"

    train_labels = labels[train_idx]
    test_labels = labels[test_idx]
    train_subject_ids = subject_ids[train_idx]
    test_subject_ids = subject_ids[test_idx]

    return {
        "train_conn": train_conn,
        "test_conn": test_conn,
        "train_labels": train_labels,
        "test_labels": test_labels,
        "train_subject_ids": train_subject_ids,
        "test_subject_ids": test_subject_ids,
        "fit_scope": fit_scope,
    }


def make_data_object(
    x_np: np.ndarray,
    y_int: int,
    subject_id,
    global_index: int,
    split_name: str,
    atlas_name: str,
    kind: str,
    n_folds: int,
    fold_idx: int,
    asd_edge_index_np: np.ndarray,
    hc_edge_index_np: np.ndarray,
) -> Data:
    x = torch.from_numpy(x_np).float()
    y = torch.tensor(y_int, dtype=torch.long)

    data = Data(x=x, y=y)

    data.edge_index_asd = torch.from_numpy(asd_edge_index_np).long()
    data.edge_index_hc = torch.from_numpy(hc_edge_index_np).long()

    data.subject_id = str(subject_id)
    data.sample_index = int(global_index)
    data.split = split_name
    data.atlas = atlas_name
    data.kind = kind
    data.num_folds = int(n_folds)
    data.fold = int(fold_idx)
    data.num_nodes = int(x.shape[0])
    data.num_node_features = int(x.shape[1])

    return data


def build_fold_dataset(
    train_conn: np.ndarray,
    test_conn: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    train_subject_ids: np.ndarray,
    test_subject_ids: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    asd_edge_index_np: np.ndarray,
    hc_edge_index_np: np.ndarray,
    atlas_name: str,
    kind: str,
    n_folds: int,
    fold_idx: int,
):
    train_list: List[Data] = []
    test_list: List[Data] = []

    for local_i, global_idx in enumerate(train_idx):
        train_list.append(
            make_data_object(
                x_np=train_conn[local_i],
                y_int=int(train_labels[local_i]),
                subject_id=train_subject_ids[local_i],
                global_index=int(global_idx),
                split_name="train",
                atlas_name=atlas_name,
                kind=kind,
                n_folds=n_folds,
                fold_idx=fold_idx,
                asd_edge_index_np=asd_edge_index_np,
                hc_edge_index_np=hc_edge_index_np,
            )
        )

    for local_i, global_idx in enumerate(test_idx):
        test_list.append(
            make_data_object(
                x_np=test_conn[local_i],
                y_int=int(test_labels[local_i]),
                subject_id=test_subject_ids[local_i],
                global_index=int(global_idx),
                split_name="test",
                atlas_name=atlas_name,
                kind=kind,
                n_folds=n_folds,
                fold_idx=fold_idx,
                asd_edge_index_np=asd_edge_index_np,
                hc_edge_index_np=hc_edge_index_np,
            )
        )

    return train_list, test_list


def save_fold_dataset(
    out_dir: Path,
    train_list: List[Data],
    test_list: List[Data],
    atlas_name: str,
    kind: str,
    connectivity_fit_scope: str,
    n_folds: int,
    fold_idx: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train_dataset.pt"
    test_path = out_dir / "test_dataset.pt"

    torch.save(train_list, train_path)
    torch.save(test_list, test_path)

    train_labels = np.array([int(d.y.item()) for d in train_list], dtype=np.int64)
    test_labels = np.array([int(d.y.item()) for d in test_list], dtype=np.int64)

    summary: Dict = {
        "atlas": atlas_name,
        "kind": kind,
        "connectivity_fit_scope": connectivity_fit_scope,
        "num_folds": int(n_folds),
        "fold": int(fold_idx),
        "num_train_samples": int(len(train_list)),
        "num_test_samples": int(len(test_list)),
        "train_label_distribution": {
            "ASD_1": int((train_labels == 1).sum()),
            "HC_0": int((train_labels == 0).sum()),
        },
        "test_label_distribution": {
            "ASD_1": int((test_labels == 1).sum()),
            "HC_0": int((test_labels == 0).sum()),
        },
        "num_nodes_per_graph": int(train_list[0].x.shape[0]) if train_list else None,
        "num_node_features": int(train_list[0].x.shape[1]) if train_list else None,
        "asd_num_edges_directed": int(train_list[0].edge_index_asd.shape[1]) if train_list else None,
        "hc_num_edges_directed": int(train_list[0].edge_index_hc.shape[1]) if train_list else None,
        "train_subject_ids_sample": [d.subject_id for d in train_list[:5]],
        "test_subject_ids_sample": [d.subject_id for d in test_list[:5]],
    }

    with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def process_fold(project_root: Path, atlas_name: str, kind: str, n_folds: int, fold_idx: int):
    train_idx, test_idx = load_split_indices(project_root, atlas_name, n_folds, fold_idx)
    asd_edge_index_np, hc_edge_index_np = load_graph_templates(
        project_root=project_root,
        atlas_name=atlas_name,
        kind=kind,
        n_folds=n_folds,
        fold_idx=fold_idx,
    )

    fold_bundle = compute_fold_connectivity(
        project_root=project_root,
        atlas_name=atlas_name,
        kind=kind,
        train_idx=train_idx,
        test_idx=test_idx,
    )

    train_list, test_list = build_fold_dataset(
        train_conn=fold_bundle["train_conn"],
        test_conn=fold_bundle["test_conn"],
        train_labels=fold_bundle["train_labels"],
        test_labels=fold_bundle["test_labels"],
        train_subject_ids=fold_bundle["train_subject_ids"],
        test_subject_ids=fold_bundle["test_subject_ids"],
        train_idx=train_idx,
        test_idx=test_idx,
        asd_edge_index_np=asd_edge_index_np,
        hc_edge_index_np=hc_edge_index_np,
        atlas_name=atlas_name,
        kind=kind,
        n_folds=n_folds,
        fold_idx=fold_idx,
    )

    out_dir = (
        project_root
        / "data"
        / "processed"
        / "pyg_datasets"
        / atlas_name
        / kind
        / f"{n_folds}_fold"
        / f"fold_{fold_idx}"
    )

    save_fold_dataset(
        out_dir=out_dir,
        train_list=train_list,
        test_list=test_list,
        atlas_name=atlas_name,
        kind=kind,
        connectivity_fit_scope=fold_bundle["fit_scope"],
        n_folds=n_folds,
        fold_idx=fold_idx,
    )

    print(
        f"[{atlas_name}] kind={kind} {n_folds}-fold fold {fold_idx}: "
        f"train={len(train_list)}, test={len(test_list)}, "
        f"nodes={train_list[0].x.shape[0]}, features={train_list[0].x.shape[1]}"
    )


def process_atlas(project_root: Path, atlas_name: str, kind: str, folds_list):
    print("=" * 72)
    print(f"Building PyG datasets for atlas={atlas_name}, kind={kind}")
    print("=" * 72)

    for n_folds in folds_list:
        for fold_idx in range(1, n_folds + 1):
            process_fold(
                project_root=project_root,
                atlas_name=atlas_name,
                kind=kind,
                n_folds=n_folds,
                fold_idx=fold_idx,
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
            kind=args.kind,
            folds_list=args.folds,
        )

    print("=" * 72)
    print("Finished building PyG datasets.")
    print("=" * 72)


if __name__ == "__main__":
    main()