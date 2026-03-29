import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
from sklearn.covariance import LedoitWolf


ATLAS_CONFIG = {
    "AAL": {
        "folder_name": "AAL",
        "suffix": "rois_aal.1D",
        "expected_rois": 116,
    },
    "HarvardOxford": {
        "folder_name": "HarvardOxford",
        "suffix": "rois_ho.1D",
        "expected_rois": 111,
    },
    # Add MODL later when integrated end-to-end.
}

# These kinds are per-subject estimators and can be computed globally
# without cross-subject leakage.
NON_LEAKY_KINDS = {
    "correlation",
    "partial correlation",
    "covariance",
    "precision",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect matched ABIDE ROI time-series and optionally build "
            "non-leaky connectivity matrices. "
            "Important: tangent connectivity is NOT computed globally here, "
            "because tangent must be fit within each training fold only."
        )
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default="D:/FGDN_Project",
        help="Path to project root directory.",
    )
    parser.add_argument(
        "--atlas",
        type=str,
        choices=["AAL", "HarvardOxford", "all"],
        default="all",
        help="Atlas to process.",
    )
    parser.add_argument(
        "--kind",
        type=str,
        choices=["tangent", "correlation", "partial correlation", "covariance", "precision"],
        default="tangent",
        help=(
            "Connectivity kind. "
            "For kind='tangent', this script will save only the subject bundle "
            "and a tangent placeholder/stub, not connectivity.npy."
        ),
    )
    parser.add_argument(
        "--min-timepoints",
        type=int,
        default=20,
        help="Minimum number of timepoints required to keep a subject.",
    )
    parser.add_argument(
        "--save-timeseries-info",
        action="store_true",
        help="If set, saves subject-level metadata CSV.",
    )
    return parser.parse_args()


def load_phenotypic_csv(project_root: Path) -> pd.DataFrame:
    phenotypic_path = (
        project_root
        / "data"
        / "raw"
        / "abide"
        / "phenotypic"
        / "Phenotypic_V1_0b_preprocessed1.csv"
    )

    if not phenotypic_path.exists():
        raise FileNotFoundError(f"Phenotypic CSV not found: {phenotypic_path}")

    df = pd.read_csv(phenotypic_path)

    required_cols = ["SUB_ID", "DX_GROUP"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Phenotypic CSV missing required columns: {missing}")

    df = df.copy()
    df["SUB_ID"] = df["SUB_ID"].astype(str).str.extract(r"(\d+)")[0].str.zfill(7)
    df = df.dropna(subset=["SUB_ID", "DX_GROUP"])

    # Map ABIDE convention to binary labels used in the project:
    # ASD -> 1, HC -> 0
    dx_map = {1: 1, 2: 0}
    df = df[df["DX_GROUP"].isin(dx_map.keys())].copy()
    df["label"] = df["DX_GROUP"].map(dx_map)

    return df


def extract_subject_id(file_path: Path) -> str:
    """
    Extract ABIDE numeric subject ID from filenames like:
      Pitt_0050004_rois_aal.1D
      Leuven_1_0050682_rois_aal.1D
      MaxMun_c_0051328_rois_ho.1D
    """
    match = re.search(r"_(\d{7})_rois_", file_path.name)
    if not match:
        raise ValueError(f"Could not extract subject ID from filename: {file_path.name}")
    return match.group(1)


def find_roi_files(project_root: Path, atlas_name: str) -> List[Path]:
    atlas_folder = ATLAS_CONFIG[atlas_name]["folder_name"]
    suffix = ATLAS_CONFIG[atlas_name]["suffix"]

    roi_root = (
        project_root
        / "data"
        / "raw"
        / "abide"
        / "roi_timeseries"
        / atlas_folder
    )

    if not roi_root.exists():
        raise FileNotFoundError(f"ROI folder not found for atlas {atlas_name}: {roi_root}")

    files = sorted(roi_root.rglob(f"*{suffix}"))
    if not files:
        raise FileNotFoundError(f"No ROI files found under {roi_root} with suffix {suffix}")

    return files


def read_timeseries(file_path: Path, expected_rois: int, min_timepoints: int) -> np.ndarray:
    ts = np.loadtxt(file_path, dtype=np.float64)

    if ts.ndim == 1:
        ts = np.expand_dims(ts, axis=0)

    if ts.ndim != 2:
        raise ValueError(f"Timeseries must be 2D, got shape {ts.shape} for {file_path}")

    n_timepoints, n_rois = ts.shape

    if n_timepoints < min_timepoints:
        raise ValueError(
            f"Too few timepoints ({n_timepoints}) in {file_path}; "
            f"minimum required is {min_timepoints}"
        )

    if n_rois != expected_rois:
        raise ValueError(
            f"ROI count mismatch in {file_path}: expected {expected_rois}, got {n_rois}"
        )

    if not np.isfinite(ts).all():
        raise ValueError(f"Non-finite values found in timeseries: {file_path}")

    return ts


def collect_subject_timeseries(
    project_root: Path,
    phenotypic_df: pd.DataFrame,
    atlas_name: str,
    min_timepoints: int,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, List[Dict]]:
    roi_files = find_roi_files(project_root, atlas_name)
    expected_rois = ATLAS_CONFIG[atlas_name]["expected_rois"]

    subj_to_label = dict(zip(phenotypic_df["SUB_ID"], phenotypic_df["label"]))

    timeseries_list: List[np.ndarray] = []
    subject_ids: List[str] = []
    labels: List[int] = []
    metadata_rows: List[Dict] = []

    skipped_no_match = 0
    skipped_bad_file = 0

    for file_path in roi_files:
        try:
            subject_id = extract_subject_id(file_path)
        except Exception:
            skipped_bad_file += 1
            continue

        if subject_id not in subj_to_label:
            skipped_no_match += 1
            continue

        try:
            ts = read_timeseries(
                file_path=file_path,
                expected_rois=expected_rois,
                min_timepoints=min_timepoints,
            )
        except Exception as exc:
            skipped_bad_file += 1
            metadata_rows.append(
                {
                    "subject_id": subject_id,
                    "file_path": str(file_path),
                    "status": "skipped_bad_file",
                    "reason": str(exc),
                }
            )
            continue

        timeseries_list.append(ts.astype(np.float32))
        subject_ids.append(subject_id)
        labels.append(int(subj_to_label[subject_id]))
        metadata_rows.append(
            {
                "subject_id": subject_id,
                "file_path": str(file_path),
                "status": "kept",
                "n_timepoints": int(ts.shape[0]),
                "n_rois": int(ts.shape[1]),
                "label": int(subj_to_label[subject_id]),
            }
        )

    if not timeseries_list:
        raise RuntimeError(f"No valid matched subjects found for atlas {atlas_name}")

    print(f"[{atlas_name}] Total ROI files found      : {len(roi_files)}")
    print(f"[{atlas_name}] Valid matched subjects    : {len(timeseries_list)}")
    print(f"[{atlas_name}] Skipped (no phenotype)   : {skipped_no_match}")
    print(f"[{atlas_name}] Skipped (bad file/shape) : {skipped_bad_file}")

    return (
        timeseries_list,
        np.array(subject_ids, dtype=object),
        np.array(labels, dtype=np.int64),
        metadata_rows,
    )


def compute_non_leaky_connectivity(timeseries_list: List[np.ndarray], kind: str) -> np.ndarray:
    """
    Safe only for per-subject estimators like correlation/covariance/precision.
    NOT used for tangent.
    """
    if kind not in NON_LEAKY_KINDS:
        raise ValueError(
            f"compute_non_leaky_connectivity called with unsupported kind={kind}. "
            "Tangent must be computed inside CV folds later."
        )

    estimator = ConnectivityMeasure(
        kind=kind,
        cov_estimator=LedoitWolf(store_precision=False),
        vectorize=False,
        discard_diagonal=False,
    )
    connectivity = estimator.fit_transform(timeseries_list)
    return connectivity.astype(np.float32)


def save_subject_bundle(
    project_root: Path,
    atlas_name: str,
    subject_ids: np.ndarray,
    labels: np.ndarray,
    timeseries_list: List[np.ndarray],
    metadata_rows: List[Dict],
    save_timeseries_info: bool,
) -> Path:
    """
    Canonical leakage-free bundle saved once per atlas.
    Later scripts should use this bundle to build fold-specific tangent features.
    """
    out_dir = project_root / "data" / "interim" / "subject_timeseries" / atlas_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use an object array so variable-length time series are supported too.
    ts_obj = np.empty(len(timeseries_list), dtype=object)
    for i, ts in enumerate(timeseries_list):
        ts_obj[i] = ts

    np.save(out_dir / "subject_ids.npy", subject_ids, allow_pickle=True)
    np.save(out_dir / "labels.npy", labels, allow_pickle=True)
    np.save(out_dir / "timeseries.npy", ts_obj, allow_pickle=True)

    summary = {
        "atlas": atlas_name,
        "num_subjects": int(len(subject_ids)),
        "label_distribution": {
            "ASD_1": int((labels == 1).sum()),
            "HC_0": int((labels == 0).sum()),
        },
        "example_timeseries_shape": list(timeseries_list[0].shape),
        "storage_note": (
            "timeseries.npy is the canonical leakage-free subject bundle. "
            "For tangent connectivity, fit the tangent model inside each training fold only."
        ),
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if save_timeseries_info:
        pd.DataFrame(metadata_rows).to_csv(out_dir / "subject_metadata.csv", index=False)

    print(f"[{atlas_name}] Saved subject bundle to: {out_dir}")
    return out_dir


def save_connectivity_outputs(
    project_root: Path,
    atlas_name: str,
    kind: str,
    subject_ids: np.ndarray,
    labels: np.ndarray,
    connectivity: np.ndarray,
) -> Path:
    out_dir = (
        project_root
        / "data"
        / "interim"
        / "connectivity_matrices"
        / atlas_name
        / kind
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "subject_ids.npy", subject_ids, allow_pickle=True)
    np.save(out_dir / "labels.npy", labels, allow_pickle=True)
    np.save(out_dir / "connectivity.npy", connectivity)

    summary = {
        "atlas": atlas_name,
        "kind": kind,
        "num_subjects": int(len(subject_ids)),
        "num_rois": int(connectivity.shape[1]),
        "matrix_shape": list(connectivity.shape),
        "label_distribution": {
            "ASD_1": int((labels == 1).sum()),
            "HC_0": int((labels == 0).sum()),
        },
        "leakage_status": "safe_global_compute_for_this_kind",
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[{atlas_name}] Saved {kind} connectivity to: {out_dir}")
    print(f"[{atlas_name}] connectivity.npy shape: {connectivity.shape}")
    return out_dir


def save_tangent_stub(
    project_root: Path,
    atlas_name: str,
    subject_ids: np.ndarray,
    labels: np.ndarray,
    timeseries_bundle_dir: Path,
) -> Path:
    """
    Keeps create_cv_splits.py usable, because that script currently reads
    subject_ids.npy and labels.npy from connectivity_matrices/<atlas>/tangent/.
    We intentionally do NOT save connectivity.npy here.
    """
    out_dir = (
        project_root
        / "data"
        / "interim"
        / "connectivity_matrices"
        / atlas_name
        / "tangent"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "subject_ids.npy", subject_ids, allow_pickle=True)
    np.save(out_dir / "labels.npy", labels, allow_pickle=True)

    summary = {
        "atlas": atlas_name,
        "kind": "tangent",
        "num_subjects": int(len(subject_ids)),
        "label_distribution": {
            "ASD_1": int((labels == 1).sum()),
            "HC_0": int((labels == 0).sum()),
        },
        "connectivity_saved": False,
        "reason": (
            "Tangent connectivity must be fit later within each training fold only. "
            "Global tangent computation here would leak information across folds."
        ),
        "timeseries_bundle_dir": str(timeseries_bundle_dir),
        "next_step": (
            "Use subject_timeseries/<atlas>/timeseries.npy with CV split indices "
            "to build fold-specific tangent matrices downstream."
        ),
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[{atlas_name}] Saved tangent stub to: {out_dir}")
    print(f"[{atlas_name}] Note: connectivity.npy was intentionally NOT created for tangent.")
    return out_dir


def process_atlas(
    project_root: Path,
    phenotypic_df: pd.DataFrame,
    atlas_name: str,
    kind: str,
    min_timepoints: int,
    save_timeseries_info: bool,
) -> None:
    print("=" * 72)
    print(f"Processing atlas: {atlas_name}")
    print("=" * 72)

    timeseries_list, subject_ids, labels, metadata_rows = collect_subject_timeseries(
        project_root=project_root,
        phenotypic_df=phenotypic_df,
        atlas_name=atlas_name,
        min_timepoints=min_timepoints,
    )

    timeseries_bundle_dir = save_subject_bundle(
        project_root=project_root,
        atlas_name=atlas_name,
        subject_ids=subject_ids,
        labels=labels,
        timeseries_list=timeseries_list,
        metadata_rows=metadata_rows,
        save_timeseries_info=save_timeseries_info,
    )

    if kind == "tangent":
        save_tangent_stub(
            project_root=project_root,
            atlas_name=atlas_name,
            subject_ids=subject_ids,
            labels=labels,
            timeseries_bundle_dir=timeseries_bundle_dir,
        )
        return

    connectivity = compute_non_leaky_connectivity(
        timeseries_list=timeseries_list,
        kind=kind,
    )

    if connectivity.shape[0] != len(subject_ids):
        raise RuntimeError("Subject count mismatch between IDs and connectivity matrices")

    save_connectivity_outputs(
        project_root=project_root,
        atlas_name=atlas_name,
        kind=kind,
        subject_ids=subject_ids,
        labels=labels,
        connectivity=connectivity,
    )


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)

    phenotypic_df = load_phenotypic_csv(project_root)

    if args.atlas == "all":
        atlas_names = ["AAL", "HarvardOxford"]
    else:
        atlas_names = [args.atlas]

    for atlas_name in atlas_names:
        process_atlas(
            project_root=project_root,
            phenotypic_df=phenotypic_df,
            atlas_name=atlas_name,
            kind=args.kind,
            min_timepoints=args.min_timepoints,
            save_timeseries_info=args.save_timeseries_info,
        )

    print("=" * 72)
    print("Finished build_connectivity step.")
    print("=" * 72)
    if args.kind == "tangent":
        print("Important: tangent connectivity matrices were NOT built globally.")
        print("Next, keep CV splits and move to fold-specific tangent construction downstream.")


if __name__ == "__main__":
    main()