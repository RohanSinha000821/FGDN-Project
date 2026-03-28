from pathlib import Path
import re
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(r"D:\FGDN_Project")
PHENO_DIR = PROJECT_ROOT / "data" / "raw" / "abide" / "phenotypic"
TS_DIR = PROJECT_ROOT / "data" / "raw" / "abide" / "roi_timeseries"
RESULTS_DIR = PROJECT_ROOT / "data" / "processed" / "results"


ATLAS_CONFIG = {
    "AAL": 116,
    "HarvardOxford": 111,
    "MODL_128": 128,
}


def find_first_csv(folder: Path):
    csv_files = sorted(folder.glob("*.csv"))
    return csv_files[0] if csv_files else None


def normalize_sub_id(value):
    if pd.isna(value):
        return None

    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]

    digits = "".join(ch for ch in s if ch.isdigit())
    if digits == "":
        return None

    return str(int(digits))


def read_sample_timeseries(file_path: Path):
    return np.loadtxt(file_path)


def get_subject_id_from_filename(file_path: Path):
    """
    Examples:
      CMU_a_0050649_rois_aal.1D       -> 50649
      Leuven_1_0050682_rois_aal.1D    -> 50682
      MaxMun_c_0051328_rois_ho.1D     -> 51328

    Rule:
    take the numeric block immediately before '_rois_'.
    """
    name = file_path.name

    match = re.search(r"_(\d{5,})_rois_", name)
    if match:
        return normalize_sub_id(match.group(1))

    # fallback: take the last 5+ digit block in filename
    matches = re.findall(r"(\d{5,})", file_path.stem)
    if matches:
        return normalize_sub_id(matches[-1])

    return None


def find_roi_files(atlas_dir: Path):
    roi_files = []
    roi_files.extend(atlas_dir.rglob("*.1D"))
    roi_files.extend(atlas_dir.rglob("*.txt"))
    return sorted(set(roi_files))


def inspect_atlas(atlas_name: str, expected_rois: int, phenotypic_df: pd.DataFrame):
    atlas_dir = TS_DIR / atlas_name

    print("\n" + "=" * 70)
    print(f"Inspecting atlas: {atlas_name}")
    print("=" * 70)

    if not atlas_dir.exists():
        print(f"[ERROR] Folder not found: {atlas_dir}")
        return

    roi_files = find_roi_files(atlas_dir)

    print(f"Atlas root folder: {atlas_dir}")
    print(f"Number of ROI files found: {len(roi_files)}")

    if len(roi_files) == 0:
        print("[ERROR] No ROI files found (*.1D or *.txt).")
        return

    file_subject_map = {}
    unmatched_files = []

    for f in roi_files:
        sub_id = get_subject_id_from_filename(f)
        if sub_id is None:
            unmatched_files.append(str(f))
            continue
        file_subject_map[sub_id] = f

    file_subject_ids = set(file_subject_map.keys())

    if "SUB_ID" not in phenotypic_df.columns:
        print("[ERROR] 'SUB_ID' column not found in phenotypic CSV.")
        print("Available columns:", list(phenotypic_df.columns))
        return

    phenotypic_df = phenotypic_df.copy()
    phenotypic_df["SUB_ID_CLEAN"] = phenotypic_df["SUB_ID"].apply(normalize_sub_id)

    pheno_subject_ids = set(phenotypic_df["SUB_ID_CLEAN"].dropna().tolist())
    matched_subject_ids = sorted(
        file_subject_ids.intersection(pheno_subject_ids),
        key=lambda x: int(x)
    )

    print(f"Subjects in phenotypic CSV: {len(pheno_subject_ids)}")
    print(f"Unique subjects in ROI files: {len(file_subject_ids)}")
    print(f"Matched subjects ({atlas_name}): {len(matched_subject_ids)}")

    print("\nFirst 10 phenotypic IDs:", sorted(list(pheno_subject_ids))[:10])
    print("First 10 ROI file IDs   :", sorted(list(file_subject_ids))[:10])

    if unmatched_files:
        print("\nFirst 5 files with no detectable subject ID:")
        for x in unmatched_files[:5]:
            print("  ", x)

    if len(matched_subject_ids) == 0:
        print("[ERROR] No matched subject IDs between phenotypic CSV and ROI files.")
        return

    sample_subject = matched_subject_ids[0]
    sample_file = file_subject_map[sample_subject]
    sample_arr = read_sample_timeseries(sample_file)

    print(f"\nSample subject ID: {sample_subject}")
    print(f"Sample file: {sample_file}")
    print(f"Sample shape: {sample_arr.shape}")

    if sample_arr.ndim != 2:
        print("[WARNING] Sample timeseries is not 2D. Expected [timepoints, n_rois].")
    else:
        timepoints, n_rois = sample_arr.shape
        print(f"Timepoints: {timepoints}")
        print(f"ROIs: {n_rois}")

        if n_rois == expected_rois:
            print(f"[OK] ROI count matches expected atlas dimension: {expected_rois}")
        else:
            print(f"[WARNING] ROI count mismatch. Expected {expected_rois}, got {n_rois}")

    matched_df = phenotypic_df[phenotypic_df["SUB_ID_CLEAN"].isin(matched_subject_ids)]

    print("\nDiagnosis distribution in matched subjects:")
    if "DX_GROUP" in matched_df.columns:
        print(matched_df["DX_GROUP"].value_counts(dropna=False).sort_index())
    else:
        print("[WARNING] 'DX_GROUP' column not found in phenotypic CSV.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / f"verify_{atlas_name}.csv"

    report_df = pd.DataFrame({
        "subject_id": matched_subject_ids,
        "has_roi_file": True,
        "atlas": atlas_name,
        "file_path": [str(file_subject_map[sid]) for sid in matched_subject_ids],
    })
    report_df.to_csv(report_path, index=False)
    print(f"\nSaved verification report: {report_path}")


def main():
    print("=" * 70)
    print("ABIDE DATA VERIFICATION")
    print("=" * 70)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Phenotypic directory: {PHENO_DIR}")
    print(f"ROI timeseries root: {TS_DIR}")

    csv_path = find_first_csv(PHENO_DIR)

    if csv_path is None:
        print("\n[ERROR] No phenotypic CSV found.")
        print(f"Put the CSV inside: {PHENO_DIR}")
        return

    print(f"\nUsing phenotypic CSV: {csv_path}")
    phenotypic_df = pd.read_csv(csv_path)

    print(f"Phenotypic CSV shape: {phenotypic_df.shape}")
    print("Phenotypic columns:")
    print(list(phenotypic_df.columns))

    for atlas_name, expected_rois in ATLAS_CONFIG.items():
        inspect_atlas(atlas_name, expected_rois, phenotypic_df)


if __name__ == "__main__":
    main()