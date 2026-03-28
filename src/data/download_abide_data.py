from pathlib import Path
import argparse
import subprocess
import sys
import urllib.request

PROJECT_ROOT = Path(r"D:\FGDN_Project")
RAW_ABIDE_DIR = PROJECT_ROOT / "data" / "raw" / "abide"
PHENO_DIR = RAW_ABIDE_DIR / "phenotypic"
DOWNLOADS_DIR = RAW_ABIDE_DIR / "downloads"
ROI_ROOT = RAW_ABIDE_DIR / "roi_timeseries"

PHENO_URL = (
    "https://raw.githubusercontent.com/"
    "preprocessed-connectomes-project/abide/master/"
    "Phenotypic_V1_0b_preprocessed1.csv"
)

OFFICIAL_DOWNLOADER_URL = (
    "https://raw.githubusercontent.com/"
    "preprocessed-connectomes-project/abide/master/"
    "download_abide_preproc.py"
)


def ensure_dirs():
    PHENO_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    ROI_ROOT.mkdir(parents=True, exist_ok=True)


def download_file(url: str, out_path: Path):
    if out_path.exists():
        print(f"[SKIP] Already exists: {out_path}")
        return
    print(f"[DOWNLOAD] {url}")
    print(f"          -> {out_path}")
    urllib.request.urlretrieve(url, out_path)
    print("[OK] Download complete")


def get_official_downloader() -> Path:
    script_path = DOWNLOADS_DIR / "download_abide_preproc.py"
    download_file(OFFICIAL_DOWNLOADER_URL, script_path)
    return script_path


def get_phenotypic_csv() -> Path:
    csv_path = PHENO_DIR / "Phenotypic_V1_0b_preprocessed1.csv"
    download_file(PHENO_URL, csv_path)
    return csv_path


def run_official_downloader(
    derivative: str,
    pipeline: str,
    strategy: str,
    out_dir: Path,
    asd: bool = False,
    tdc: bool = False,
):
    script_path = get_official_downloader()
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script_path),
        "-d", derivative,
        "-p", pipeline,
        "-s", strategy,
        "-o", str(out_dir),
    ]

    if asd:
        cmd.append("-a")
    if tdc:
        cmd.append("-c")

    print("\n[RUNNING OFFICIAL DOWNLOADER]")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Download ABIDE phenotypic CSV and selected preprocessed derivatives."
    )
    parser.add_argument(
        "--download-phenotypic",
        action="store_true",
        help="Download the official Phenotypic_V1_0b_preprocessed1.csv",
    )
    parser.add_argument(
        "--download-official-script",
        action="store_true",
        help="Download the official ABIDE downloader script only",
    )
    parser.add_argument(
        "--derivative",
        type=str,
        default=None,
        help="Derivative for official downloader, e.g. rois_aal or rois_ho",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="cpac",
        help="Pipeline, e.g. cpac",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="nofilt_noglobal",
        help="Noise-removal strategy, e.g. nofilt_noglobal",
    )
    parser.add_argument(
        "--out-subdir",
        type=str,
        default=None,
        help="Local output subfolder under data/raw/abide/roi_timeseries",
    )
    parser.add_argument(
        "--asd",
        action="store_true",
        help="Download only ASD participants",
    )
    parser.add_argument(
        "--tdc",
        action="store_true",
        help="Download only TDC participants",
    )

    args = parser.parse_args()
    ensure_dirs()

    if args.download_phenotypic:
        get_phenotypic_csv()

    if args.download_official_script:
        get_official_downloader()

    if args.derivative is not None:
        if args.out_subdir is None:
            raise ValueError(
                "When using --derivative, also provide --out-subdir "
                "(for example AAL or HarvardOxford)."
            )

        out_dir = ROI_ROOT / args.out_subdir
        run_official_downloader(
            derivative=args.derivative,
            pipeline=args.pipeline,
            strategy=args.strategy,
            out_dir=out_dir,
            asd=args.asd,
            tdc=args.tdc,
        )

    print("\nDone.")
    print(f"Phenotypic folder: {PHENO_DIR}")
    print(f"ROI root folder   : {ROI_ROOT}")


if __name__ == "__main__":
    main()