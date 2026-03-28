# FGDN Project

Implementation and reproduction workflow for the paper:

**Identification of Autism Spectrum Disorder With Functional Graph
Discriminative Network (FGDN)**

This repository aims to reproduce the FGDN pipeline using ABIDE
preprocessed rs-fMRI ROI time-series data and later extend it to dynamic
functional connectivity (dynamic FC).

------------------------------------------------------------------------

## Current Status

### Completed

-   Project structure setup\
-   Python virtual environment setup\
-   PyTorch + PyTorch Geometric installation\
-   CUDA verification\
-   ABIDE phenotypic CSV download\
-   ROI timeseries download for:
    -   AAL\
    -   HarvardOxford\
-   Dataset verification pipeline\
-   Subject matching verification

### Pending

-   Connectivity matrix generation\
-   Cross-validation splits\
-   Graph template construction\
-   PyTorch Geometric dataset creation\
-   FGDN model implementation\
-   Training and evaluation\
-   MODL_128 data sourcing\
-   Dynamic FC extension

------------------------------------------------------------------------

## Quick Start

### 1. Clone repository

``` bash
git clone https://github.com/RohanSinha000821/FGDN-Project.git
cd FGDN-Project
```

### 2. Create virtual environment

``` bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

``` bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Dataset Download

### Download phenotypic CSV

``` bash
python src\data\download_abide_data.py --download-phenotypic
```

### Download ROI timeseries

#### AAL

``` bash
python src\data\download_abide_data.py --derivative rois_aal --pipeline cpac --strategy nofilt_noglobal --out-subdir AAL
```

#### HarvardOxford

``` bash
python src\data\download_abide_data.py --derivative rois_ho --pipeline cpac --strategy nofilt_noglobal --out-subdir HarvardOxford
```

------------------------------------------------------------------------

## Verify Dataset

``` bash
python src\data\verify_abide.py
```

------------------------------------------------------------------------

## Project Structure

    FGDN_Project/
    ├── .gitignore
    ├── README.md
    ├── requirements.txt
    ├── data/
    │   ├── raw/
    │   ├── interim/
    │   └── processed/
    ├── src/
    │   ├── data/
    │   ├── evaluation/
    │   ├── models/
    │   ├── training/
    │   └── utils/
    ├── outputs/
    └── docs/

------------------------------------------------------------------------

## Environment Notes

-   PyTorch with CUDA support\
-   PyTorch Geometric working\
-   GPU verified: NVIDIA RTX 3050

------------------------------------------------------------------------

## Verified Data

### AAL

-   Subjects: 531\
-   Shape: (196, 116)

### HarvardOxford

-   Subjects: 884\
-   Shape: (196, 111)

**Note:** HarvardOxford has 111 ROIs (not 118).

------------------------------------------------------------------------

## Next Steps

1.  Build connectivity matrices\
2.  Create cross-validation splits\
3.  Construct ASD & HC graph templates\
4.  Build PyTorch Geometric datasets\
5.  Implement FGDN model\
6.  Train and evaluate\
7.  Extend to dynamic FC

------------------------------------------------------------------------

## Repository Policy

This repository does NOT include:

-   Raw ABIDE data\
-   ROI datasets\
-   Large `.npy`, `.pt` files\
-   Model checkpoints\
-   Virtual environments

All data must be generated locally.

------------------------------------------------------------------------

## Notes

-   AAL and HarvardOxford are verified and ready\
-   MODL_128 is not yet included\
-   Project is being built step-by-step with validation at each stage
