# FGDN Project

Implementation and reproduction workflow for the paper:

**Identification of Autism Spectrum Disorder With Functional Graph Discriminative Network (FGDN)**

This repository reproduces the FGDN pipeline using ABIDE preprocessed rs-fMRI ROI time-series data and is designed to support a later extension toward dynamic functional connectivity (dynamic FC).

---

## Current Status

### Completed

- Project structure setup
- Python virtual environment setup
- PyTorch + PyTorch Geometric installation
- CUDA verification
- ABIDE phenotypic CSV download
- ROI timeseries download for:
  - AAL
  - HarvardOxford
- Dataset verification pipeline
- Subject matching verification
- Connectivity matrix generation
- Stratified cross-validation split generation
- ASD/HC graph template construction
- PyTorch Geometric dataset creation
- FGDN model implementation
- FGDN training pipeline
- FGDN evaluation pipeline
- Paper-faithful FGDN branch-head revision
- Strict inner-train / monitor split inside each training fold
- Inner-train-only template rebuilding during training/evaluation
- AAL 5-fold baseline run
- AAL 10-fold baseline run
- HarvardOxford 5-fold baseline run
- HarvardOxford 10-fold baseline run

### Pending

- MODL_128 data sourcing and integration
- Static FGDN reproduction on MODL_128
- Dynamic FC extension
- Final paper/report write-up and visualizations

---

## Final Static Baseline Results

### AAL 5-fold CV

- **Accuracy mean:** `0.676089`
- **Accuracy std:** `0.028268`
- **AUC mean:** `0.738340`
- **AUC std:** `0.043578`

#### Fold-wise AAL 5-fold results

- Fold 1: Accuracy `0.672897`, AUC `0.723427`
- Fold 2: Accuracy `0.679245`, AUC `0.765419`
- Fold 3: Accuracy `0.660377`, AUC `0.720143`
- Fold 4: Accuracy `0.726415`, AUC `0.805348`
- Fold 5: Accuracy `0.641509`, AUC `0.677362`

### AAL 10-fold CV

- **Accuracy mean:** `0.676031`
- **Accuracy std:** `0.056095`
- **AUC mean:** `0.746586`
- **AUC std:** `0.067851`

#### Fold-wise AAL 10-fold results

- Fold 1: Accuracy `0.703704`, AUC `0.835165`
- Fold 2: Accuracy `0.679245`, AUC `0.715714`
- Fold 3: Accuracy `0.716981`, AUC `0.728571`
- Fold 4: Accuracy `0.735849`, AUC `0.797143`
- Fold 5: Accuracy `0.603774`, AUC `0.634286`
- Fold 6: Accuracy `0.679245`, AUC `0.790598`
- Fold 7: Accuracy `0.735849`, AUC `0.823362`
- Fold 8: Accuracy `0.716981`, AUC `0.790598`
- Fold 9: Accuracy `0.622642`, AUC `0.645299`
- Fold 10: Accuracy `0.566038`, AUC `0.705128`

### HarvardOxford 5-fold CV

- **Accuracy mean:** `0.686653`
- **Accuracy std:** `0.018399`
- **AUC mean:** `0.741884`
- **AUC std:** `0.015011`

#### Fold-wise HarvardOxford 5-fold results

- Fold 1: Accuracy `0.661017`, AUC `0.718357`
- Fold 2: Accuracy `0.700565`, AUC `0.751605`
- Fold 3: Accuracy `0.711864`, AUC `0.759435`
- Fold 4: Accuracy `0.672316`, AUC `0.730967`
- Fold 5: Accuracy `0.687500`, AUC `0.749058`

### HarvardOxford 10-fold CV

- **Accuracy mean:** `0.684334`
- **Accuracy std:** `0.043705`
- **AUC mean:** `0.751571`
- **AUC std:** `0.035639`

#### Fold-wise HarvardOxford 10-fold results

- Fold 1: Accuracy `0.651685`, AUC `0.736280`
- Fold 2: Accuracy `0.707865`, AUC `0.753049`
- Fold 3: Accuracy `0.662921`, AUC `0.708333`
- Fold 4: Accuracy `0.764045`, AUC `0.805386`
- Fold 5: Accuracy `0.715909`, AUC `0.777893`
- Fold 6: Accuracy `0.715909`, AUC `0.773742`
- Fold 7: Accuracy `0.647727`, AUC `0.714063`
- Fold 8: Accuracy `0.613636`, AUC `0.691230`
- Fold 9: Accuracy `0.715909`, AUC `0.784375`
- Fold 10: Accuracy `0.647727`, AUC `0.771354`

### Current Best Static Result

- **Best AUC:** HarvardOxford 10-fold, `0.751571`
- **Best accuracy:** HarvardOxford 5-fold, `0.686653`

### Methodology Note

The current results use the corrected strict protocol:

- outer train / outer test cross-validation split
- inner-train / monitor split inside each outer training fold
- ASD/HC templates rebuilt from **inner-train only**
- checkpoint selection by **lowest monitor loss**
- final reporting on the untouched outer test fold

---

## Quick Start

### 1. Clone repository

```bash
git clone https://github.com/RohanSinha000821/FGDN-Project.git
cd FGDN-Project
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## Dataset Download

### Download phenotypic CSV

```bash
python src\data\download_abide_data.py --download-phenotypic
```

### Download ROI timeseries

#### AAL

```bash
python src\data\download_abide_data.py --derivative rois_aal --pipeline cpac --strategy nofilt_noglobal --out-subdir AAL
```

#### HarvardOxford

```bash
python src\data\download_abide_data.py --derivative rois_ho --pipeline cpac --strategy nofilt_noglobal --out-subdir HarvardOxford
```

---

## Verify Dataset

```bash
python src\data\verify_abide.py
```

---

## Build Connectivity Matrices

```bash
python src\data\build_connectivity.py --atlas all --kind tangent --save-timeseries-info
```

This creates outputs under:

```text
data\interim\connectivity_matrices\AAL\tangent\
data\interim\connectivity_matrices\HarvardOxford\tangent\
```

---

## Create Cross-Validation Splits

```bash
python src\data\create_cv_splits.py --atlas all --folds 5 10 --seed 42
```

This creates stratified 5-fold and 10-fold splits for AAL and HarvardOxford.

---

## Build Graph Templates

```bash
python src\data\build_graph_templates.py --atlas all --folds 5 10 --k 20
```

For each fold, this builds:

- ASD graph template
- HC graph template

using **training subjects only** in the original preprocessing pipeline.

---

## Build PyTorch Geometric Datasets

```bash
python src\data\build_pyg_datasets.py --atlas all --folds 5 10
```

This converts each subject into a PyG graph sample with:

- node features from subject FC rows
- `edge_index_asd`
- `edge_index_hc`
- label
- subject metadata

---

## Model Sanity Check

```bash
python src\models\test_fgdn_model.py
```

Expected behavior:

- dataset loads correctly
- FGDN model forward pass works
- output logits shape is `(1, 2)`

---

## Train FGDN

### Example: AAL 5-fold, fold 1

```bash
python -m src.training.train_fgdn --atlas AAL --num-folds 5 --fold 1 --epochs 50 --batch-size 16
```

### Example: HarvardOxford 10-fold, fold 1

```bash
python -m src.training.train_fgdn --atlas HarvardOxford --num-folds 10 --fold 1 --epochs 50 --batch-size 16
```

Training outputs are saved under:

```text
outputs\checkpoints\fgdn\...
data\processed\logs\fgdn\...
```

---

## Evaluate FGDN

### Evaluate best checkpoint for one fold

```bash
python -m src.evaluation.evaluate_fgdn --atlas AAL --num-folds 5 --fold 1 --checkpoint-type best --batch-size 16
```

This saves:

- evaluation JSON
- predictions NPZ

under:

```text
outputs\tables\fgdn\...
```

---

## Summarize Cross-Validation Results

### Example: summarize AAL 10-fold

```bash
python -m src.evaluation.summarize_fgdn_cv --atlas AAL --num-folds 10 --checkpoint-type best
```

### Example: summarize HarvardOxford 10-fold

```bash
python -m src.evaluation.summarize_fgdn_cv --atlas HarvardOxford --num-folds 10 --checkpoint-type best
```

This generates the overall CV summary for the requested atlas and fold configuration.

---

## Project Structure

```text
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
```

---

## Environment Notes

- PyTorch with CUDA support
- PyTorch Geometric working
- GPU verified locally
- Current working system used during development: NVIDIA RTX 3050 Laptop GPU

---

## Verified Data

### AAL

- Subjects: 531
- Example timeseries shape: `(196, 116)`

### HarvardOxford

- Subjects: 884
- Example timeseries shape: `(196, 111)`

**Important:** HarvardOxford has **111 ROIs**, not 118, in the downloaded version used here.

---

## Repository Policy

This repository does **not** include:

- raw ABIDE data
- ROI datasets
- generated connectivity arrays
- PyG dataset `.pt` files
- model checkpoints

Lightweight result summaries and logs can be tracked, but large data artifacts must be generated locally.

---

## Next Steps

1. Integrate MODL_128 data
2. Run static FGDN on MODL_128
3. Compare static results across AAL, HarvardOxford, and MODL
4. Add tables/figures for the report
5. Extend the project toward dynamic FC

---

## Notes

- Static FGDN reproduction for AAL and HarvardOxford is complete under a strict protocol
- MODL_128 is not yet included
- Dynamic FC work has not started yet
- Current results are meaningful reproduction baselines, though still somewhat below the paper's best reported numbers
