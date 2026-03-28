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
- AAL 5-fold baseline run

### Pending

- Stricter train/validation split inside each training fold
- HarvardOxford full baseline run
- MODL_128 data sourcing
- Dynamic FC extension
- Further paper-faithful refinements and tuning

---

## Current Baseline Result

### AAL 5-fold CV (current implementation)

- **Accuracy mean:** `0.557468`
- **Accuracy std:** `0.028092`
- **AUC mean:** `0.603745`
- **AUC std:** `0.022490`

### Fold-wise AAL results

- Fold 1: Accuracy `0.542056`, AUC `0.572378`
- Fold 2: Accuracy `0.603774`, AUC `0.621390`
- Fold 3: Accuracy `0.566038`, AUC `0.629234`
- Fold 4: Accuracy `0.556604`, AUC `0.581818`
- Fold 5: Accuracy `0.518868`, AUC `0.613904`

**Note:** The current setup selects the best checkpoint using the fold test set for debugging and baseline construction. The next methodological improvement is to introduce an internal validation split inside the training fold.

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

using **training subjects only**.

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

### Example: single-fold sanity run

```bash
python -m src.training.train_fgdn --atlas AAL --num-folds 5 --fold 1 --epochs 5 --batch-size 8
```

### Example: stronger single-fold run

```bash
python -m src.training.train_fgdn --atlas AAL --num-folds 5 --fold 1 --epochs 50 --batch-size 16
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

```bash
python -m src.evaluation.summarize_fgdn_cv --atlas AAL --num-folds 5 --checkpoint-type best
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
- logs
- virtual environments

All large data artifacts must be generated locally.

---

## Next Steps

1. Add internal validation split inside each training fold
2. Re-run cleaner AAL 5-fold evaluation with validation-based checkpoint selection
3. Run HarvardOxford baseline
4. Search for MODL_128 source
5. Improve faithfulness to original FGDN details
6. Extend the project toward dynamic FC

---

## Notes

- AAL and HarvardOxford are verified and ready
- MODL_128 is not yet included
- The current pipeline is fully runnable end to end
- The current AAL baseline is above chance, but still below strong reproduction quality
