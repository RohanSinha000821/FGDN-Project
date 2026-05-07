# FGDN Project — Full Reproduction and Run Guide

Implementation and reproduction workflow for the paper:

**Identification of Autism Spectrum Disorder With Functional Graph Discriminative Network (FGDN)**

This document is the full working reference for this repository: project structure, dataset setup, exact execution order, commands to run, corrected methodology notes, and the final results obtained in the current codebase.

---

## 1. Project Objective

This repository reproduces the **FGDN** pipeline on **ABIDE preprocessed rs-fMRI ROI time-series data** for ASD vs HC classification.

Current scope:
- Static functional connectivity reproduction
- Atlases used:
  - **AAL**
  - **HarvardOxford**
- Model variants:
  - **Unweighted FGDN**
  - **Weighted FGDN**

Planned future scope:
- **MODL_128** integration
- **Dynamic functional connectivity (dynamic FC)** extension
- Report / paper writing and visualizations

---

## 2. Important Methodology Corrections Applied

The current codebase is **not** the original initial pipeline. It includes important corrections made during sanity-checking.

### 2.1 Tangent leakage fix
Earlier, tangent connectivity was being fit globally before cross-validation, which is incorrect.

**Corrected behavior now:**
- `build_connectivity.py` stores a leakage-free **subject time-series bundle**
- Tangent connectivity is fit **inside each outer training fold only**
- Outer test subjects are transformed using the estimator fit on outer-train only

### 2.2 Strict fold protocol
Current results use:

- outer train / outer test split from CV
- inner-train / monitor split inside each outer training fold
- ASD/HC templates rebuilt from **inner-train only**
- checkpoint chosen by **lowest monitor loss**
- final evaluation on untouched outer test fold

### 2.3 Class/logit ordering fix
Project labels are:

- **HC = 0**
- **ASD = 1**

The model output was corrected so logits are ordered as:

- `[HC_logit, ASD_logit]`

This keeps:
- `CrossEntropyLoss` correct
- `softmax(logits)[:, 1]` = ASD probability

### 2.4 Module execution fix
Training/evaluation scripts should be run using:

```bash
python -m src.training.train_fgdn ...
python -m src.evaluation.evaluate_fgdn ...
```

and **not** via direct script path execution like:

```bash
python src\training\train_fgdn.py ...
```

because direct script execution caused `ModuleNotFoundError: No module named 'src'`.

---

## 3. Repository Structure

```text
FGDN_Project/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── abide/
│   │       ├── phenotypic/
│   │       ├── roi_timeseries/
│   │       │   ├── AAL/
│   │       │   └── HarvardOxford/
│   │       └── downloads/
│   ├── interim/
│   │   ├── subject_timeseries/
│   │   │   ├── AAL/
│   │   │   └── HarvardOxford/
│   │   ├── connectivity_matrices/
│   │   │   ├── AAL/
│   │   │   │   └── tangent/
│   │   │   └── HarvardOxford/
│   │   │       └── tangent/
│   │   ├── cv_splits/
│   │   │   ├── AAL/
│   │   │   └── HarvardOxford/
│   │   └── graph_templates/
│   │       ├── AAL/
│   │       │   └── tangent/
│   │       └── HarvardOxford/
│   │           └── tangent/
│   └── processed/
│       ├── pyg_datasets/
│       │   ├── AAL/
│       │   │   └── tangent/
│       │   └── HarvardOxford/
│       │       └── tangent/
│       └── logs/
│           ├── fgdn/
│           └── fgdn_weighted/
├── outputs/
│   ├── checkpoints/
│   │   ├── fgdn/
│   │   └── fgdn_weighted/
│   └── tables/
│       ├── fgdn/
│       └── fgdn_weighted/
├── src/
│   ├── data/
│   │   ├── download_abide_data.py
│   │   ├── verify_abide.py
│   │   ├── build_connectivity.py
│   │   ├── create_cv_splits.py
│   │   ├── build_graph_templates.py
│   │   └── build_pyg_datasets.py
│   ├── models/
│   │   ├── fgdn_model.py
│   │   └── fgdn_model_weighted.py
│   ├── training/
│   │   ├── train_fgdn.py
│   │   └── train_fgdn_weighted.py
│   └── evaluation/
│       ├── evaluate_fgdn.py
│       ├── evaluate_fgdn_weighted.py
│       ├── summarize_fgdn_cv.py
│       └── summarize_fgdn_cv_weighted.py
└── docs/
```

---

## 4. Environment Setup

### 4.1 Clone repository

```bash
git clone https://github.com/RohanSinha000821/FGDN-Project.git
cd FGDN-Project
```

### 4.2 Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 4.3 Install dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 5. Data Download

### 5.1 Download ABIDE phenotypic CSV

```bash
python src\data\download_abide_data.py --download-phenotypic
```

### 5.2 Download ROI time-series

#### AAL

```bash
python src\data\download_abide_data.py --derivative rois_aal --pipeline cpac --strategy nofilt_noglobal --out-subdir AAL
```

#### HarvardOxford

```bash
python src\data\download_abide_data.py --derivative rois_ho --pipeline cpac --strategy nofilt_noglobal --out-subdir HarvardOxford
```

---

## 6. Dataset Verification

Run:

```bash
python src\data\verify_abide.py
```

What this checks:
- phenotypic CSV presence
- ROI file discovery
- subject ID extraction
- phenotype ↔ ROI matching
- ROI dimension sanity check

---

## 7. Verified Dataset Used

### AAL
- Subjects: **531**
- ROI dimension: **116**

### HarvardOxford
- Subjects: **884**
- ROI dimension: **111**

> Important: the HarvardOxford derivative used here has **111 ROIs**.

---

## 8. Correct Execution Order

This is the correct order for the **current corrected pipeline**.

### Step 1 — Build leakage-free subject bundle / tangent stub

```bash
python src\data\build_connectivity.py --atlas all --kind tangent --save-timeseries-info
```

This creates:
- `data/interim/subject_timeseries/<atlas>/...`
- `data/interim/connectivity_matrices/<atlas>/tangent/...`

For tangent mode, **`connectivity.npy` is intentionally not created globally**.

### Step 2 — Create CV splits

```bash
python src\data\create_cv_splits.py --atlas all --folds 5 10 --seed 42
```

### Step 3 — Build graph templates

```bash
python src\data\build_graph_templates.py --atlas all --kind tangent --folds 5 10 --k 20
```

### Step 4 — Build PyTorch Geometric datasets

```bash
python src\data\build_pyg_datasets.py --atlas all --kind tangent --folds 5 10
```

### Step 5 — Train FGDN  
### Step 6 — Evaluate FGDN  
### Step 7 — Summarize CV results

---

## 9. Core Commands by File

### 9.1 Data preparation files

#### `download_abide_data.py`
Downloads phenotypic CSV and ROI time-series derivatives.

Phenotypic CSV:
```bash
python src\data\download_abide_data.py --download-phenotypic
```

AAL:
```bash
python src\data\download_abide_data.py --derivative rois_aal --pipeline cpac --strategy nofilt_noglobal --out-subdir AAL
```

HarvardOxford:
```bash
python src\data\download_abide_data.py --derivative rois_ho --pipeline cpac --strategy nofilt_noglobal --out-subdir HarvardOxford
```

#### `verify_abide.py`
```bash
python src\data\verify_abide.py
```

#### `build_connectivity.py`
```bash
python src\data\build_connectivity.py --atlas all --kind tangent --save-timeseries-info
```

#### `create_cv_splits.py`
```bash
python src\data\create_cv_splits.py --atlas all --folds 5 10 --seed 42
```

#### `build_graph_templates.py`
```bash
python src\data\build_graph_templates.py --atlas all --kind tangent --folds 5 10 --k 20
```

#### `build_pyg_datasets.py`
```bash
python src\data\build_pyg_datasets.py --atlas all --kind tangent --folds 5 10
```

---

## 10. Training Commands

### 10.1 Unweighted FGDN — AAL 5-fold

Train:
```bash
python -m src.training.train_fgdn --atlas AAL --kind tangent --num-folds 5 --fold 1
python -m src.training.train_fgdn --atlas AAL --kind tangent --num-folds 5 --fold 2
python -m src.training.train_fgdn --atlas AAL --kind tangent --num-folds 5 --fold 3
python -m src.training.train_fgdn --atlas AAL --kind tangent --num-folds 5 --fold 4
python -m src.training.train_fgdn --atlas AAL --kind tangent --num-folds 5 --fold 5
```

Evaluate:
```bash
python -m src.evaluation.evaluate_fgdn --atlas AAL --kind tangent --num-folds 5 --fold 1 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn --atlas AAL --kind tangent --num-folds 5 --fold 2 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn --atlas AAL --kind tangent --num-folds 5 --fold 3 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn --atlas AAL --kind tangent --num-folds 5 --fold 4 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn --atlas AAL --kind tangent --num-folds 5 --fold 5 --checkpoint-type best
```

Summarize:
```bash
python -m src.evaluation.summarize_fgdn_cv --atlas AAL --num-folds 5 --checkpoint-type best
```

### 10.2 Unweighted FGDN — AAL 10-fold

Train:
```bash
python -m src.training.train_fgdn --atlas AAL --kind tangent --num-folds 10 --fold 1
python -m src.training.train_fgdn --atlas AAL --kind tangent --num-folds 10 --fold 2
python -m src.training.train_fgdn --atlas AAL --kind tangent --num-folds 10 --fold 3
python -m src.training.train_fgdn --atlas AAL --kind tangent --num-folds 10 --fold 4
python -m src.training.train_fgdn --atlas AAL --kind tangent --num-folds 10 --fold 5
python -m src.training.train_fgdn --atlas AAL --kind tangent --num-folds 10 --fold 6
python -m src.training.train_fgdn --atlas AAL --kind tangent --num-folds 10 --fold 7
python -m src.training.train_fgdn --atlas AAL --kind tangent --num-folds 10 --fold 8
python -m src.training.train_fgdn --atlas AAL --kind tangent --num-folds 10 --fold 9
python -m src.training.train_fgdn --atlas AAL --kind tangent --num-folds 10 --fold 10
```

Evaluate:
```bash
python -m src.evaluation.evaluate_fgdn --atlas AAL --kind tangent --num-folds 10 --fold 1 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn --atlas AAL --kind tangent --num-folds 10 --fold 2 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn --atlas AAL --kind tangent --num-folds 10 --fold 3 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn --atlas AAL --kind tangent --num-folds 10 --fold 4 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn --atlas AAL --kind tangent --num-folds 10 --fold 5 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn --atlas AAL --kind tangent --num-folds 10 --fold 6 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn --atlas AAL --kind tangent --num-folds 10 --fold 7 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn --atlas AAL --kind tangent --num-folds 10 --fold 8 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn --atlas AAL --kind tangent --num-folds 10 --fold 9 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn --atlas AAL --kind tangent --num-folds 10 --fold 10 --checkpoint-type best
```

Summarize:
```bash
python -m src.evaluation.summarize_fgdn_cv --atlas AAL --num-folds 10 --checkpoint-type best
```

### 10.3 Unweighted FGDN — HarvardOxford 5-fold

Train:
```bash
python -m src.training.train_fgdn --atlas HarvardOxford --kind tangent --num-folds 5 --fold 1
python -m src.training.train_fgdn --atlas HarvardOxford --kind tangent --num-folds 5 --fold 2
python -m src.training.train_fgdn --atlas HarvardOxford --kind tangent --num-folds 5 --fold 3
python -m src.training.train_fgdn --atlas HarvardOxford --kind tangent --num-folds 5 --fold 4
python -m src.training.train_fgdn --atlas HarvardOxford --kind tangent --num-folds 5 --fold 5
```

Evaluate:
```bash
python -m src.evaluation.evaluate_fgdn --atlas HarvardOxford --kind tangent --num-folds 5 --fold 1 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn --atlas HarvardOxford --kind tangent --num-folds 5 --fold 2 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn --atlas HarvardOxford --kind tangent --num-folds 5 --fold 3 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn --atlas HarvardOxford --kind tangent --num-folds 5 --fold 4 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn --atlas HarvardOxford --kind tangent --num-folds 5 --fold 5 --checkpoint-type best
```

Summarize:
```bash
python -m src.evaluation.summarize_fgdn_cv --atlas HarvardOxford --num-folds 5 --checkpoint-type best
```

### 10.4 Unweighted FGDN — HarvardOxford 10-fold

Train:
```bash
python -m src.training.train_fgdn --atlas HarvardOxford --kind tangent --num-folds 10 --fold 1
python -m src.training.train_fgdn --atlas HarvardOxford --kind tangent --num-folds 10 --fold 2
python -m src.training.train_fgdn --atlas HarvardOxford --kind tangent --num-folds 10 --fold 3
python -m src.training.train_fgdn --atlas HarvardOxford --kind tangent --num-folds 10 --fold 4
python -m src.training.train_fgdn --atlas HarvardOxford --kind tangent --num-folds 10 --fold 5
python -m src.training.train_fgdn --atlas HarvardOxford --kind tangent --num-folds 10 --fold 6
python -m src.training.train_fgdn --atlas HarvardOxford --kind tangent --num-folds 10 --fold 7
python -m src.training.train_fgdn --atlas HarvardOxford --kind tangent --num-folds 10 --fold 8
python -m src.training.train_fgdn --atlas HarvardOxford --kind tangent --num-folds 10 --fold 9
python -m src.training.train_fgdn --atlas HarvardOxford --kind tangent --num-folds 10 --fold 10
```

Evaluate:
```bash
python -m src.evaluation.evaluate_fgdn --atlas HarvardOxford --kind tangent --num-folds 10 --fold 1 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn --atlas HarvardOxford --kind tangent --num-folds 10 --fold 2 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn --atlas HarvardOxford --kind tangent --num-folds 10 --fold 3 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn --atlas HarvardOxford --kind tangent --num-folds 10 --fold 4 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn --atlas HarvardOxford --kind tangent --num-folds 10 --fold 5 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn --atlas HarvardOxford --kind tangent --num-folds 10 --fold 6 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn --atlas HarvardOxford --kind tangent --num-folds 10 --fold 7 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn --atlas HarvardOxford --kind tangent --num-folds 10 --fold 8 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn --atlas HarvardOxford --kind tangent --num-folds 10 --fold 9 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn --atlas HarvardOxford --kind tangent --num-folds 10 --fold 10 --checkpoint-type best
```

Summarize:
```bash
python -m src.evaluation.summarize_fgdn_cv --atlas HarvardOxford --num-folds 10 --checkpoint-type best
```

### 10.5 Weighted FGDN — AAL 5-fold

Train:
```bash
python -m src.training.train_fgdn_weighted --atlas AAL --kind tangent --num-folds 5 --fold 1
python -m src.training.train_fgdn_weighted --atlas AAL --kind tangent --num-folds 5 --fold 2
python -m src.training.train_fgdn_weighted --atlas AAL --kind tangent --num-folds 5 --fold 3
python -m src.training.train_fgdn_weighted --atlas AAL --kind tangent --num-folds 5 --fold 4
python -m src.training.train_fgdn_weighted --atlas AAL --kind tangent --num-folds 5 --fold 5
```

Evaluate:
```bash
python -m src.evaluation.evaluate_fgdn_weighted --atlas AAL --kind tangent --num-folds 5 --fold 1 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn_weighted --atlas AAL --kind tangent --num-folds 5 --fold 2 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn_weighted --atlas AAL --kind tangent --num-folds 5 --fold 3 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn_weighted --atlas AAL --kind tangent --num-folds 5 --fold 4 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn_weighted --atlas AAL --kind tangent --num-folds 5 --fold 5 --checkpoint-type best
```

Summarize:
```bash
python -m src.evaluation.summarize_fgdn_cv_weighted --atlas AAL --num-folds 5 --checkpoint-type best
```

### 10.6 Weighted FGDN — AAL 10-fold

Train:
```bash
python -m src.training.train_fgdn_weighted --atlas AAL --kind tangent --num-folds 10 --fold 1
python -m src.training.train_fgdn_weighted --atlas AAL --kind tangent --num-folds 10 --fold 2
python -m src.training.train_fgdn_weighted --atlas AAL --kind tangent --num-folds 10 --fold 3
python -m src.training.train_fgdn_weighted --atlas AAL --kind tangent --num-folds 10 --fold 4
python -m src.training.train_fgdn_weighted --atlas AAL --kind tangent --num-folds 10 --fold 5
python -m src.training.train_fgdn_weighted --atlas AAL --kind tangent --num-folds 10 --fold 6
python -m src.training.train_fgdn_weighted --atlas AAL --kind tangent --num-folds 10 --fold 7
python -m src.training.train_fgdn_weighted --atlas AAL --kind tangent --num-folds 10 --fold 8
python -m src.training.train_fgdn_weighted --atlas AAL --kind tangent --num-folds 10 --fold 9
python -m src.training.train_fgdn_weighted --atlas AAL --kind tangent --num-folds 10 --fold 10
```

Evaluate:
```bash
python -m src.evaluation.evaluate_fgdn_weighted --atlas AAL --kind tangent --num-folds 10 --fold 1 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn_weighted --atlas AAL --kind tangent --num-folds 10 --fold 2 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn_weighted --atlas AAL --kind tangent --num-folds 10 --fold 3 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn_weighted --atlas AAL --kind tangent --num-folds 10 --fold 4 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn_weighted --atlas AAL --kind tangent --num-folds 10 --fold 5 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn_weighted --atlas AAL --kind tangent --num-folds 10 --fold 6 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn_weighted --atlas AAL --kind tangent --num-folds 10 --fold 7 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn_weighted --atlas AAL --kind tangent --num-folds 10 --fold 8 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn_weighted --atlas AAL --kind tangent --num-folds 10 --fold 9 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn_weighted --atlas AAL --kind tangent --num-folds 10 --fold 10 --checkpoint-type best
```

Summarize:
```bash
python -m src.evaluation.summarize_fgdn_cv_weighted --atlas AAL --num-folds 10 --checkpoint-type best
```

### 10.7 Weighted FGDN — HarvardOxford 5-fold

Train:
```bash
python -m src.training.train_fgdn_weighted --atlas HarvardOxford --kind tangent --num-folds 5 --fold 1
python -m src.training.train_fgdn_weighted --atlas HarvardOxford --kind tangent --num-folds 5 --fold 2
python -m src.training.train_fgdn_weighted --atlas HarvardOxford --kind tangent --num-folds 5 --fold 3
python -m src.training.train_fgdn_weighted --atlas HarvardOxford --kind tangent --num-folds 5 --fold 4
python -m src.training.train_fgdn_weighted --atlas HarvardOxford --kind tangent --num-folds 5 --fold 5
```

Evaluate:
```bash
python -m src.evaluation.evaluate_fgdn_weighted --atlas HarvardOxford --kind tangent --num-folds 5 --fold 1 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn_weighted --atlas HarvardOxford --kind tangent --num-folds 5 --fold 2 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn_weighted --atlas HarvardOxford --kind tangent --num-folds 5 --fold 3 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn_weighted --atlas HarvardOxford --kind tangent --num-folds 5 --fold 4 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn_weighted --atlas HarvardOxford --kind tangent --num-folds 5 --fold 5 --checkpoint-type best
```

Summarize:
```bash
python -m src.evaluation.summarize_fgdn_cv_weighted --atlas HarvardOxford --num-folds 5 --checkpoint-type best
```

### 10.8 Weighted FGDN — HarvardOxford 10-fold

Train:
```bash
python -m src.training.train_fgdn_weighted --atlas HarvardOxford --kind tangent --num-folds 10 --fold 1
python -m src.training.train_fgdn_weighted --atlas HarvardOxford --kind tangent --num-folds 10 --fold 2
python -m src.training.train_fgdn_weighted --atlas HarvardOxford --kind tangent --num-folds 10 --fold 3
python -m src.training.train_fgdn_weighted --atlas HarvardOxford --kind tangent --num-folds 10 --fold 4
python -m src.training.train_fgdn_weighted --atlas HarvardOxford --kind tangent --num-folds 10 --fold 5
python -m src.training.train_fgdn_weighted --atlas HarvardOxford --kind tangent --num-folds 10 --fold 6
python -m src.training.train_fgdn_weighted --atlas HarvardOxford --kind tangent --num-folds 10 --fold 7
python -m src.training.train_fgdn_weighted --atlas HarvardOxford --kind tangent --num-folds 10 --fold 8
python -m src.training.train_fgdn_weighted --atlas HarvardOxford --kind tangent --num-folds 10 --fold 9
python -m src.training.train_fgdn_weighted --atlas HarvardOxford --kind tangent --num-folds 10 --fold 10
```

Evaluate:
```bash
python -m src.evaluation.evaluate_fgdn_weighted --atlas HarvardOxford --kind tangent --num-folds 10 --fold 1 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn_weighted --atlas HarvardOxford --kind tangent --num-folds 10 --fold 2 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn_weighted --atlas HarvardOxford --kind tangent --num-folds 10 --fold 3 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn_weighted --atlas HarvardOxford --kind tangent --num-folds 10 --fold 4 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn_weighted --atlas HarvardOxford --kind tangent --num-folds 10 --fold 5 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn_weighted --atlas HarvardOxford --kind tangent --num-folds 10 --fold 6 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn_weighted --atlas HarvardOxford --kind tangent --num-folds 10 --fold 7 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn_weighted --atlas HarvardOxford --kind tangent --num-folds 10 --fold 8 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn_weighted --atlas HarvardOxford --kind tangent --num-folds 10 --fold 9 --checkpoint-type best
python -m src.evaluation.evaluate_fgdn_weighted --atlas HarvardOxford --kind tangent --num-folds 10 --fold 10 --checkpoint-type best
```

Summarize:
```bash
python -m src.evaluation.summarize_fgdn_cv_weighted --atlas HarvardOxford --num-folds 10 --checkpoint-type best
```

---

## 11. Results Obtained in the Current Corrected Pipeline

### 11.1 Unweighted FGDN

| Atlas | Folds | Accuracy Mean | Accuracy Std | AUC Mean | AUC Std |
|---|---:|---:|---:|---:|---:|
| AAL | 5 | 0.657150 | 0.052083 | 0.733335 | 0.039354 |
| AAL | 10 | 0.662858 | 0.058932 | 0.747608 | 0.066337 |
| HarvardOxford | 5 | 0.671970 | 0.016687 | 0.732362 | 0.011817 |
| HarvardOxford | 10 | 0.692352 | 0.050286 | 0.739556 | 0.038684 |

### 11.2 Weighted FGDN

| Atlas | Folds | Accuracy Mean | Accuracy Std | AUC Mean | AUC Std |
|---|---:|---:|---:|---:|---:|
| AAL | 5 | 0.664750 | 0.042253 | 0.746564 | 0.031682 |
| AAL | 10 | 0.670335 | 0.038649 | 0.760592 | 0.061851 |
| HarvardOxford | 5 | 0.656112 | 0.030566 | 0.722158 | 0.023552 |
| HarvardOxford | 10 | 0.677605 | 0.034057 | 0.744680 | 0.032152 |

### 11.3 Best settings

#### Best by AUC
- **AAL, 10-fold, weighted**
- Accuracy = **0.670335**
- AUC = **0.760592**

#### Best by Accuracy
- **HarvardOxford, 10-fold, unweighted**
- Accuracy = **0.692352**
- AUC = **0.739556**

---

## 12. Interpretation of Results

### AAL
- Weighted templates helped AAL consistently
- Best AAL result came from **10-fold weighted**
- AAL achieved the **best AUC overall**

### HarvardOxford
- HarvardOxford unweighted performed better in accuracy
- Weighted templates did **not** give a clear consistent benefit
- HarvardOxford produced the **best accuracy overall**

### 5-fold vs 10-fold
- 10-fold generally improved or matched 5-fold performance
- 10-fold is the stronger evaluation setting in this project

---

## 13. Output Locations

### Checkpoints
```text
outputs/checkpoints/fgdn/...
outputs/checkpoints/fgdn_weighted/...
```

### Fold evaluation tables
```text
outputs/tables/fgdn/...
outputs/tables/fgdn_weighted/...
```

### Training logs
```text
data/processed/logs/fgdn/...
data/processed/logs/fgdn_weighted/...
```

### PyG datasets
```text
data/processed/pyg_datasets/<atlas>/tangent/<n_folds>_fold/fold_<k>/
```

---

## 14. What is Not Included in Git

This repository should **not** include:
- raw ABIDE dataset files
- ROI time-series files
- large generated arrays
- `.pt` PyG datasets
- large checkpoint files

These should be generated locally.

Recommended to keep in Git:
- source code
- lightweight summaries
- markdown / docs
- small logs if needed

---

## 15. Clean Rebuild Order

If rebuilding from scratch after cloning and downloading data, use this order:

1. `download_abide_data.py`
2. `verify_abide.py`
3. `build_connectivity.py`
4. `create_cv_splits.py`
5. `build_graph_templates.py`
6. `build_pyg_datasets.py`
7. `train_fgdn.py`
8. `evaluate_fgdn.py`
9. `summarize_fgdn_cv.py`
10. `train_fgdn_weighted.py`
11. `evaluate_fgdn_weighted.py`
12. `summarize_fgdn_cv_weighted.py`

---

## 16. Pending Work / Future Extensions

1. Integrate **MODL_128**
2. Run static FGDN on MODL_128
3. Compare AAL vs HarvardOxford vs MODL
4. Add final report tables and visualizations
5. Extend to **dynamic FC**
6. Explore whether paper-faithful loss/head changes improve performance further

---

## 17. Final Status

### Completed
- Folder structure setup
- Environment setup
- ABIDE data download pipeline
- Verification pipeline
- Corrected tangent-safe preprocessing pipeline
- CV split generation
- Graph template generation
- PyG dataset generation
- FGDN model implementation
- Weighted FGDN implementation
- Unweighted experiments:
  - AAL 5-fold
  - AAL 10-fold
  - HarvardOxford 5-fold
  - HarvardOxford 10-fold
- Weighted experiments:
  - AAL 5-fold
  - AAL 10-fold
  - HarvardOxford 5-fold
  - HarvardOxford 10-fold
- Result summarization and comparison

### Pending
- MODL_128
- Dynamic FC
- Final report writing / visualization polishing

---
