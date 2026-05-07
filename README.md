# FGDN Project

This repository contains my implementation and reproduction of the paper:

**Identification of Autism Spectrum Disorder With Functional Graph Discriminative Network (FGDN)**  
Li et al., *Frontiers in Neuroscience*, 2021.

The main goal of this project is to classify **Autism Spectrum Disorder (ASD)** and **Healthy Control (HC)** subjects using **ABIDE resting-state fMRI ROI time-series** and a graph neural network model based on the FGDN architecture.

I have completed the static functional connectivity version of the project using two atlases:

- **AAL**
- **HarvardOxford**

I also implemented and evaluated two model variants:

- **Unweighted FGDN**
- **Weighted FGDN**

The next extension of this project is to add a clean ablation-study framework without disturbing the completed baseline results.

---

## 1. Current Project Status

### Completed

- ABIDE phenotypic and ROI time-series download pipeline
- Dataset verification
- AAL and HarvardOxford subject matching
- Leakage-safe tangent functional connectivity pipeline
- Stratified 5-fold and 10-fold CV splits
- Fold-specific ASD/HC graph template construction
- PyTorch Geometric dataset creation
- Unweighted FGDN training and evaluation
- Weighted FGDN training and evaluation
- Cross-validation result summarization
- Report figures and presentation figures
- ACM-style LaTeX report
- Beamer presentation

### Current extension in progress

I am now extending the project with **ablation studies**. The goal is to vary important parameters such as KNN neighbours, Chebyshev order, hidden channels, dropout, learning rate, weight decay, batch size, and weighted vs unweighted graph templates while keeping the existing project files mostly unchanged.

---

## 2. Dataset

I used **ABIDE preprocessed resting-state fMRI ROI time-series**.

The current pipeline uses:

- Pipeline: `cpac`
- Strategy: `nofilt_noglobal`
- Input file format: `.1D` ROI time-series files

Each subject has a time-series matrix:

```text
X_i ∈ R^(T_i × N)
```

where:

- `T_i` = number of fMRI time points for subject `i`
- `N` = number of ROIs in the atlas

The label convention used in the project is:

```text
HC  = 0
ASD = 1
```

### Dataset summary

| Atlas | Matched Subjects | ROIs |
|---|---:|---:|
| AAL | 531 | 116 |
| HarvardOxford | 884 | 111 |

---

## 3. Methodology Summary

The overall pipeline is:

```text
ABIDE ROI time-series
        ↓
Subject bundle with labels and time-series
        ↓
Stratified 5-fold / 10-fold CV splits
        ↓
Fold-specific tangent functional connectivity
        ↓
ASD and HC KNN graph templates
        ↓
PyTorch Geometric datasets
        ↓
FGDN / Weighted FGDN training
        ↓
Fold-wise evaluation and CV summary
```

---

## 4. Important Corrections Already Applied

This project went through several corrections while reproducing the paper. These corrections are important because they affect the validity of the results.

### 4.1 Tangent leakage correction

Initially, tangent connectivity was being fit globally before cross-validation. That is not correct because the test subjects would influence the tangent reference.

The corrected behaviour is:

- For each CV fold, tangent connectivity is fit only on the training subjects.
- Test subjects are transformed using the tangent estimator fitted on the training subjects.
- The tangent reference is therefore fold-specific.

This means the test fold remains unseen during preprocessing.

### 4.2 Strict fold protocol

The corrected pipeline follows this protocol:

```text
Outer train/test split
        ↓
Inner train/monitor split inside outer train
        ↓
Tangent estimator fit on training data
        ↓
ASD/HC templates built from inner-train only
        ↓
Checkpoint selected using monitor loss
        ↓
Final evaluation on untouched outer test fold
```

### 4.3 Class/logit ordering correction

The label convention is:

```text
HC  = 0
ASD = 1
```

The model logits are ordered as:

```text
[HC_logit, ASD_logit]
```

This keeps `CrossEntropyLoss` consistent and makes:

```text
softmax(logits)[:, 1]
```

the ASD probability.

### 4.4 Module execution correction

Training and evaluation should be run with Python module syntax:

```bash
python -m src.training.train_fgdn ...
python -m src.evaluation.evaluate_fgdn ...
```

Direct script execution such as:

```bash
python src\training\train_fgdn.py ...
```

can cause import errors because Python may not resolve the `src` package correctly.

---

## 5. Tangent Functional Connectivity

For each subject, the ROI time-series is converted into a covariance matrix.

The sample covariance is:

```text
S_i(p, q) = cov(X_i[:, p], X_i[:, q])
```

Instead of directly using raw sample covariance, I use Ledoit-Wolf shrinkage covariance:

```text
Σ_i = (1 - α)S_i + αμI
```

where:

- `S_i` = sample covariance matrix
- `I` = identity matrix
- `μI` = shrinkage target
- `α` = shrinkage coefficient

Then a reference covariance is computed from the training subjects using the geometric mean:

```text
Σ_bar = argmin_Σ Σ_i d_R²(Σ, Σ_i)
```

where the Riemannian distance is:

```text
d_R(A, B) = || log(A^(-1/2) B A^(-1/2)) ||_F
```

Each subject covariance is projected to the tangent space:

```text
T_i^tan = log(Σ_bar^(-1/2) Σ_i Σ_bar^(-1/2))
```

The tangent matrix is used as the node feature matrix:

```text
X_graph = T_i^tan
```

So each ROI is a node, and each row of the tangent matrix is the feature vector for one ROI.

---

## 6. Graph Construction

For each fold, I compute class-specific mean functional connectivity matrices using only training data:

```text
F_ASD_bar = mean of ASD training FC matrices
F_HC_bar  = mean of HC training FC matrices
```

Then KNN graph templates are built:

```text
A_ASD = KNN(F_ASD_bar)
A_HC  = KNN(F_HC_bar)
```

The default number of KNN neighbours is:

```text
k = 20
```

For every subject, two graph views are created:

```text
G_ASD = (X, A_ASD)
G_HC  = (X, A_HC)
```

The same subject feature matrix `X` is used in both branches, but the graph structure is different.

---

## 7. FGDN Model

The FGDN model has two class-specific branches:

- ASD branch using `A_ASD`
- HC branch using `A_HC`

Each branch uses:

```text
ChebConv → PReLU → Dropout
        → ChebConv → PReLU → Dropout
        → Flatten → Linear → Branch score
```

The final logits are:

```text
[s_HC, s_ASD]
```

---

## 8. Weighted FGDN

I also implemented a weighted version of FGDN.

The unweighted model uses binary graph edges:

```text
A_ij ∈ {0, 1}
```

The weighted model uses Gaussian-style KNN edge weights:

```text
w_ij = exp( - d(i, j)^2 / (2θ^2) )
```

where:

- `d(i, j)` = distance between ROI feature vectors
- `θ` = scale parameter estimated from non-zero KNN distances

The aim of weighted FGDN is to preserve edge strength instead of using only binary connectivity.

---

## 9. Repository Structure

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
│   │   ├── connectivity_matrices/
│   │   ├── cv_splits/
│   │   └── graph_templates/
│   └── processed/
│       ├── pyg_datasets/
│       └── logs/
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
├── reports/
├── presentation/
└── docs/
```

---

## 10. Environment Setup

### 10.1 Clone the repository

```bash
git clone https://github.com/RohanSinha000821/FGDN-Project.git
cd FGDN-Project
```

### 10.2 Create and activate virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 10.3 Install dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 11. Data Download

### 11.1 Download ABIDE phenotypic file

```bash
python src\data\download_abide_data.py --download-phenotypic
```

### 11.2 Download AAL ROI time-series

```bash
python src\data\download_abide_data.py --derivative rois_aal --pipeline cpac --strategy nofilt_noglobal --out-subdir AAL
```

### 11.3 Download HarvardOxford ROI time-series

```bash
python src\data\download_abide_data.py --derivative rois_ho --pipeline cpac --strategy nofilt_noglobal --out-subdir HarvardOxford
```

---

## 12. Dataset Verification

```bash
python src\data\verify_abide.py
```

This checks phenotypic CSV presence, ROI file discovery, subject ID extraction, phenotype and ROI file matching, and ROI dimension consistency.

---

## 13. Full Pipeline Execution Order

### Step 1: Build subject time-series bundle

```bash
python src\data\build_connectivity.py --atlas all --kind tangent --save-timeseries-info
```

For tangent mode, global `connectivity.npy` is intentionally not created. Tangent connectivity is computed later inside each fold.

### Step 2: Create CV splits

```bash
python src\data\create_cv_splits.py --atlas all --folds 5 10 --seed 42
```

### Step 3: Build graph templates

```bash
python src\data\build_graph_templates.py --atlas all --kind tangent --folds 5 10 --k 20
```

### Step 4: Build PyTorch Geometric datasets

```bash
python src\data\build_pyg_datasets.py --atlas all --kind tangent --folds 5 10
```

---

## 14. Training and Evaluation

### 14.1 Unweighted FGDN

Train one fold:

```bash
python -m src.training.train_fgdn --atlas AAL --kind tangent --num-folds 5 --fold 1
```

Evaluate one fold:

```bash
python -m src.evaluation.evaluate_fgdn --atlas AAL --kind tangent --num-folds 5 --fold 1 --checkpoint-type best
```

Summarize all folds:

```bash
python -m src.evaluation.summarize_fgdn_cv --atlas AAL --num-folds 5 --checkpoint-type best
```

### 14.2 Weighted FGDN

Train one fold:

```bash
python -m src.training.train_fgdn_weighted --atlas AAL --kind tangent --num-folds 5 --fold 1
```

Evaluate one fold:

```bash
python -m src.evaluation.evaluate_fgdn_weighted --atlas AAL --kind tangent --num-folds 5 --fold 1 --checkpoint-type best
```

Summarize all folds:

```bash
python -m src.evaluation.summarize_fgdn_cv_weighted --atlas AAL --num-folds 5 --checkpoint-type best
```

For 5-fold CV, run folds 1 to 5. For 10-fold CV, run folds 1 to 10.

---

## 15. Baseline Experimental Setup

These are the baseline values I used before ablation studies:

| Parameter | Value |
|---|---:|
| KNN neighbours | 20 |
| Chebyshev order | 3 |
| Hidden channels | 64 |
| Dropout | 0.1 |
| Batch size | 16 |
| Learning rate | 0.0001 |
| Weight decay | 0.0005 |
| Optimizer | Adam |
| CV settings | 5-fold, 10-fold |

---

## 16. Results Obtained

### 16.1 Unweighted FGDN

| Atlas | Folds | Accuracy Mean | Accuracy Std | AUC Mean | AUC Std |
|---|---:|---:|---:|---:|---:|
| AAL | 5 | 0.657150 | 0.052083 | 0.733335 | 0.039354 |
| AAL | 10 | 0.662858 | 0.058932 | 0.747608 | 0.066337 |
| HarvardOxford | 5 | 0.671970 | 0.016687 | 0.732362 | 0.011817 |
| HarvardOxford | 10 | 0.692352 | 0.050286 | 0.739556 | 0.038684 |

### 16.2 Weighted FGDN

| Atlas | Folds | Accuracy Mean | Accuracy Std | AUC Mean | AUC Std |
|---|---:|---:|---:|---:|---:|
| AAL | 5 | 0.664750 | 0.042253 | 0.746564 | 0.031682 |
| AAL | 10 | 0.670335 | 0.038649 | 0.760592 | 0.061851 |
| HarvardOxford | 5 | 0.656112 | 0.030566 | 0.722158 | 0.023552 |
| HarvardOxford | 10 | 0.677605 | 0.034057 | 0.744680 | 0.032152 |

### 16.3 Best settings

Best AUC:

```text
AAL 10-fold Weighted FGDN
Accuracy = 0.670335
AUC      = 0.760592
```

Best accuracy:

```text
HarvardOxford 10-fold Unweighted FGDN
Accuracy = 0.692352
AUC      = 0.739556
```

---

## 17. Result Interpretation

From the completed runs:

- Weighted templates improved AUC for AAL.
- HarvardOxford unweighted gave the best accuracy.
- AAL 10-fold weighted FGDN gave the best AUC overall.
- HarvardOxford 10-fold unweighted FGDN gave the best accuracy overall.
- 10-fold CV generally gave stronger or more stable results than 5-fold CV.
- Atlas choice clearly affected performance.

---

## 18. Ablation Study Extension

The next goal is to add ablation studies while keeping the current baseline files and results unchanged.

The ablation results should be stored separately:

```text
outputs/ablations/
```

Recommended structure:

```text
outputs/ablations/
├── configs/
├── AAL/
│   ├── knn/
│   ├── cheb_k/
│   ├── hidden_channels/
│   ├── dropout/
│   ├── learning_rate/
│   ├── weight_decay/
│   ├── batch_size/
│   └── weighted_vs_unweighted/
├── HarvardOxford/
│   ├── knn/
│   ├── cheb_k/
│   ├── hidden_channels/
│   ├── dropout/
│   ├── learning_rate/
│   ├── weight_decay/
│   ├── batch_size/
│   └── weighted_vs_unweighted/
└── summaries/
    ├── ablation_results.csv
    ├── ablation_results.md
    ├── ablation_auc_plot.png
    └── ablation_accuracy_plot.png
```

### Planned ablations

| Ablation | Values | Purpose |
|---|---|---|
| KNN neighbours | 5, 10, 20, 30, 40 | Test graph sparsity/density |
| Chebyshev order | 1, 2, 3, 4, 5 | Test graph receptive field |
| Hidden channels | 16, 32, 64, 128 | Test model capacity |
| Dropout | 0.0, 0.1, 0.3, 0.5 | Test regularization |
| Learning rate | 1e-5, 5e-5, 1e-4, 5e-4 | Test optimizer stability |
| Weight decay | 0, 1e-5, 5e-4, 1e-3 | Test L2 regularization |
| Batch size | 8, 16, 32 | Test training stability |
| Template type | binary, weighted | Test edge-weight usefulness |
| Atlas | AAL, HarvardOxford | Test atlas sensitivity |
| Connectivity type | tangent, correlation | Test tangent vs simpler FC, only if safely supported |

### Practical ablation strategy

To avoid running too many expensive experiments at once:

1. Start with AAL 5-fold fold 1 only.
2. Run quick single-fold ablations for the main parameters.
3. Select promising settings.
4. Run full 5-fold CV for the promising settings.
5. Run 10-fold CV only for the strongest final candidates.

This keeps the ablation process manageable.

---

## 19. Suggested Future Work

The main future direction is dynamic functional connectivity.

Instead of one static connectivity matrix per subject, the time-series can be divided into temporal windows, producing a sequence of connectivity matrices. A future model can combine graph convolution with temporal modelling to learn how functional connectivity changes over time.

Other possible extensions:

- MODL_128 integration
- Dynamic FC
- Site-aware evaluation
- Stronger regularization
- Better hyperparameter tuning
- Atlas-coordinate-based brain visualization

---

## 20. What Should Not Be Committed

This repository should not include large generated files such as:

- raw ABIDE data
- downloaded ROI time-series files
- large `.npy` arrays
- PyTorch Geometric `.pt` datasets
- checkpoint files
- large logs

These should be generated locally.

Recommended to keep in Git:

- source code
- README
- lightweight summary tables
- report and presentation source files
- small figures used in report/presentation
- configuration files for reproducible experiments

---

## 21. Final Rebuild Order

If I clone the project again and rebuild from scratch, the order is:

```text
1. download_abide_data.py
2. verify_abide.py
3. build_connectivity.py
4. create_cv_splits.py
5. build_graph_templates.py
6. build_pyg_datasets.py
7. train_fgdn.py
8. evaluate_fgdn.py
9. summarize_fgdn_cv.py
10. train_fgdn_weighted.py
11. evaluate_fgdn_weighted.py
12. summarize_fgdn_cv_weighted.py
```

---

## 22. Final Notes

This repository now contains a working FGDN reproduction pipeline for ABIDE ASD classification using AAL and HarvardOxford ROI time-series. The completed baseline experiments are saved separately from the planned ablation extension. The next major step is to add a configuration-driven ablation runner so that parameter studies can be run cleanly without disturbing the current baseline code or results.
