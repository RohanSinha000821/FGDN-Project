# FGDN Project

Implementation and reproduction workflow for the paper:

**Identification of Autism Spectrum Disorder With Functional Graph Discriminative Network (FGDN)**

## Current Status

Completed:
- Project structure setup
- Python virtual environment setup
- PyTorch + PyTorch Geometric installation
- CUDA verification
- ABIDE phenotypic CSV download
- ROI timeseries download for:
  - AAL
  - HarvardOxford
- Dataset verification pipeline
- Subject matching verification for AAL and HarvardOxford

Pending:
- Connectivity matrix generation
- Cross-validation split generation
- Graph template construction
- PyTorch Geometric dataset generation
- FGDN model implementation
- Training and evaluation
- MODL_128 data sourcing
- Dynamic FC extension

## Project Structure

```text
FGDN_Project/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── abide/
│   │       ├── downloads/
│   │       ├── phenotypic/
│   │       └── roi_timeseries/
│   ├── interim/
│   │   ├── connectivity_matrices/
│   │   ├── cv_splits/
│   │   └── graph_templates/
│   └── processed/
│       ├── logs/
│       ├── pyg_datasets/
│       └── results/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── training/
│   └── utils/
├── outputs/
│   ├── checkpoints/
│   ├── figures/
│   └── tables/
└── docs/
    ├── paper/
    └── reference_code/