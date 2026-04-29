# Microbial_Genomics_ML

Machine learning pipeline for geographic source attribution of Shiga-toxigenic *E. coli* (STEC) from whole genome sequencing kmer data. Trains and evaluates Random Forest, Gradient Boosted Decision Trees (GBDT), and LinearSVC classifiers to predict country and region of origin from kmer frequency counts.

## Overview

- Parameter testing across rare kmer removal, UK class capping, and chi-squared feature selection
- Random oversampling (ROS) and no rebalancing conditions evaluated for all models
- Final models trained on 2014-2018 data and evaluated on held-out 2019 test set
- Macro F1 used as primary evaluation metric

## Scripts

| Script | Description |
|--------|-------------|
| `ML_models_remove_rare.py` | With rare kmer removal and normalisation |
| `ML_models_no_removal.py` | Baseline — no rare removal, n=100, k=2000 |
| `ML_models_n_300.py` | UK cap n=300, k=2000 |
| `ML_models_n_500.py` | UK cap n=500, k=2000 |
| `ML_models_n_nocap.py` | No UK cap, k=2000 |
| `ML_models_k_5000.py` | n=300, k=5000 |
| `ML_models_k_10000.py` | n=300, k=10000 |
| `ML_models_k_nocap.py` | n=300, no feature selection |
| `ML_models_Final.py` | Final model — trained on 2014-2018, evaluated on 2019 |
| `ML_visualizations.py` | Parameter comparison figures and heatmaps |

## Outputs

- `ML_outputs/` — Summary CSVs and classification report Excel files for each parameter condition
- `ML_figures/` — Parameter comparison figures (rare removal, N cap, feature selection, heatmaps) and final model confusion matrices and feature importance plots

## Dependencies
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
openpyxl

## Data

Data is not included in this repository. Input files required:
- `14-18kmerdata.txt` — kmer frequency matrix (2014-2018)
- `14-18metadata` — sample metadata with Country and Region labels
- `19kmerdata.txt` — kmer frequency matrix (2019)
- `19metadata` — 2019 sample metadata

Update the `filepath` variable at the top of each script to point to your data directory.
