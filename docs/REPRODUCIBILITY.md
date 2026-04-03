# Reproducibility Guide

## Prerequisites

```bash
pip install -r requirements.txt
```

## Step 1: Download Data

Download HMC from PhysioNet (requires credentialed access):
```bash
mkdir -p data/hmc
# Download from https://physionet.org/content/hmc-sleep-staging/
```

## Step 2: Combined Feature Extraction

```bash
python scripts/sleep/extract_combined_features.py --data-dir data/hmc
```

This produces `results/combined_features.npz` with:
- `torus_individual`: (N, 96) — 4 channels × 3 τ × 8 features
- `spectral`: (N, 32) — 4 channels × 8 spectral features
- `stages`: (N,) — AASM labels
- `subjects`: (N,) — subject IDs

## Step 3: Core Analyses

```bash
# Classification and feature importance
python scripts/sleep/classify_stages.py

# Beta robustness sweep
python scripts/sleep/beta_analysis.py

# FDR correction
python scripts/sleep/fdr_correction.py

# Embedding comparison (torus vs sphere vs cylinder vs PCA)
python scripts/sleep/embedding_comparison.py

# Hypno-PC comparison (Guendelman & Shriki 2025)
python scripts/sleep/hypno_pc_comparison.py
```

## Step 4: Cross-Domain Validation

```bash
python scripts/cross_domain/climate.py
python scripts/cross_domain/finance.py
python scripts/cross_domain/ecg_pathology.py
```

## Step 5: Universal Torus (any domain)

```bash
python scripts/torus_universal.py your_data.csv --label-col class
```

## Expected Results

| Analysis | Expected |
|----------|----------|
| Sleep 5-class κ | 0.607 ± 0.01 |
| All 128 features survive FDR | Yes (p < 0.001) |
| ω₁ gradient: N3 < N2 < REM < N1 < W | Yes |
| β = 0.57 [0.52, 0.62] | Yes |
| Torus beats PCA ratio | ~2.3× |

## Notes

- All cross-validations use StratifiedGroupKFold (subject-stratified)
- Random states are fixed (42) for reproducibility
- Large .npz files are not tracked in git — regenerate from scripts
