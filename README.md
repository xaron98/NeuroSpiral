# NeuroSpiral

Clifford torus embedding of EEG for sleep stage classification.

## Overview

NeuroSpiral extracts geometric features from sleep EEG by projecting Takens time-delay embeddings onto the Clifford torus (S¹ × S¹ ⊂ S³ ⊂ ℝ⁴) and computing winding numbers, inter-hemispheric phase coherence, and tesseract vertex discretisation. Cross-dataset validation on 168 subjects from two independent polysomnography datasets (Sleep-EDF and HMC) demonstrates that continuous torus features capture 75–102% additional information beyond spectral delta power (p < 0.001, permutation test).

## Key Results

| Metric | Sleep-EDF (n=18) | HMC (n=150) |
|--------|:----------------:|:-----------:|
| CMI(ω₁ \| delta) | 0.1637 (p<0.001) | 0.2535 (p<0.001) |
| Δ F1 (geometric adds) | +0.009 | +0.046 |
| Cohen's κ | 0.807 ± 0.050 | 0.679 ± 0.087 |
| Best geometric feature MI | — | 0.4269 (phase_diff_std) |

## Installation

```bash
git clone https://github.com/xaron98/neurospiral.git
cd neurospiral
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Requirements

- Python 3.10+
- MNE-Python ≥ 1.5
- NumPy, SciPy, scikit-learn, pandas
- matplotlib (for figures)

## Reproducing Results

### 1. Download datasets

Both datasets are freely available from PhysioNet (credentialed access):

- [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/)
- [HMC Sleep Staging](https://physionet.org/content/hmc-sleep-staging/)

```bash
# Sleep-EDF (18 subjects)
bash scripts/download_sleep_edf.sh

# HMC (154 subjects)
bash scripts/download_hmc.sh
```

### 2. Run Sleep-EDF validation

```bash
python scripts/publish_validate.py --n-subjects 85 --output-dir data/results/publication
```

### 3. Run HMC cross-dataset validation

```bash
python scripts/run_hmc_validation.py --data-dir data/hmc --n-subjects 152
```

### 4. Run permutation tests for enhanced features

```bash
python scripts/phase1_permtest.py
```

### 5. Generate publication figures

```bash
python scripts/phase1_figures.py
```

## Project Structure

```
neurospiral/
├── src/
│   ├── features/
│   │   ├── spectral.py          # Band powers, Hjorth parameters
│   │   ├── takens.py            # Takens time-delay embedding
│   │   └── enhanced.py          # Multi-channel phase features
│   ├── geometry/
│   │   ├── tesseract.py         # 16 vertices, Q(x)=sgn(x), 24-cell
│   │   ├── wasserstein.py       # Bures-Wasserstein distance
│   │   └── alignment.py         # Procrustes alignment, fixed τ
│   ├── models/
│   │   └── tcn_torus.py         # TCN with torus regularisation
│   └── preprocessing/
│       └── pipeline.py          # Bandpass, ICA, quality check
├── scripts/
│   ├── publish_validate.py      # Main validation (Sleep-EDF)
│   ├── run_hmc_validation.py    # Cross-dataset validation (HMC)
│   ├── phase1_permtest.py       # Permutation tests
│   ├── phase1_figures.py        # Publication figures
│   └── download_*.sh            # Dataset download scripts
├── paper/
│   ├── paper_draft_v2.md        # Manuscript
│   └── tables_corrected.md      # Tables
└── data/
    └── results/
        └── figures/             # Generated figures (PDF + PNG)
```

## Mathematical Framework

The framework applies four transformations to each 30-second EEG epoch:

1. **Takens embedding** (d=4, τ=25 at 100 Hz): maps scalar EEG to ℝ⁴
2. **Clifford torus projection**: the embedded trajectory lies on T² = S¹ × S¹ ⊂ S³ (Gakhar & Perea, 2024)
3. **Winding number extraction**: ω₁, ω₂ measure rotation rates in two orthogonal planes
4. **Tesseract discretisation**: Q(x) = sgn(x) maps to 16 vertices of {±1}⁴

Theoretical justification: Gakhar and Perea (2024) proved that sliding-window embeddings of quasiperiodic functions are dense in N-tori, where N equals the number of linearly independent frequencies. Sleep EEG satisfies this condition.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{perea2026clifford,
  title={Clifford Torus Embedding of {EEG} Reveals Non-Redundant Geometric 
         Features for Sleep Stage Classification: Cross-Dataset Validation},
  author={Perea, Carlos},
  year={2026},
  journal={Submitted}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

Carlos Perea — xaron98@gmail.com
