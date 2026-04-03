# NeuroSpiral

**Clifford torus embedding for multi-periodic signal analysis**

NeuroSpiral applies Takens delay embedding and Clifford torus geometry to extract interpretable features from periodic and quasi-periodic signals. Originally developed for sleep polysomnography, the framework generalizes across multiple domains.

## Key Findings

### Sleep Geometry
- The winding number omega1 orders sleep phases as a continuous gradient: N3 < N2 < REM < N1 < W
- Replicates across 4 independent datasets (HMC n=142, CAP n=34, Sleep-EDF, DREAMT)
- Geometric position parameter **beta = 0.57 [0.52, 0.62]** quantifies REM's location on the Wake-to-N3 axis
- EMG torus features discriminate REM behavior disorder with Cohen's d = 1.4-1.7 (FDR p < 0.001)
- Classification: kappa = 0.648 (5-class), matching a 3-layer MLP (kappa = 0.606 vs 0.593) with full interpretability
- Persistent homology confirms the torus is a computational tool, not data topology (beta_1 != 2, p = 0.667)

### Cross-Domain Validation
Torus features outperform PCA across all tested domains (mean ratio 2.1x):

| Domain | kappa (torus) | kappa (PCA) | Ratio |
|--------|-----------|---------|-------|
| Solar Activity | 0.701 | 0.315 | 2.2x |
| Sleep (PSG) | 0.607 | 0.264 | 2.3x |
| Climate | 0.537 | 0.209 | 2.6x |
| Traffic | 0.532 | 0.454 | 1.2x |
| Finance | 0.359 | 0.151 | 2.4x |

## Method

```
Signal (1D) -> Takens embedding (4D) -> Clifford torus (theta, phi) -> 8 geometric features
```

### The 8 torus features per channel
1. **omega1** -- winding number (trajectory speed)
2. **kappa_torus** -- curvature (trajectory shape)
3. **alpha_mean** -- angular acceleration (dynamics)
4. **d_geo** -- geodesic distance (path length)
5. **H_angular** -- angular entropy (occupancy uniformity)
6. **sigma_dphi** -- phase difference std (phase relationship)
7. **R** -- phase coherence (synchronization)
8. **lambda_trans** -- transition rate (regime changes)

### Architecture
- **Form A** (recommended): Independent torus per signal channel
- Embedding dimension: d = 4 (validated via false nearest neighbors, FNN = 1.8%)
- Multi-scale tau: 3 values per channel for temporal range coverage

## Installation

```bash
git clone https://github.com/xaron98/NeuroSpiral.git
cd NeuroSpiral
pip install -r requirements.txt
```

## Quick Start

### Python API

```python
from neurospiral import TorusEmbedding

# Single channel, single tau
torus = TorusEmbedding(d=4, tau=25)
features = torus.extract_features(signal_epoch)  # -> (8,)

# Multi-scale
torus_ms = TorusEmbedding(d=4, taus=[10, 25, 40])
features_ms = torus_ms.extract_features(signal_epoch)  # -> (24,)

# Multi-channel (Form A)
features_mc = torus_ms.extract_features_multichannel({
    "EEG": eeg_signal,
    "ECG": ecg_signal,
    "EOG": eog_signal,
    "EMG": emg_signal,
})  # -> (96,)
```

### Universal Torus (any domain)

```bash
python scripts/torus_universal.py data.csv --label-col class --epoch-size 1000
```

## Project Structure

```
NeuroSpiral/
├── neurospiral/           # Core Python package
│   ├── embedding.py       # Takens delay embedding
│   ├── torus.py           # Clifford torus projection + 8 features
│   ├── decomposition.py   # Beta, gamma/d, class decomposition
│   ├── classifier.py      # RF classification pipeline
│   └── utils.py           # Spectral features, FDR correction
├── scripts/
│   ├── sleep/             # Sleep staging analyses
│   ├── cross_domain/      # Climate, finance, ECG, solar
│   └── torus_universal.py # Apply to any CSV
├── docs/
│   ├── METHODS.md         # Mathematical framework
│   ├── DATASETS.md        # Data sources
│   └── REPRODUCIBILITY.md # Step-by-step replication
├── tests/                 # Unit tests
├── results/               # Analysis outputs (not tracked)
└── src/                   # Legacy source (deprecated)
```

## Datasets

All datasets are publicly available. Download from:
- **Sleep:** [HMC](https://physionet.org/content/hmc-sleep-staging/), [CAP Sleep](https://physionet.org/content/capslpdb/), [Sleep-EDF](https://physionet.org/content/sleep-edfx/), [DREAMT](https://physionet.org/content/dreamt/)
- **ECG:** [PTB-XL](https://physionet.org/content/ptb-xl/)

See [docs/DATASETS.md](docs/DATASETS.md) for complete details.

## Reproducibility

All results can be reproduced with the scripts in `scripts/`. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for step-by-step instructions.

## Citation

```bibtex
@article{perea2026neurospiral,
  title={Toroidal embedding of multimodal polysomnography reveals a
         reproducible geometric gradient of sleep states},
  author={Perea Gallego, Carlos Javier},
  year={2026},
  url={https://github.com/xaron98/NeuroSpiral}
}
```

## Author

**Carlos Javier Perea Gallego**
Independent Researcher, Mataro, Barcelona, Spain
GitHub: [@xaron98](https://github.com/xaron98)

## License

MIT License. See [LICENSE](LICENSE) for details.
