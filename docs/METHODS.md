# Methods

## Mathematical Framework

### Takens Delay Embedding

For a 1D signal x(t), the delay embedding maps:

```
v(t) = [x(t), x(t+τ), x(t+2τ), x(t+3τ)] ∈ ℝ⁴
```

By Takens' theorem (1981), for a d-dimensional dynamical system, an embedding in m ≥ 2d+1 dimensions produces a diffeomorphic reconstruction. We use d=4, validated by False Nearest Neighbors analysis (FNN = 1.8% at d=4).

The delay τ is estimated as the first local minimum of mutual information (Fraser & Swinney, 1986).

### Clifford Torus Projection

The 4D embedded point cloud is projected onto the Clifford torus T² = S¹ × S¹ ⊂ S³ via angle extraction:

```
θ = arctan2(v₂, v₁)    (first S¹)
φ = arctan2(v₄, v₃)    (second S¹)
```

This discards radial information (amplitudes r₁, r₂) and retains the angular dynamics.

### The 8 Torus Features

From the angular time series θ(t) and φ(t):

| # | Feature | Formula | Interpretation |
|---|---------|---------|----------------|
| 1 | ω₁ (omega1) | mean(|dθ/dt|) | Angular velocity / rotation speed |
| 2 | κ_torus (curvature) | mean(|d²θ/dt²|) | Trajectory shape / bending |
| 3 | α_mean (acceleration) | var(dθ/dt) | Speed variability |
| 4 | d_geo (geodesic distance) | Σ√(dθ² + dφ²) | Total path length on torus |
| 5 | H_angular (entropy) | -Σp·log₂(p) of θ in 16 bins | Occupancy uniformity |
| 6 | σ_Δφ (phase diff std) | circ_std(θ - φ) | Phase relationship variability |
| 7 | R (phase coherence) | |mean(e^{i(θ-φ)})| | Synchronization strength |
| 8 | λ_trans (transition rate) | frac(sgn changes) | Regime change frequency |

### Multi-Channel Architecture (Form A)

For multi-channel signals (e.g., EEG + ECG + EOG + EMG), each channel is embedded on its own independent torus. This preserves per-channel information that averaging would destroy.

4 channels × 3 τ values × 8 features = 96 torus features

### REM Decomposition

The geometric position parameter β quantifies where REM falls on the Wake→N3 axis:

```
μ_REM = α·μ_W + β·μ_N3 + γ·e_⊥
```

- β = 0: REM = Wake
- β = 1: REM = N3
- Observed: β = 0.57 [0.52, 0.62] (95% CI, bootstrap)

The residual γ/d = 0.42, meaning 42% of REM's geometry is orthogonal to the W→N3 axis.

## Preprocessing

| Channel | Bandpass (Hz) | Resample | Artifact |
|---------|--------------|----------|----------|
| EEG | 0.5 – 30.0 | 100 Hz | > 500 μV |
| ECG | 0.5 – 40.0 | 100 Hz | — |
| EOG | 0.3 – 35.0 | 100 Hz | — |
| EMG | 10.0 – 49.0 | 100 Hz | — |

Filtering: MNE-Python FIR (firwin design). Epoch duration: 30 seconds.

## Classification

Random Forest with subject-stratified cross-validation (StratifiedGroupKFold, 5 folds). Hyperparameters validated via grid search: n_estimators=300, max_depth=15, class_weight="balanced".

## Statistical Testing

All p-values corrected for multiple comparisons using Benjamini-Hochberg FDR. 128/128 features survive FDR at α = 0.001. 1204/1280 pairwise stage comparisons survive FDR at α = 0.05.
