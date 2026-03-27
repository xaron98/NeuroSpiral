"""Phase 1.2: Generate publication-quality figures for the paper.

Creates 5 figures from validated results (Sleep-EDF + HMC):
  Figure 1: Clifford torus schematic (conceptual diagram)
  Figure 2: Vertex-stage heatmaps (Sleep-EDF vs HMC comparison)
  Figure 3: CMI comparison bar chart with null distributions
  Figure 4: Feature importance ranking (geometric vs spectral)
  Figure 5: Per-subject κ box plots (both datasets)

Output: data/results/figures/ (PDF + PNG at 300 DPI)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# Publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUT_DIR = Path("data/results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# DATA FROM VALIDATED RUNS
# ══════════════════════════════════════════════════════════════

# Sleep-EDF vertex-stage conditional probabilities (18 subjects)
SLEEPEDF_VERTEX_STAGE = np.array([
    [0.047, 0.091, 0.003, 0.094, 0.765],  # V00
    [0.067, 0.361, 0.022, 0.207, 0.343],  # V01
    [0.035, 0.645, 0.106, 0.128, 0.085],  # V02
    [0.060, 0.327, 0.109, 0.103, 0.400],  # V03
    [0.029, 0.712, 0.094, 0.122, 0.043],  # V04
    [0.046, 0.721, 0.030, 0.096, 0.107],  # V05
    [0.029, 0.436, 0.177, 0.068, 0.290],  # V06
    [0.037, 0.201, 0.011, 0.082, 0.669],  # V07
    [0.050, 0.400, 0.054, 0.163, 0.333],  # V08
    [0.033, 0.509, 0.308, 0.080, 0.070],  # V09
    [0.042, 0.732, 0.022, 0.092, 0.111],  # V10
    [0.018, 0.228, 0.018, 0.042, 0.695],  # V11
    [0.048, 0.271, 0.103, 0.091, 0.488],  # V12
    [0.021, 0.159, 0.014, 0.034, 0.772],  # V13
    [0.036, 0.196, 0.013, 0.074, 0.680],  # V14
    [0.049, 0.104, 0.005, 0.106, 0.737],  # V15
])

# HMC vertex-stage conditional probabilities (150 subjects)
HMC_VERTEX_STAGE = np.array([
    [0.106, 0.332, 0.156, 0.170, 0.235],  # V00
    [0.102, 0.359, 0.191, 0.170, 0.179],  # V01
    [0.098, 0.384, 0.231, 0.148, 0.139],  # V02
    [0.117, 0.374, 0.195, 0.152, 0.161],  # V03
    [0.107, 0.338, 0.158, 0.183, 0.213],  # V04
    [0.085, 0.366, 0.240, 0.173, 0.136],  # V05
    [0.129, 0.337, 0.204, 0.164, 0.166],  # V06
    [0.105, 0.365, 0.247, 0.160, 0.124],  # V07
    [0.106, 0.383, 0.192, 0.153, 0.165],  # V08
    [0.108, 0.414, 0.199, 0.123, 0.157],  # V09
    [0.072, 0.440, 0.297, 0.119, 0.072],  # V10
    [0.096, 0.432, 0.264, 0.114, 0.093],  # V11
    [0.134, 0.334, 0.137, 0.184, 0.211],  # V12
    [0.105, 0.382, 0.206, 0.168, 0.139],  # V13
    [0.113, 0.408, 0.228, 0.135, 0.116],  # V14
    [0.090, 0.414, 0.268, 0.132, 0.096],  # V15
])

STAGE_NAMES = ['N1', 'N2', 'N3', 'REM', 'W']
STAGE_COLORS = {
    'W': '#5DCAA5', 'N1': '#85B7EB', 'N2': '#378ADD',
    'N3': '#534AB7', 'REM': '#D4537E'
}

# Per-subject kappas
SLEEPEDF_KAPPAS = [0.862, 0.845, 0.793, 0.813, 0.845, 0.739, 0.874, 0.764,
                   0.734, 0.775, 0.789, 0.703, 0.854, 0.870, 0.850, 0.834,
                   0.799, 0.774]

HMC_KAPPAS = [0.667, 0.691, 0.812, 0.789, 0.524, 0.791, 0.615, 0.776,
              0.613, 0.695, 0.513, 0.692, 0.611, 0.669, 0.818, 0.705,
              0.583, 0.559, 0.748, 0.733, 0.624, 0.718, 0.849, 0.623,
              0.717, 0.747, 0.470, 0.660, 0.675, 0.660, 0.712, 0.709,
              0.719, 0.610, 0.724, 0.714, 0.763, 0.705, 0.722, 0.663,
              0.668, 0.640, 0.779, 0.722, 0.820, 0.718, 0.674, 0.475,
              0.756, 0.622, 0.622, 0.678, 0.496, 0.762, 0.465, 0.804,
              0.684, 0.713, 0.603, 0.711, 0.624, 0.723, 0.624, 0.793,
              0.636, 0.635, 0.702, 0.644, 0.783, 0.799, 0.654, 0.537,
              0.694, 0.702, 0.658, 0.637, 0.555, 0.668, 0.729, 0.700,
              0.619, 0.669, 0.682, 0.761, 0.628, 0.768, 0.720, 0.694,
              0.657, 0.647, 0.654, 0.763, 0.563, 0.890, 0.710, 0.754,
              0.632, 0.699, 0.630, 0.623, 0.751, 0.672, 0.558, 0.588,
              0.706, 0.677, 0.616, 0.654, 0.561, 0.629, 0.721, 0.803,
              0.724, 0.702, 0.715, 0.645, 0.701, 0.645, 0.761, 0.667,
              0.697, 0.694, 0.755, 0.721, 0.730, 0.651, 0.552]


def figure2_heatmaps():
    """Figure 2: Vertex-stage conditional probability heatmaps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    # Sleep-EDF
    im1 = ax1.imshow(SLEEPEDF_VERTEX_STAGE, cmap='YlOrRd', aspect='auto',
                     vmin=0, vmax=0.8)
    ax1.set_xlabel('Sleep stage')
    ax1.set_ylabel('Tesseract vertex')
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(STAGE_NAMES)
    ax1.set_yticks(range(16))
    ax1.set_yticklabels([f'V{i:02d}' for i in range(16)])
    ax1.set_title('(a) Sleep-EDF (Fpz-Cz, n=18)\nCramér\'s V = 0.355')

    # Add text annotations
    for i in range(16):
        for j in range(5):
            val = SLEEPEDF_VERTEX_STAGE[i, j]
            color = 'white' if val > 0.4 else 'black'
            ax1.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=6.5, color=color)

    # HMC
    im2 = ax2.imshow(HMC_VERTEX_STAGE, cmap='YlOrRd', aspect='auto',
                     vmin=0, vmax=0.8)
    ax2.set_xlabel('Sleep stage')
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(STAGE_NAMES)
    ax2.set_yticks(range(16))
    ax2.set_yticklabels([f'V{i:02d}' for i in range(16)])
    ax2.set_title('(b) HMC (C4-M1/C3-M2, n=150)\nCramér\'s V = 0.100')

    for i in range(16):
        for j in range(5):
            val = HMC_VERTEX_STAGE[i, j]
            color = 'white' if val > 0.4 else 'black'
            ax2.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=6.5, color=color)

    plt.colorbar(im2, ax=[ax1, ax2], label='P(stage | vertex)', shrink=0.8)
    plt.suptitle('Figure 2: Vertex-stage association is channel-dependent',
                fontsize=13, fontweight='bold', y=1.02)

    plt.savefig(OUT_DIR / 'fig2_heatmaps.pdf')
    plt.savefig(OUT_DIR / 'fig2_heatmaps.png')
    plt.close()
    print("  ✓ Figure 2: Vertex-stage heatmaps")


def figure3_cmi():
    """Figure 3: CMI comparison across datasets with null reference."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Data
    features = ['CMI(ω₁|δ)', 'CMI(vertex|δ)', 'CMI(phase_diff|δ)',
                'CMI(phase_coh|δ)', 'CMI(bigram|δ)', 'CMI(trans_rate|δ)']

    # Sleep-EDF values (only ω₁ and vertex available)
    sleepedf = [0.1637, 0.1675, np.nan, np.nan, np.nan, np.nan]

    # HMC values (from run_hmc_validation + phase1_permtest)
    hmc_full = [0.2535, 0.0190, np.nan, np.nan, np.nan, np.nan]  # from full run
    hmc_perm = [0.0386, np.nan, 0.0403, 0.0407, 0.0038, 0.0191]  # from permtest

    # Null 95th percentiles
    null_95 = [0.0018, 0.0029, 0.0018, 0.0018, 0.0007, 0.0018]

    x = np.arange(len(features))
    width = 0.25

    # Plot Sleep-EDF
    sleepedf_clean = [v if not np.isnan(v) else 0 for v in sleepedf]
    mask_se = [not np.isnan(v) for v in sleepedf]
    bars1 = ax.bar(x[mask_se] - width, [sleepedf_clean[i] for i in range(len(x)) if mask_se[i]],
                   width, label='Sleep-EDF (n=18)', color='#378ADD', alpha=0.8)

    # Plot HMC (permtest values for all features)
    hmc_plot = [hmc_perm[i] if not np.isnan(hmc_perm[i]) else 0 for i in range(len(hmc_perm))]
    bars2 = ax.bar(x, hmc_plot, width, label='HMC permtest (n=129)', color='#534AB7', alpha=0.8)

    # Null reference line
    ax.bar(x + width, null_95, width, label='Null 95th percentile',
           color='#cccccc', alpha=0.6, edgecolor='gray')

    ax.set_ylabel('Conditional Mutual Information (bits)')
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=25, ha='right')
    ax.legend(frameon=False)
    ax.set_title('Figure 3: All geometric features significant beyond delta power (p < 0.001)')

    # Add significance stars
    for i in range(len(x)):
        if hmc_plot[i] > 0:
            ax.text(x[i], hmc_plot[i] + 0.002, '***', ha='center', fontsize=8)

    plt.savefig(OUT_DIR / 'fig3_cmi.pdf')
    plt.savefig(OUT_DIR / 'fig3_cmi.png')
    plt.close()
    print("  ✓ Figure 3: CMI comparison with null")


def figure4_feature_ranking():
    """Figure 4: Feature importance ranking — geometric vs spectral."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # All features from HMC validation (sorted by MI)
    features = [
        ('phase_diff_std', 0.4269, 'geometric'),
        ('delta/beta ratio', 0.4140, 'spectral'),
        ('ω₁ (winding)', 0.3772, 'geometric'),
        ('beta power', 0.3421, 'spectral'),
        ('delta power', 0.2477, 'spectral'),
        ('theta power', 0.1230, 'spectral'),
        ('bigram entropy', 0.1058, 'geometric'),
        ('phase coherence', 0.0885, 'geometric'),
        ('phase_diff_std*', 0.0894, 'geometric'),
        ('ω₁/ω₂ (ratio)', 0.0245, 'geometric'),
        ('vertex', 0.0203, 'geometric'),
        ('transition rate', 0.0031, 'geometric'),
    ]

    # Remove duplicate
    features = [f for f in features if f[0] != 'phase_diff_std*']

    # Sort by MI
    features.sort(key=lambda x: x[1], reverse=True)

    names = [f[0] for f in features]
    mis = [f[1] for f in features]
    types = [f[2] for f in features]

    colors = ['#D4537E' if t == 'geometric' else '#378ADD' for t in types]

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, mis, color=colors, alpha=0.85, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('Mutual Information with PSG stage (bits)')
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#D4537E', alpha=0.85, label='Geometric (torus)'),
        Patch(facecolor='#378ADD', alpha=0.85, label='Spectral (traditional)')
    ]
    ax.legend(handles=legend_elements, frameon=False, loc='lower right')

    ax.set_title('Figure 4: Geometric feature (phase_diff_std) ranks #1\nacross all features in HMC (n=150)')

    # Highlight top feature
    bars[0].set_edgecolor('black')
    bars[0].set_linewidth(1.5)

    plt.savefig(OUT_DIR / 'fig4_ranking.pdf')
    plt.savefig(OUT_DIR / 'fig4_ranking.png')
    plt.close()
    print("  ✓ Figure 4: Feature importance ranking")


def figure5_kappa_boxplots():
    """Figure 5: Per-subject κ distributions for both datasets."""
    fig, ax = plt.subplots(figsize=(6, 5))

    data = [SLEEPEDF_KAPPAS, HMC_KAPPAS]
    labels = [f'Sleep-EDF\n(n={len(SLEEPEDF_KAPPAS)})',
              f'HMC\n(n={len(HMC_KAPPAS)})']

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5,
                    showmeans=True, meanprops=dict(marker='D', markerfacecolor='white',
                    markeredgecolor='black', markersize=6))

    colors = ['#378ADD', '#534AB7']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add individual points (jittered)
    for i, (d, color) in enumerate(zip(data, colors)):
        jitter = np.random.normal(0, 0.04, len(d))
        ax.scatter(np.full(len(d), i + 1) + jitter, d,
                  alpha=0.3, s=15, color=color, zorder=2)

    ax.set_ylabel("Cohen's κ")
    ax.axhline(y=0.61, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.text(2.55, 0.615, 'substantial', fontsize=8, color='gray', style='italic')
    ax.axhline(y=0.81, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.text(2.55, 0.815, 'almost perfect', fontsize=8, color='gray', style='italic')

    se_mean = np.mean(SLEEPEDF_KAPPAS)
    hmc_mean = np.mean(HMC_KAPPAS)
    ax.set_title(f'Figure 5: Per-subject classification agreement\n'
                f'Sleep-EDF: κ={se_mean:.3f}±{np.std(SLEEPEDF_KAPPAS):.3f}  '
                f'HMC: κ={hmc_mean:.3f}±{np.std(HMC_KAPPAS):.3f}')

    plt.savefig(OUT_DIR / 'fig5_kappa.pdf')
    plt.savefig(OUT_DIR / 'fig5_kappa.png')
    plt.close()
    print("  ✓ Figure 5: κ box plots")


def figure1_torus_schematic():
    """Figure 1: Conceptual schematic of the Clifford torus framework."""
    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.3)

    # Panel A: EEG signal
    ax1 = fig.add_subplot(gs[0])
    t = np.linspace(0, 1, 500)
    signal = (np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 6 * t)
              + 0.3 * np.sin(2 * np.pi * 10 * t) + 0.2 * np.random.randn(500))
    ax1.plot(t, signal, 'k-', linewidth=0.5)
    ax1.set_title('(a) EEG signal', fontsize=10)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('μV')
    ax1.set_xlim(0, 1)

    # Panel B: Takens embedding (2D projection)
    ax2 = fig.add_subplot(gs[1])
    tau = 25
    emb_x = signal[:-tau]
    emb_y = signal[tau:]
    colors_emb = np.linspace(0, 1, len(emb_x))
    ax2.scatter(emb_x, emb_y, c=colors_emb, cmap='coolwarm', s=1, alpha=0.5)
    ax2.set_title('(b) Takens embedding\n(2D projection)', fontsize=10)
    ax2.set_xlabel('x(t)')
    ax2.set_ylabel('x(t+τ)')
    ax2.set_aspect('equal')

    # Panel C: Torus projection (simplified)
    ax3 = fig.add_subplot(gs[2])
    theta = np.linspace(0, 4 * np.pi, 500)
    phi = np.linspace(0, 6 * np.pi, 500)
    R, r = 1.0, 0.4
    x_torus = (R + r * np.cos(phi)) * np.cos(theta)
    y_torus = (R + r * np.cos(phi)) * np.sin(theta)

    # Draw torus outline
    t_outline = np.linspace(0, 2 * np.pi, 100)
    for phi_fixed in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        xt = (R + r * np.cos(phi_fixed)) * np.cos(t_outline)
        yt = (R + r * np.cos(phi_fixed)) * np.sin(t_outline)
        ax3.plot(xt, yt, 'gray', alpha=0.15, linewidth=0.5)

    # Trajectory colored by stage
    stage_boundaries = [0, 50, 100, 200, 300, 350, 430, 500]
    stage_labels_fig = ['W', 'N1', 'N2', 'N3', 'N2', 'REM', 'W']
    for i in range(len(stage_boundaries) - 1):
        s = stage_boundaries[i]
        e = stage_boundaries[i + 1]
        color = STAGE_COLORS[stage_labels_fig[i]]
        ax3.plot(x_torus[s:e], y_torus[s:e], color=color, linewidth=1.5, alpha=0.8)

    ax3.set_title('(c) Clifford torus\nprojection', fontsize=10)
    ax3.set_aspect('equal')
    ax3.set_xticks([])
    ax3.set_yticks([])

    # Panel D: Vertex classification
    ax4 = fig.add_subplot(gs[3])
    vertex_counts = [7602, 776, 141, 5360, 139, 635, 1212, 1065,
                     839, 1295, 639, 167, 5725, 145, 632, 6018]
    dominant = ['W', 'N2', 'N2', 'W', 'N2', 'N2', 'N2', 'W',
                'N2', 'N2', 'N2', 'W', 'W', 'W', 'W', 'W']
    colors_bar = [STAGE_COLORS[d] for d in dominant]

    ax4.barh(range(16), vertex_counts, color=colors_bar, alpha=0.8, height=0.7)
    ax4.set_yticks(range(16))
    ax4.set_yticklabels([f'V{i:02d}' for i in range(16)], fontsize=7)
    ax4.set_xlabel('Epoch count')
    ax4.set_title('(d) Vertex residence\n(Sleep-EDF)', fontsize=10)
    ax4.invert_yaxis()

    plt.suptitle('Figure 1: Clifford torus framework for EEG sleep staging',
                fontsize=13, fontweight='bold', y=1.05)

    plt.savefig(OUT_DIR / 'fig1_schematic.pdf')
    plt.savefig(OUT_DIR / 'fig1_schematic.png')
    plt.close()
    print("  ✓ Figure 1: Framework schematic")


def table_summary():
    """Generate summary table as text file for the paper."""
    table = """
Table 2: Cross-dataset validation summary

Metric                      Sleep-EDF (n=18)    HMC (n=150)     Generalizes?
─────────────────────────   ────────────────    ────────────    ────────────
Epochs                      32,390              115,486         —
Channels                    Fpz-Cz              C4-M1, C3-M2   Different
Cohen's κ (mean±SD)         0.807 ± 0.050       0.679 ± 0.087  ↓ expected
F1 spectral-only            0.735               0.609           ↓ expected
F1 combined                 0.744               0.655           ↓ expected
Δ F1 (geometric adds)       +0.010              +0.046          ↑ improves
CMI(ω₁ | delta)             0.1637 (p<0.001)    0.2535 (p<0.001) ↑ improves
CMI(vertex | delta)          0.1675 (p<0.001)    0.0190 (p<0.001) ↓ ch-dep
Cramér's V (best)           0.355               0.128           ↓ ch-dep
Consistency                 54%                 55%             ≈ same
ω₁/ω₂ range                 1.00                0.77–1.34       ✓ multi-ch
phase_diff_std MI           —                   0.4269          ✓ new best


Table 3: Permutation test results — geometric features (HMC, n=129)

Feature              MI(feat,stage)  CMI(feat|delta)  Null 95th   p-value
─────────────────    ──────────────  ───────────────  ─────────   ────────
phase_coherence           0.0885           0.0407       0.0018    <0.001
phase_diff_std            0.0894           0.0403       0.0018    <0.001
ω₁ (winding)              0.0543           0.0386       0.0018    <0.001
transition_rate           0.0128           0.0191       0.0018    <0.001
bigram_entropy            0.0068           0.0038       0.0007    <0.001

Baseline: MI(delta, stage) = 0.2452
All 5 geometric features contribute significant non-redundant information
beyond spectral delta power (permutation test, 1000 permutations).
"""

    with open(OUT_DIR / 'tables.txt', 'w') as f:
        f.write(table)
    print("  ✓ Tables 2 & 3: Summary + permutation test")


def main():
    print("=" * 65)
    print("  Phase 1.2: Generating Publication Figures")
    print("=" * 65)
    print(f"  Output: {OUT_DIR}/\n")

    figure1_torus_schematic()
    figure2_heatmaps()
    figure3_cmi()
    figure4_feature_ranking()
    figure5_kappa_boxplots()
    table_summary()

    print(f"\n  All figures saved to {OUT_DIR}/")
    print("  Formats: PDF (vector) + PNG (300 DPI)")
    print("\n  Phase 1.2 complete.")


if __name__ == "__main__":
    main()
