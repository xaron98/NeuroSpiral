#!/usr/bin/env python3
"""Unsupervised analysis of sleep structure using only geometric features.

Loads results/hmc_features.npz (117,571 epochs, 146 subjects, 16 features)
and uses ONLY the 8 geometric features (columns 8–15) for clustering.
Stage labels are used only for evaluation, never during clustering.

Four analyses:
  1. Unsupervised geometric clustering (KMeans k=5, GMM k=5, optimal k search)
  2. Disagreement analysis: what geometry sees that AASM does not
  3. Geometric sleep depth score (PCA + interpretable formula)
  4. Optimal number of geometric states (k=2..10)

Outputs:
  results/figures/geometric_clusters_vs_aasm.png
  results/figures/optimal_k.png
  results/figures/geometric_depth_overnight.png
  results/figures/geometric_depth_correlation.png
  results/figures/cluster_feature_profiles.png
  results/geometric_clustering.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

STAGES = ["W", "N1", "N2", "N3", "REM"]
STAGE_TO_INT = {s: i for i, s in enumerate(STAGES)}

GEOM_NAMES = [
    "omega1", "torus_curvature", "angular_acceleration",
    "geodesic_distance", "angular_entropy", "phase_diff_std",
    "phase_coherence", "transition_rate",
]

SPECTRAL_NAMES = [
    "delta", "theta", "alpha", "sigma", "beta",
    "delta_beta", "hjorth_activity", "hjorth_mobility",
]

# Colors matching CLAUDE.md design
STAGE_COLORS = {
    "W": "#f5c842", "N1": "#5b8bd4", "N2": "#5b8bd4",
    "N3": "#1a1a6e", "REM": "#6e3fa0",
}
CLUSTER_COLORS = plt.cm.Set2(np.linspace(0, 1, 8))


# ===================================================================
# Load data
# ===================================================================
def load_data():
    npz_path = PROJECT_ROOT / "results" / "hmc_features.npz"
    data = np.load(npz_path)
    features = data["features"]         # (N, 16)
    stages = data["stages"]             # (N,) int8
    subjects = data["subjects"]         # (N,) int16

    spectral = features[:, :8]          # columns 0-7
    geometric = features[:, 8:]         # columns 8-15

    print(f"  Loaded {len(stages):,} epochs, {len(np.unique(subjects))} subjects")
    print(f"  Geometric features: {geometric.shape[1]}")
    print(f"  Stage distribution: "
          + ", ".join(f"{STAGES[i]}={c}" for i, c in
                      zip(*np.unique(stages, return_counts=True))))

    return spectral, geometric, stages, subjects


# ===================================================================
# Analysis 1: Unsupervised geometric clustering
# ===================================================================
def map_clusters_to_stages(cluster_labels: np.ndarray,
                           true_labels: np.ndarray,
                           n_clusters: int) -> tuple[dict, float]:
    """Map each cluster to its dominant AASM stage by majority vote.

    Returns (mapping dict, accuracy).
    """
    mapping = {}
    for c in range(n_clusters):
        mask = cluster_labels == c
        if not np.any(mask):
            mapping[c] = -1
            continue
        counts = np.bincount(true_labels[mask], minlength=5)
        mapping[c] = int(np.argmax(counts))

    predicted = np.array([mapping[c] for c in cluster_labels])
    accuracy = float(np.mean(predicted == true_labels))
    return mapping, accuracy


def analysis_1(geom_z: np.ndarray, stages: np.ndarray) -> dict:
    """KMeans k=5, GMM k=5, ARI, NMI, confusion matrix."""
    print("\n" + "=" * 70)
    print("Analysis 1: Unsupervised Geometric Clustering")
    print("=" * 70)

    results = {}

    # --- KMeans k=5 ---
    km5 = KMeans(n_clusters=5, n_init=20, random_state=42)
    km5_labels = km5.fit_predict(geom_z)

    ari_km5 = adjusted_rand_score(stages, km5_labels)
    nmi_km5 = normalized_mutual_info_score(stages, km5_labels)
    mapping_km5, acc_km5 = map_clusters_to_stages(km5_labels, stages, 5)

    print(f"  KMeans k=5:  ARI={ari_km5:.4f}  NMI={nmi_km5:.4f}  "
          f"Mapping accuracy={acc_km5:.4f}")
    print(f"  Cluster->Stage mapping: "
          + ", ".join(f"C{c}->{STAGES[s]}" for c, s in sorted(mapping_km5.items())))

    results["kmeans_k5"] = {
        "ARI": round(ari_km5, 4),
        "NMI": round(nmi_km5, 4),
        "mapping_accuracy": round(acc_km5, 4),
        "cluster_to_stage": {str(c): STAGES[s] for c, s in mapping_km5.items()},
    }

    # --- GMM k=5 ---
    gmm5 = GaussianMixture(n_components=5, covariance_type="full",
                            n_init=5, random_state=42)
    gmm5_labels = gmm5.fit_predict(geom_z)
    gmm5_probs = gmm5.predict_proba(geom_z)

    ari_gmm = adjusted_rand_score(stages, gmm5_labels)
    nmi_gmm = normalized_mutual_info_score(stages, gmm5_labels)
    mapping_gmm, acc_gmm = map_clusters_to_stages(gmm5_labels, stages, 5)

    # Mean max probability (confidence)
    mean_confidence = float(np.mean(np.max(gmm5_probs, axis=1)))
    # Entropy of assignments
    entropy = float(-np.mean(np.sum(gmm5_probs * np.log(gmm5_probs + 1e-10), axis=1)))

    print(f"  GMM k=5:     ARI={ari_gmm:.4f}  NMI={nmi_gmm:.4f}  "
          f"Mapping accuracy={acc_gmm:.4f}")
    print(f"  GMM mean confidence={mean_confidence:.3f}  "
          f"mean entropy={entropy:.3f}")

    results["gmm_k5"] = {
        "ARI": round(ari_gmm, 4),
        "NMI": round(nmi_gmm, 4),
        "mapping_accuracy": round(acc_gmm, 4),
        "cluster_to_stage": {str(c): STAGES[s] for c, s in mapping_gmm.items()},
        "mean_confidence": round(mean_confidence, 4),
        "mean_entropy": round(entropy, 4),
    }

    return results, km5_labels, mapping_km5, gmm5_probs


def plot_confusion_matrix(km5_labels, mapping, stages, fig_dir):
    """Plot confusion matrix mapping geometric clusters to AASM stages."""
    # Reorder clusters to match stage order for cleaner visualization
    # Build confusion matrix: rows=AASM stages, cols=geometric clusters
    cm = confusion_matrix(stages, km5_labels, labels=list(range(5)))
    # Normalize by row (per AASM stage)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Reorder columns so mapped stage order is diagonal-ish
    col_order = sorted(range(5), key=lambda c: mapping.get(c, c))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    ax = axes[0]
    im = ax.imshow(cm[:, col_order], cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"C{c}\n({STAGES[mapping[c]]})" for c in col_order], fontsize=9)
    ax.set_yticks(range(5))
    ax.set_yticklabels(STAGES, fontsize=10)
    ax.set_xlabel("Geometric Cluster (mapped stage)", fontsize=10)
    ax.set_ylabel("AASM Stage", fontsize=10)
    ax.set_title("Counts", fontsize=11)
    for i in range(5):
        for j in range(5):
            val = cm[i, col_order[j]]
            color = "white" if val > cm.max() * 0.6 else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center",
                    fontsize=8, color=color)
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Normalized (row-wise)
    ax = axes[1]
    im = ax.imshow(cm_norm[:, col_order], cmap="YlGnBu", aspect="auto",
                   vmin=0, vmax=1)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"C{c}\n({STAGES[mapping[c]]})" for c in col_order], fontsize=9)
    ax.set_yticks(range(5))
    ax.set_yticklabels(STAGES, fontsize=10)
    ax.set_xlabel("Geometric Cluster (mapped stage)", fontsize=10)
    ax.set_title("Row-normalized", fontsize=11)
    for i in range(5):
        for j in range(5):
            val = cm_norm[i, col_order[j]]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color)
    plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Geometric Clusters vs AASM Stages (KMeans k=5)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "geometric_clusters_vs_aasm.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved geometric_clusters_vs_aasm.png")


# ===================================================================
# Analysis 2: Disagreement analysis
# ===================================================================
def analysis_2(geom_z: np.ndarray, spectral: np.ndarray, geometric: np.ndarray,
               km5_labels: np.ndarray, mapping: dict,
               stages: np.ndarray) -> dict:
    """Identify and characterize epochs where geometry disagrees with AASM."""
    print("\n" + "=" * 70)
    print("Analysis 2: What Geometry Sees That AASM Does Not")
    print("=" * 70)

    predicted_stages = np.array([mapping[c] for c in km5_labels])
    agree_mask = predicted_stages == stages
    disagree_mask = ~agree_mask

    n_agree = int(np.sum(agree_mask))
    n_disagree = int(np.sum(disagree_mask))
    print(f"  Agreement: {n_agree:,} ({n_agree/len(stages)*100:.1f}%)")
    print(f"  Disagreement: {n_disagree:,} ({n_disagree/len(stages)*100:.1f}%)")

    results = {
        "n_agree": n_agree,
        "n_disagree": n_disagree,
        "agreement_rate": round(n_agree / len(stages), 4),
        "disagreement_types": {},
    }

    # Analyze top disagreement types
    disagree_types: dict[tuple, list] = {}
    for idx in np.where(disagree_mask)[0]:
        key = (int(predicted_stages[idx]), int(stages[idx]))
        if key not in disagree_types:
            disagree_types[key] = []
        disagree_types[key].append(idx)

    # Sort by count
    sorted_types = sorted(disagree_types.items(), key=lambda x: -len(x[1]))

    print(f"\n  Top disagreement types:")
    print(f"  {'Geom->':>10s} {'AASM':>6s} {'Count':>8s} {'%':>6s}")
    print(f"  {'-'*35}")

    all_features = np.hstack([spectral, geometric])  # (N, 16)

    for (geo_stage, aasm_stage), indices in sorted_types[:10]:
        indices = np.array(indices)
        pct = len(indices) / len(stages) * 100
        print(f"  {STAGES[geo_stage]:>10s} {STAGES[aasm_stage]:>6s} "
              f"{len(indices):>8,} {pct:>5.1f}%")

        # Mean features of disagreement epochs
        mean_feats = all_features[indices].mean(axis=0)
        # Compare with stage means
        geo_stage_mean = all_features[stages == geo_stage].mean(axis=0)
        aasm_stage_mean = all_features[stages == aasm_stage].mean(axis=0)

        # How intermediate are they? (0 = exactly like AASM stage, 1 = exactly
        # like geometric stage, 0.5 = perfectly intermediate)
        denom = np.abs(geo_stage_mean - aasm_stage_mean) + 1e-10
        intermediacy = np.mean(np.abs(mean_feats - aasm_stage_mean) / denom)

        results["disagreement_types"][f"{STAGES[geo_stage]}_vs_{STAGES[aasm_stage]}"] = {
            "count": len(indices),
            "pct": round(pct, 2),
            "intermediacy": round(float(intermediacy), 4),
            "mean_geometric": {
                name: round(float(v), 4)
                for name, v in zip(GEOM_NAMES, mean_feats[8:])
            },
        }

    # Summary: are disagreement epochs intermediate?
    if sorted_types:
        top_type = sorted_types[0]
        (geo_s, aasm_s), top_indices = top_type
        top_indices = np.array(top_indices)
        geo_mean = geometric[stages == geo_s].mean(axis=0)
        aasm_mean = geometric[stages == aasm_s].mean(axis=0)
        dis_mean = geometric[top_indices].mean(axis=0)

        print(f"\n  Top disagreement ({STAGES[geo_s]} vs {STAGES[aasm_s]}) "
              f"geometric profile:")
        print(f"  {'Feature':<25s} {'AASM mean':>10s} {'Disagree':>10s} "
              f"{'Geom mean':>10s}")
        for i, name in enumerate(GEOM_NAMES):
            print(f"  {name:<25s} {aasm_mean[i]:>10.4f} {dis_mean[i]:>10.4f} "
                  f"{geo_mean[i]:>10.4f}")

    return results


# ===================================================================
# Analysis 3: Geometric sleep depth score
# ===================================================================
def analysis_3(geometric: np.ndarray, geom_z: np.ndarray,
               spectral: np.ndarray, stages: np.ndarray,
               subjects: np.ndarray, fig_dir: Path) -> dict:
    """Compute continuous geometric depth score and correlate with delta/AASM."""
    print("\n" + "=" * 70)
    print("Analysis 3: Geometric Sleep Depth Score")
    print("=" * 70)

    results = {}

    # --- PCA-based depth ---
    pca = PCA(n_components=1, random_state=42)
    pca_score = pca.fit_transform(geom_z).ravel()

    # Orient so deeper sleep (N3) has higher values
    # Check correlation with stage integers (W=0 < N3=3)
    nrem_mask = stages <= 3  # W, N1, N2, N3 only
    corr_with_stage = np.corrcoef(pca_score[nrem_mask], stages[nrem_mask])[0, 1]
    if corr_with_stage < 0:
        pca_score = -pca_score

    pca_var_explained = float(pca.explained_variance_ratio_[0])
    print(f"  PCA depth: {pca_var_explained*100:.1f}% variance explained")

    # --- Interpretable depth formula ---
    # Normalize individual features to [0, 1] range for the formula
    omega1_z = geom_z[:, 0]        # omega1
    phase_coh_z = geom_z[:, 6]     # phase_coherence
    torus_curv_z = geom_z[:, 1]    # torus_curvature

    formula_depth = -omega1_z + phase_coh_z - torus_curv_z
    # Orient
    corr_formula = np.corrcoef(formula_depth[nrem_mask], stages[nrem_mask])[0, 1]
    if corr_formula < 0:
        formula_depth = -formula_depth

    # --- Correlations ---
    delta_power = spectral[:, 0]  # column 0 = delta relative power

    # PCA depth vs delta
    r_pca_delta, p_pca_delta = pearsonr(pca_score, delta_power)
    # Formula depth vs delta
    r_form_delta, p_form_delta = pearsonr(formula_depth, delta_power)

    # Depth vs AASM integer (NREM only, excluding REM)
    r_pca_nrem, p_pca_nrem = pearsonr(pca_score[nrem_mask], stages[nrem_mask])
    r_form_nrem, p_form_nrem = pearsonr(formula_depth[nrem_mask], stages[nrem_mask])

    # Including REM (coded as 4)
    r_pca_all, p_pca_all = pearsonr(pca_score, stages)
    r_form_all, p_form_all = pearsonr(formula_depth, stages)

    print(f"\n  Correlations with delta power:")
    print(f"    PCA depth:     r={r_pca_delta:.4f}  p={p_pca_delta:.2e}")
    print(f"    Formula depth: r={r_form_delta:.4f}  p={p_form_delta:.2e}")
    print(f"\n  Correlations with AASM stage (NREM only, W=0..N3=3):")
    print(f"    PCA depth:     r={r_pca_nrem:.4f}  p={p_pca_nrem:.2e}")
    print(f"    Formula depth: r={r_form_nrem:.4f}  p={p_form_nrem:.2e}")
    print(f"\n  Correlations with AASM stage (all, incl REM=4):")
    print(f"    PCA depth:     r={r_pca_all:.4f}  p={p_pca_all:.2e}")
    print(f"    Formula depth: r={r_form_all:.4f}  p={p_form_all:.2e}")

    # PCA loadings
    loadings = pca.components_[0]
    print(f"\n  PCA loadings (depth direction):")
    for name, w in sorted(zip(GEOM_NAMES, loadings), key=lambda x: -abs(x[1])):
        print(f"    {name:<25s} {w:+.4f}")

    results["pca_depth"] = {
        "variance_explained": round(pca_var_explained, 4),
        "loadings": {name: round(float(w), 4)
                     for name, w in zip(GEOM_NAMES, loadings)},
        "corr_delta": round(r_pca_delta, 4),
        "corr_delta_p": float(p_pca_delta),
        "corr_nrem_stage": round(r_pca_nrem, 4),
        "corr_nrem_p": float(p_pca_nrem),
        "corr_all_stage": round(r_pca_all, 4),
        "corr_all_p": float(p_pca_all),
    }
    results["formula_depth"] = {
        "formula": "-omega1_z + phase_coherence_z - torus_curvature_z",
        "corr_delta": round(r_form_delta, 4),
        "corr_delta_p": float(p_form_delta),
        "corr_nrem_stage": round(r_form_nrem, 4),
        "corr_nrem_p": float(p_form_nrem),
        "corr_all_stage": round(r_form_all, 4),
        "corr_all_p": float(p_form_all),
    }

    # --- Plot: depth vs delta scatter ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Subsample for scatter (too many points)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(pca_score), size=min(5000, len(pca_score)), replace=False)

    stage_colors_arr = np.array([
        STAGE_COLORS[STAGES[s]] for s in stages[idx]
    ])

    ax = axes[0]
    ax.scatter(pca_score[idx], delta_power[idx], c=stage_colors_arr,
               alpha=0.3, s=4, edgecolors="none")
    ax.set_xlabel("PCA Geometric Depth", fontsize=10)
    ax.set_ylabel("Delta Power (relative)", fontsize=10)
    ax.set_title(f"PCA Depth vs Delta Power (r={r_pca_delta:.3f})", fontsize=11)
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.scatter(formula_depth[idx], delta_power[idx], c=stage_colors_arr,
               alpha=0.3, s=4, edgecolors="none")
    ax.set_xlabel("Formula Depth (-ω₁ + coherence - curvature)", fontsize=10)
    ax.set_ylabel("Delta Power (relative)", fontsize=10)
    ax.set_title(f"Formula Depth vs Delta Power (r={r_form_delta:.3f})", fontsize=11)
    ax.grid(alpha=0.2)

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=STAGE_COLORS[s], label=s) for s in STAGES]
    axes[1].legend(handles=legend_handles, fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(fig_dir / "geometric_depth_correlation.png", dpi=150)
    plt.close(fig)
    print(f"  Saved geometric_depth_correlation.png")

    # --- Plot: overnight trajectories for 3 subjects ---
    unique_subj = np.unique(subjects)
    # Pick 3 subjects with many epochs for good visualization
    subj_counts = [(s, np.sum(subjects == s)) for s in unique_subj]
    subj_counts.sort(key=lambda x: -x[1])
    plot_subjects = [s for s, _ in subj_counts[:3]]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

    for ax_i, subj_id in enumerate(plot_subjects):
        ax = axes[ax_i]
        mask = subjects == subj_id
        n_ep = int(np.sum(mask))
        time_hrs = np.arange(n_ep) * 30 / 3600  # epochs to hours

        subj_pca = pca_score[mask]
        subj_delta = delta_power[mask]
        subj_stages_arr = stages[mask]

        # Normalize depth and delta to [0, 1] for overlay
        d_min, d_max = subj_pca.min(), subj_pca.max()
        if d_max > d_min:
            depth_norm = (subj_pca - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(subj_pca)

        dl_min, dl_max = subj_delta.min(), subj_delta.max()
        if dl_max > dl_min:
            delta_norm = (subj_delta - dl_min) / (dl_max - dl_min)
        else:
            delta_norm = np.zeros_like(subj_delta)

        # Hypnogram (inverted: W at top, N3 at bottom)
        # Map stages to depth values for hypnogram line
        hyp_map = {0: 4, 1: 3, 2: 2, 3: 1, 4: 3.5}  # W=4, N1=3, N2=2, N3=1, REM=3.5
        hyp_vals = np.array([hyp_map[s] for s in subj_stages_arr])

        ax2 = ax.twinx()

        # Plot geometric depth and delta power
        ax.plot(time_hrs, depth_norm, color="#5bffa8", alpha=0.8,
                linewidth=0.8, label="Geometric depth")
        ax.plot(time_hrs, delta_norm, color="#5b8bd4", alpha=0.7,
                linewidth=0.8, label="Delta power")

        # Plot hypnogram as step function
        ax2.step(time_hrs, hyp_vals, color="#f5c842", alpha=0.6,
                 linewidth=1.5, where="post", label="Hypnogram")
        ax2.set_yticks([1, 2, 3, 3.5, 4])
        ax2.set_yticklabels(["N3", "N2", "N1", "R", "W"], fontsize=8)
        ax2.set_ylim(0.5, 4.8)
        ax2.tick_params(axis="y", colors="#f5c842")

        ax.set_ylim(-0.1, 1.1)
        ax.set_ylabel("Normalized score", fontsize=9)
        ax.set_title(f"Subject {subj_id} ({n_ep} epochs, {time_hrs[-1]:.1f}h)",
                     fontsize=10)
        ax.grid(alpha=0.15)

        if ax_i == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

        if ax_i == 2:
            ax.set_xlabel("Time (hours)", fontsize=10)

    fig.suptitle("Geometric Depth Score Across the Night", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(fig_dir / "geometric_depth_overnight.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved geometric_depth_overnight.png")

    return results, pca_score, formula_depth


# ===================================================================
# Analysis 4: Optimal number of geometric states
# ===================================================================
def analysis_4(geom_z: np.ndarray, stages: np.ndarray, fig_dir: Path) -> dict:
    """Sweep k=2..10, find optimal k via silhouette and Calinski-Harabasz."""
    print("\n" + "=" * 70)
    print("Analysis 4: Optimal Number of Geometric States")
    print("=" * 70)

    k_range = list(range(2, 11))
    sil_scores = []
    ch_scores = []
    ari_scores = []
    nmi_scores = []

    # Subsample for silhouette (too expensive on full dataset)
    rng = np.random.RandomState(42)
    sil_sample_size = min(20000, len(geom_z))
    sil_idx = rng.choice(len(geom_z), size=sil_sample_size, replace=False)
    geom_z_sil = geom_z[sil_idx]
    stages_sil = stages[sil_idx]

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels_full = km.fit_predict(geom_z)
        labels_sil = km.predict(geom_z_sil)

        sil = silhouette_score(geom_z_sil, labels_sil)
        ch = calinski_harabasz_score(geom_z, labels_full)
        ari = adjusted_rand_score(stages, labels_full)
        nmi = normalized_mutual_info_score(stages, labels_full)

        sil_scores.append(sil)
        ch_scores.append(ch)
        ari_scores.append(ari)
        nmi_scores.append(nmi)

        _, acc = map_clusters_to_stages(labels_full, stages, k)
        print(f"  k={k:>2d}  Sil={sil:.4f}  CH={ch:>10.0f}  "
              f"ARI={ari:.4f}  NMI={nmi:.4f}  Acc={acc:.4f}")

    best_k_sil = k_range[int(np.argmax(sil_scores))]
    best_k_ch = k_range[int(np.argmax(ch_scores))]

    print(f"\n  Best k (silhouette): {best_k_sil}")
    print(f"  Best k (Calinski-Harabasz): {best_k_ch}")

    # Physiological interpretation
    interp = {
        2: "Wake vs Sleep",
        3: "Wake / NREM / REM",
        4: "Wake / Light NREM / Deep NREM / REM",
        5: "Matches AASM: W / N1 / N2 / N3 / REM",
        6: "AASM + 1 sub-state (possible N2 split or REM subtypes)",
        7: "AASM + 2 sub-states",
        8: "Fine-grained geometric states",
    }

    results = {
        "k_range": k_range,
        "silhouette_scores": [round(s, 4) for s in sil_scores],
        "calinski_harabasz_scores": [round(c, 1) for c in ch_scores],
        "ARI_scores": [round(a, 4) for a in ari_scores],
        "NMI_scores": [round(n, 4) for n in nmi_scores],
        "best_k_silhouette": best_k_sil,
        "best_k_calinski_harabasz": best_k_ch,
        "interpretation_best_sil": interp.get(best_k_sil, f"k={best_k_sil} states"),
        "interpretation_best_ch": interp.get(best_k_ch, f"k={best_k_ch} states"),
    }

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(k_range, sil_scores, "o-", color="#5bffa8", linewidth=2, markersize=6)
    ax.axvline(best_k_sil, color="#5bffa8", alpha=0.4, linestyle="--")
    ax.set_xlabel("k (number of clusters)", fontsize=10)
    ax.set_ylabel("Silhouette Score", fontsize=10)
    ax.set_title(f"Silhouette (best k={best_k_sil})", fontsize=11)
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    ax.plot(k_range, ch_scores, "o-", color="#5b8bd4", linewidth=2, markersize=6)
    ax.axvline(best_k_ch, color="#5b8bd4", alpha=0.4, linestyle="--")
    ax.set_xlabel("k (number of clusters)", fontsize=10)
    ax.set_ylabel("Calinski-Harabasz Index", fontsize=10)
    ax.set_title(f"Calinski-Harabasz (best k={best_k_ch})", fontsize=11)
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    ax.plot(k_range, ari_scores, "s-", color="#f5c842", linewidth=2, markersize=6)
    ax.set_xlabel("k (number of clusters)", fontsize=10)
    ax.set_ylabel("Adjusted Rand Index", fontsize=10)
    ax.set_title("ARI vs AASM labels", fontsize=11)
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    ax.plot(k_range, nmi_scores, "D-", color="#6e3fa0", linewidth=2, markersize=6)
    ax.set_xlabel("k (number of clusters)", fontsize=10)
    ax.set_ylabel("Normalized MI", fontsize=10)
    ax.set_title("NMI vs AASM labels", fontsize=11)
    ax.grid(alpha=0.2)

    fig.suptitle("Optimal Number of Geometric States", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(fig_dir / "optimal_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved optimal_k.png")

    return results


# ===================================================================
# Cluster feature profiles plot
# ===================================================================
def plot_cluster_profiles(geom_z: np.ndarray, km5_labels: np.ndarray,
                          mapping: dict, fig_dir: Path):
    """Bar chart showing mean geometric feature profile of each cluster."""
    n_features = len(GEOM_NAMES)
    n_clusters = 5

    # Compute mean z-scored features per cluster
    cluster_means = np.zeros((n_clusters, n_features))
    for c in range(n_clusters):
        mask = km5_labels == c
        if np.any(mask):
            cluster_means[c] = geom_z[mask].mean(axis=0)

    # Reorder clusters by mapped stage for readability
    col_order = sorted(range(5), key=lambda c: mapping.get(c, c))

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(n_features)
    width = 0.15
    offsets = np.linspace(-2 * width, 2 * width, n_clusters)

    for i, c in enumerate(col_order):
        stage_label = STAGES[mapping[c]]
        bars = ax.bar(x + offsets[i], cluster_means[c], width,
                      label=f"C{c} ({stage_label})",
                      color=CLUSTER_COLORS[i], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(GEOM_NAMES, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean z-score", fontsize=10)
    ax.set_title("Geometric Feature Profiles by Cluster (KMeans k=5)", fontsize=13)
    ax.legend(fontsize=9, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.18))
    ax.grid(axis="y", alpha=0.2)
    ax.axhline(0, color="gray", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(fig_dir / "cluster_feature_profiles.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved cluster_feature_profiles.png")


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    fig_dir = PROJECT_ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Geometric Clustering: Unsupervised Sleep Structure Analysis")
    print("=" * 70)

    # --- Load ---
    spectral, geometric, stages, subjects = load_data()

    # --- Z-score normalize geometric features (float64 to avoid overflow) ---
    scaler = StandardScaler()
    geom_z = scaler.fit_transform(geometric.astype(np.float64))

    # --- Analysis 1: Clustering ---
    res1, km5_labels, mapping_km5, gmm5_probs = analysis_1(geom_z, stages)
    plot_confusion_matrix(km5_labels, mapping_km5, stages, fig_dir)

    # --- Analysis 2: Disagreement ---
    res2 = analysis_2(geom_z, spectral, geometric, km5_labels, mapping_km5, stages)

    # --- Analysis 3: Depth score ---
    res3, pca_score, formula_depth = analysis_3(
        geometric, geom_z, spectral, stages, subjects, fig_dir
    )

    # --- Analysis 4: Optimal k ---
    res4 = analysis_4(geom_z, stages, fig_dir)

    # --- Cluster profiles ---
    plot_cluster_profiles(geom_z, km5_labels, mapping_km5, fig_dir)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  KMeans k=5:  ARI={res1['kmeans_k5']['ARI']:.4f}  "
          f"NMI={res1['kmeans_k5']['NMI']:.4f}  "
          f"Acc={res1['kmeans_k5']['mapping_accuracy']:.4f}")
    print(f"  GMM k=5:     ARI={res1['gmm_k5']['ARI']:.4f}  "
          f"NMI={res1['gmm_k5']['NMI']:.4f}  "
          f"Acc={res1['gmm_k5']['mapping_accuracy']:.4f}")
    print(f"  Optimal k (silhouette):        {res4['best_k_silhouette']}")
    print(f"  Optimal k (Calinski-Harabasz): {res4['best_k_calinski_harabasz']}")
    print(f"  PCA depth vs delta power:  r={res3['pca_depth']['corr_delta']:.4f}")
    print(f"  Formula depth vs delta:    r={res3['formula_depth']['corr_delta']:.4f}")
    print(f"  PCA depth vs AASM (NREM):  r={res3['pca_depth']['corr_nrem_stage']:.4f}")

    # --- Save JSON ---
    all_results = {
        "analysis_1_clustering": res1,
        "analysis_2_disagreement": res2,
        "analysis_3_depth": res3,
        "analysis_4_optimal_k": res4,
    }

    json_path = PROJECT_ROOT / "results" / "geometric_clustering.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results -> {json_path}")


if __name__ == "__main__":
    main()
