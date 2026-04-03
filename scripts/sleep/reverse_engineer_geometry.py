#!/usr/bin/env python3
"""NeuroSpiral — Reverse Engineering the Geometry of Sleep Staging.

What geometry does a neural network discover when classifying sleep?
Does it match our Clifford torus, or is it fundamentally different?

Steps:
  1. Train MLP on 128 combined features (96 torus + 32 spectral)
  2. Extract 64D internal representations (last hidden layer)
  3. Persistent homology on learned representations (Z_47)
  4. Compare: omega1 gradient, decomposition, intrinsic dimension
  5. Apply torus features TO the CNN representations
  6. PH-as-features: add persistence features to classification
  7. Full comparative report

Usage:
    python scripts/reverse_engineer_geometry.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import ripser
import umap
from scipy.signal import argrelextrema
from scipy.stats import kruskal
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, f1_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGES = ["W", "N1", "N2", "N3", "REM"]
COEFF = 47

CHANNELS = {
    "EEG": slice(0, 24), "ECG": slice(24, 48),
    "EOG": slice(48, 72), "EMG": slice(72, 96),
}


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────
def _wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def torus_features_8(embedding):
    """8 canonical torus features from a 4D embedding."""
    if embedding is None or embedding.shape[0] < 20 or embedding.shape[1] < 4:
        return None
    theta = np.arctan2(embedding[:, 1], embedding[:, 0])
    phi = np.arctan2(embedding[:, 3], embedding[:, 2])
    dtheta = _wrap(np.diff(theta))
    dphi = _wrap(np.diff(phi))
    N = len(dtheta)
    if N < 5:
        return None
    feats = []
    feats.append(float(np.mean(np.abs(dtheta))))
    feats.append(float(np.mean(np.abs(np.diff(dtheta)))) if N >= 2 else 0.0)
    feats.append(float(np.var(dtheta)))
    feats.append(float(np.sum(np.sqrt(dtheta**2 + dphi**2))))
    counts, _ = np.histogram(theta, bins=16, range=(-np.pi, np.pi))
    c = counts.astype(np.float64)
    total = c.sum()
    if total > 0:
        p = c / total; p = p[p > 0]
        feats.append(float(-np.sum(p * np.log2(p))))
    else:
        feats.append(0.0)
    pd = theta - phi
    R_len = np.abs(np.mean(np.exp(1j * pd)))
    feats.append(float(np.sqrt(-2 * np.log(max(R_len, 1e-10)))) if R_len < 1 else 0.0)
    feats.append(float(R_len))
    signs = (embedding >= 0).astype(int)
    verts = signs[:, 0]*8 + signs[:, 1]*4 + signs[:, 2]*2 + signs[:, 3]
    feats.append(float(np.sum(np.diff(verts) != 0) / max(len(verts)-1, 1)))
    return np.array(feats, dtype=np.float64)


def compute_ph(X, maxdim=2, coeff=COEFF):
    """Compute PH with gap-based Betti estimation."""
    from scipy.spatial.distance import pdist
    dists = pdist(X[:min(300, len(X))])
    thresh = float(np.percentile(dists, 60)) * 1.5
    thresh = max(thresh, 0.5)
    result = ripser.ripser(X, maxdim=maxdim, thresh=thresh, coeff=coeff)
    diagrams = result["dgms"]
    betti = []
    lifetimes_all = []
    max_pers = []
    for dim in range(maxdim + 1):
        dgm = diagrams[dim]
        finite = dgm[np.isfinite(dgm[:, 1])]
        if len(finite) == 0:
            betti.append(0); lifetimes_all.append(np.array([])); max_pers.append(0.0)
            continue
        lt = finite[:, 1] - finite[:, 0]
        lt = lt[lt > 1e-10]
        lt = np.sort(lt)[::-1]
        if len(lt) == 0:
            betti.append(0); lifetimes_all.append(np.array([])); max_pers.append(0.0)
            continue
        # Gap method
        if len(lt) > 1:
            ratios = lt[:-1] / np.maximum(lt[1:], 1e-15)
            best_idx = int(np.argmax(ratios))
            n_pers = best_idx + 1 if ratios[best_idx] >= 2.0 else 0
        else:
            n_pers = 1
        betti.append(n_pers)
        lifetimes_all.append(lt)
        max_pers.append(float(lt[0]))
    return {"betti": betti, "lifetimes": lifetimes_all, "max_persistence": max_pers}


def intrinsic_dimension_mle(X, k=10):
    """MLE estimate of intrinsic dimension (Levina & Bickel 2004)."""
    from scipy.spatial import KDTree
    tree = KDTree(X)
    dists, _ = tree.query(X[:min(2000, len(X))], k=k+1)
    dists = dists[:, 1:]  # exclude self
    dists = dists[dists[:, -1] > 0]  # remove degenerate
    log_ratios = np.log(dists[:, -1:] / np.maximum(dists[:, :-1], 1e-15))
    return float(1.0 / np.mean(log_ratios))


def persistence_features_single(X_epoch, maxdim=1, n_sub=200, coeff=2):
    """Extract persistence-based features from a single epoch's features."""
    if len(X_epoch) < 4:
        return np.zeros(6)
    # Treat the epoch's 128 features as a 1D signal, do Takens embed
    signal = X_epoch.copy()
    # Reshape: treat 128D feature vector as a single point — not useful for PH per epoch
    # Instead: compute simple topological summary from the feature vector's structure
    # Use differences between consecutive features as a "signal"
    diffs = np.diff(signal)
    if len(diffs) < 10:
        return np.zeros(6)
    # Basic persistence-inspired features without full PH (too slow per epoch)
    sorted_d = np.sort(np.abs(diffs))[::-1]
    feats = [
        float(sorted_d[0]) if len(sorted_d) > 0 else 0,  # max gap
        float(np.sum(sorted_d[:5])) if len(sorted_d) >= 5 else 0,  # top-5 sum
        float(np.std(sorted_d)),  # gap variability
        float(-np.sum((sorted_d/max(sorted_d.sum(), 1e-10)) *
              np.log2(sorted_d/max(sorted_d.sum(), 1e-10) + 1e-15))),  # entropy
        float(np.sum(sorted_d > np.median(sorted_d) + 2*np.std(sorted_d))),  # n_outlier_gaps
        float(sorted_d[0] / max(sorted_d[1], 1e-10)) if len(sorted_d) > 1 else 0,  # gap ratio
    ]
    return np.array(feats, dtype=np.float64)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    results_dir = PROJECT_ROOT / "results"

    print()
    print("=" * 70)
    print("  NEUROSPIRAL — Reverse Engineering Sleep Geometry")
    print("  What geometry does a neural network discover?")
    print("=" * 70)

    # ── STEP 1-2: Load data and prepare splits ────────────────
    print(f"\n[1/8] Loading data...")
    d = np.load(results_dir / "combined_features.npz")
    X_torus = d["torus_individual"]   # (117510, 96)
    X_spectral = d["spectral"]        # (117510, 32)
    X_all = np.hstack([X_torus, X_spectral])  # (117510, 128)
    y = d["stages"]                   # (117510,)
    subjects = d["subjects"]          # (117510,)

    print(f"  Features: {X_all.shape} (96 torus + 32 spectral)")
    print(f"  Subjects: {len(np.unique(subjects))}")

    # Subject-wise split: 80/20
    unique_subj = np.unique(subjects)
    rng = np.random.default_rng(42)
    rng.shuffle(unique_subj)
    n_train_subj = int(0.8 * len(unique_subj))
    train_subj = set(unique_subj[:n_train_subj])

    train_mask = np.array([s in train_subj for s in subjects])
    test_mask = ~train_mask

    X_train, y_train = X_all[train_mask], y[train_mask]
    X_test, y_test = X_all[test_mask], y[test_mask]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_all_s = scaler.transform(X_all)

    print(f"  Train: {len(y_train):,} epochs ({n_train_subj} subjects)")
    print(f"  Test:  {len(y_test):,} epochs ({len(unique_subj) - n_train_subj} subjects)")

    # ── STEP 3: Train MLP ─────────────────────────────────────
    print(f"\n[2/8] Training MLP (128 -> 256 -> 128 -> 64 -> 5)...")
    t0 = time.time()

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=200,
        batch_size=512,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=False,
    )
    mlp.fit(X_train_s, y_train)

    y_pred_train = mlp.predict(X_train_s)
    y_pred_test = mlp.predict(X_test_s)
    kappa_train = cohen_kappa_score(y_train, y_pred_train)
    kappa_test = cohen_kappa_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test, average="macro", zero_division=0)

    print(f"  Train kappa: {kappa_train:.3f}")
    print(f"  Test kappa:  {kappa_test:.3f}")
    print(f"  Test F1:     {f1_test:.3f}")
    print(f"  Converged in {mlp.n_iter_} iterations ({time.time()-t0:.1f}s)")
    print()
    print(classification_report(y_test, y_pred_test,
          target_names=STAGES, digits=3, zero_division=0))

    # ── STEP 4: Extract 64D representations ───────────────────
    print(f"[3/8] Extracting 64D internal representations...")

    # MLP layers: input(128) -> W0(256) -> W1(128) -> W2(64) -> output(5)
    # We want the activations after the 3rd hidden layer (64D)
    def get_representations(X_scaled):
        """Forward through hidden layers, return last hidden activations."""
        h = X_scaled
        for i in range(len(mlp.coefs_) - 1):  # all except last (classifier) layer
            h = h @ mlp.coefs_[i] + mlp.intercepts_[i]
            h = np.maximum(h, 0)  # ReLU
        return h

    repr_all = get_representations(X_all_s)  # (117510, 64)
    print(f"  Representations: {repr_all.shape}")

    # Save
    np.savez_compressed(
        results_dir / "cnn_representations.npz",
        representations=repr_all.astype(np.float32),
        stages=y,
        subjects=subjects,
    )
    print(f"  Saved: results/cnn_representations.npz")

    # ── STEP 5: Persistent homology of representations ────────
    print(f"\n[4/8] Persistent homology of 64D representations (Z_{COEFF})...")

    repr_scaler = StandardScaler()
    repr_scaled = repr_scaler.fit_transform(repr_all)

    ph_results = {}
    for s_int, s_name in enumerate(STAGES):
        mask = y == s_int
        X_s = repr_scaled[mask]
        # Subsample
        idx = rng.choice(len(X_s), size=min(500, len(X_s)), replace=False)
        X_sub = X_s[idx]
        # PCA to 6D (like Gardner 2022)
        pca6 = PCA(n_components=6, random_state=42)
        X_pca = pca6.fit_transform(X_sub)
        ph = compute_ph(X_pca, maxdim=2)
        ph_results[s_name] = ph
        b = ph["betti"]
        mp1 = ph["max_persistence"][1]
        marker = " <-- beta_1=2!" if b[1] == 2 else ""
        print(f"  {s_name:<6} beta=[{b[0]}, {b[1]}, {b[2]}]  "
              f"H1 max_pers={mp1:.3f}{marker}")

    # Controls
    X_shuf = repr_scaled[y == 3].copy()  # N3
    idx_s = rng.choice(len(X_shuf), size=min(500, len(X_shuf)), replace=False)
    X_shuf = X_shuf[idx_s]
    for col in range(X_shuf.shape[1]):
        rng.shuffle(X_shuf[:, col])
    X_shuf_pca = PCA(n_components=6, random_state=42).fit_transform(X_shuf)
    ph_shuf = compute_ph(X_shuf_pca, maxdim=2)
    bs = ph_shuf["betti"]
    print(f"  SHUF   beta=[{bs[0]}, {bs[1]}, {bs[2]}]  "
          f"H1 max_pers={ph_shuf['max_persistence'][1]:.3f}")

    X_gauss = rng.standard_normal((500, 64))
    X_gauss_pca = PCA(n_components=6, random_state=42).fit_transform(X_gauss)
    ph_gauss = compute_ph(X_gauss_pca, maxdim=2)
    bg = ph_gauss["betti"]
    print(f"  GAUSS  beta=[{bg[0]}, {bg[1]}, {bg[2]}]  "
          f"H1 max_pers={ph_gauss['max_persistence'][1]:.3f}")

    # ── STEP 6: Compare geometries ────────────────────────────
    print(f"\n[5/8] Comparing geometries...")

    # 6a: UMAP 3D
    print(f"\n  6a. UMAP 3D projection...")
    idx_umap = rng.choice(len(repr_all), size=min(10000, len(repr_all)), replace=False)
    reducer = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.3, random_state=42)
    umap_3d = reducer.fit_transform(repr_scaled[idx_umap])
    y_umap = y[idx_umap]
    print(f"  UMAP shape: {umap_3d.shape}")

    # Per-stage centroids in UMAP space
    print(f"\n  Stage centroids in UMAP 3D:")
    umap_centroids = {}
    for s_int, s_name in enumerate(STAGES):
        mask = y_umap == s_int
        if mask.sum() > 0:
            c = umap_3d[mask].mean(axis=0)
            umap_centroids[s_name] = c
            print(f"    {s_name}: [{c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}]")

    # 6b: omega1 gradient in representation space
    print(f"\n  6b. Omega1 gradient in 64D representation space...")
    # Compute omega1 from first 4 dims of representation (analogous to torus)
    for s_int, s_name in enumerate(STAGES):
        mask = y == s_int
        r = repr_scaled[mask]
        omega1_vals = []
        for i in range(min(1000, len(r))):
            row = r[i, :4].reshape(1, -1)  # first 4D
            if np.std(row) > 1e-10:
                theta = np.arctan2(row[0, 1], row[0, 0])
                omega1_vals.append(abs(theta))
        if omega1_vals:
            print(f"    {s_name}: mean |theta| = {np.mean(omega1_vals):.4f}")

    # 6b-alt: Centroid distances in full 64D
    print(f"\n  Centroid distances in 64D representation space:")
    centroids_64 = {}
    for s_int, s_name in enumerate(STAGES):
        mask = y == s_int
        centroids_64[s_name] = repr_scaled[mask].mean(axis=0)

    print(f"  {'':>6}", end="")
    for s in STAGES:
        print(f"  {s:>6}", end="")
    print()
    for s1 in STAGES:
        print(f"  {s1:>6}", end="")
        for s2 in STAGES:
            d = np.linalg.norm(centroids_64[s1] - centroids_64[s2])
            print(f"  {d:>6.2f}", end="")
        print()

    # 6c: Decomposition (beta, gamma/d) in 64D
    print(f"\n  6c. Class decomposition in 64D space:")
    classes = sorted(centroids_64.keys())
    for target in classes:
        mu = centroids_64[target]
        others = [c for c in classes if c != target]
        A = np.column_stack([centroids_64[c] for c in others])
        x, _, _, _ = np.linalg.lstsq(A, mu, rcond=None)
        proj = A @ x
        resid = mu - proj
        norm_mu = np.linalg.norm(mu)
        beta = float(np.linalg.norm(proj) / max(norm_mu, 1e-15))
        gamma_d = float(np.linalg.norm(resid) / max(norm_mu, 1e-15))
        top = sorted(zip(others, x), key=lambda p: abs(p[1]), reverse=True)[:2]
        contrib = " + ".join(f"{v:+.2f}*{k}" for k, v in top)
        print(f"    {target:<6} beta={beta:.3f}  gamma/d={gamma_d:.3f}  {contrib}")

    # 6d: Intrinsic dimension
    print(f"\n  6d. Intrinsic dimension of representations:")
    for s_int, s_name in enumerate(STAGES):
        mask = y == s_int
        r = repr_scaled[mask]
        idx_d = rng.choice(len(r), size=min(2000, len(r)), replace=False)
        dim_est = intrinsic_dimension_mle(r[idx_d], k=10)
        print(f"    {s_name}: dim_intrinsic = {dim_est:.1f}")

    dim_all = intrinsic_dimension_mle(
        repr_scaled[rng.choice(len(repr_scaled), 2000, replace=False)], k=10)
    print(f"    ALL:  dim_intrinsic = {dim_all:.1f}")

    # ── STEP 6e: Torus features ON CNN representations ────────
    print(f"\n[6/8] Applying torus features to CNN representations...")

    # For each epoch, the 64D representation is a single point.
    # To apply torus features, we need a TRAJECTORY.
    # Use temporal context: take ±2 epochs as a 5-point trajectory in 64D.
    # Then project first 4 dims to get torus angles.
    # This tests: does the CNN's trajectory on consecutive epochs look toroidal?

    # Sort by subject then epoch order (they should already be ordered)
    # Use sliding window of 5 epochs from same subject
    print(f"  Building 5-epoch trajectories per subject...")
    torus_on_cnn = []
    torus_on_cnn_labels = []
    window = 5

    for subj in np.unique(subjects):
        mask_s = subjects == subj
        idx_s = np.where(mask_s)[0]
        if len(idx_s) < window:
            continue
        r_subj = repr_scaled[idx_s]
        y_subj = y[idx_s]
        # Use PCA to 4D for torus projection
        if len(r_subj) < 10:
            continue
        pca4 = PCA(n_components=4, random_state=42)
        r4 = pca4.fit_transform(r_subj)

        for i in range(len(r4) - window + 1):
            trajectory = r4[i:i+window]
            f = torus_features_8(trajectory)
            if f is not None:
                torus_on_cnn.append(f)
                torus_on_cnn_labels.append(y_subj[i + window // 2])

    if torus_on_cnn:
        X_torus_cnn = np.array(torus_on_cnn)
        y_torus_cnn = np.array(torus_on_cnn_labels)
        print(f"  Torus-on-CNN features: {X_torus_cnn.shape}")

        # Classify
        clf_tc = RandomForestClassifier(n_estimators=200, max_depth=15,
                                         class_weight="balanced", random_state=42, n_jobs=-1)
        cv_tc = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred_tc = cross_val_predict(clf_tc, X_torus_cnn, y_torus_cnn, cv=cv_tc)
        kappa_torus_cnn = cohen_kappa_score(y_torus_cnn, y_pred_tc)
        print(f"  Torus-on-CNN kappa: {kappa_torus_cnn:.3f}")

        # omega1 gradient
        print(f"  omega1 gradient (torus on CNN trajectories):")
        for s_int, s_name in enumerate(STAGES):
            mask = y_torus_cnn == s_int
            if mask.sum() > 0:
                print(f"    {s_name}: omega1 = {X_torus_cnn[mask, 0].mean():.4f}")
    else:
        kappa_torus_cnn = 0.0
        print(f"  No valid trajectories extracted.")

    # ── STEP 7: PH-as-features ────────────────────────────────
    print(f"\n[7/8] PH-inspired features for classification...")

    # Extract persistence-inspired features per epoch from the 128D feature vector
    print(f"  Computing 6 persistence-inspired features per epoch...")
    t0 = time.time()
    ph_feats = np.zeros((len(X_all), 6))
    for i in range(len(X_all)):
        ph_feats[i] = persistence_features_single(X_all_s[i])
    print(f"  Done in {time.time()-t0:.1f}s")

    # Combine: 128 original + 6 PH-inspired = 134
    X_aug = np.hstack([X_all_s, ph_feats])
    print(f"  Augmented features: {X_aug.shape}")

    # Cross-validate with StratifiedGroupKFold
    clf_aug = RandomForestClassifier(n_estimators=200, max_depth=15,
                                      class_weight="balanced", random_state=42, n_jobs=-1)
    cv_aug = StratifiedGroupKFold(n_splits=5)
    y_pred_aug = cross_val_predict(clf_aug, X_aug, y, groups=subjects, cv=cv_aug)
    kappa_aug = cohen_kappa_score(y, y_pred_aug)
    f1_aug = f1_score(y, y_pred_aug, average="macro", zero_division=0)
    print(f"  128 + 6 PH features: kappa={kappa_aug:.3f}, F1={f1_aug:.3f}")

    # Baseline: just 128 features
    clf_base = RandomForestClassifier(n_estimators=200, max_depth=15,
                                       class_weight="balanced", random_state=42, n_jobs=-1)
    y_pred_base = cross_val_predict(clf_base, X_all_s, y, groups=subjects, cv=cv_aug)
    kappa_base = cohen_kappa_score(y, y_pred_base)
    f1_base = f1_score(y, y_pred_base, average="macro", zero_division=0)
    print(f"  128 features only:   kappa={kappa_base:.3f}, F1={f1_base:.3f}")
    print(f"  Delta kappa: {kappa_aug - kappa_base:+.3f}")

    # ── STEP 8: Final report ──────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  FINAL COMPARATIVE TABLE")
    print(f"{'=' * 70}\n")

    # Collect beta_1 for CNN representations
    cnn_b1 = [ph_results[s]["betti"][1] for s in STAGES]
    cnn_b1_max = max(cnn_b1)

    # Check gradient in original feature space
    omega1_means = {}
    for s_int, s_name in enumerate(STAGES):
        mask = y == s_int
        omega1_means[s_name] = float(X_torus[mask, 0].mean())  # omega1_eeg_t10
    ordered = sorted(omega1_means.items(), key=lambda x: x[1])
    gradient_original = " < ".join(f"{k}" for k, v in ordered)

    # CNN centroid gradient (by distance from N3)
    n3_cent = centroids_64["N3"]
    cnn_dists = {s: np.linalg.norm(centroids_64[s] - n3_cent) for s in STAGES}
    ordered_cnn = sorted(cnn_dists.items(), key=lambda x: x[1])
    gradient_cnn = " < ".join(f"{k}" for k, v in ordered_cnn)

    print(f"  {'Method':<30} {'Dim':>5} {'beta_1':>7} {'Gradient':>20} {'kappa':>7}")
    print(f"  {'-'*30} {'-'*5} {'-'*7} {'-'*20} {'-'*7}")
    print(f"  {'Torus features (ours)':<30} {'128':>5} {'N/A':>7} "
          f"{gradient_original:>20} {kappa_base:>7.3f}")
    print(f"  {'MLP representations':<30} {'64':>5} {cnn_b1_max:>7} "
          f"{gradient_cnn:>20} {kappa_test:>7.3f}")
    print(f"  {'Torus ON MLP trajectories':<30} {'8':>5} {'N/A':>7} "
          f"{'(see above)':>20} {kappa_torus_cnn:>7.3f}")
    print(f"  {'128 feat + PH-inspired':<30} {'134':>5} {'N/A':>7} "
          f"{gradient_original:>20} {kappa_aug:>7.3f}")

    # Intrinsic dimension summary
    print(f"\n  Intrinsic dimension: {dim_all:.1f} (learned) vs 96D (torus)")

    # PH summary
    print(f"\n  PH of CNN representations (64D -> PCA 6D, Z_{COEFF}):")
    for s in STAGES:
        b = ph_results[s]["betti"]
        print(f"    {s}: beta=[{b[0]}, {b[1]}, {b[2]}]")
    print(f"    SHUF: beta=[{bs[0]}, {bs[1]}, {bs[2]}]")
    print(f"    GAUSS: beta=[{bg[0]}, {bg[1]}, {bg[2]}]")

    # ── VERDICT ──────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  VERDICT")
    print(f"{'=' * 70}\n")

    any_b1_2_cnn = any(ph_results[s]["betti"][1] == 2 for s in STAGES)
    ctrl_b1_2 = ph_shuf["betti"][1] >= 2 or ph_gauss["betti"][1] >= 2

    if any_b1_2_cnn and not ctrl_b1_2:
        print("  [+++] The CNN's internal representation shows beta_1=2!")
        print("        The neural network INDEPENDENTLY discovered toroidal geometry.")
        print("        This validates the Clifford torus as a natural geometry for sleep.")
    elif any_b1_2_cnn and ctrl_b1_2:
        print("  [??] beta_1=2 in both CNN and controls — inconclusive.")
    else:
        max_b1_cnn = max(ph_results[s]["betti"][1] for s in STAGES)
        print(f"  The CNN's internal geometry has beta_1={max_b1_cnn}.")
        if max_b1_cnn > 0:
            print(f"  It discovers CIRCULAR structure (beta_1>0) but not a full torus (beta_1!=2).")
        else:
            print(f"  No persistent topological structure detected in learned representations.")
        print(f"\n  The CNN (kappa={kappa_test:.3f}) outperforms our torus features "
              f"(kappa={kappa_base:.3f})")
        print(f"  by {kappa_test - kappa_base:+.3f} — the gap comes from the MLP learning")
        print(f"  non-linear feature combinations, not from discovering a different topology.")

    delta_ph = kappa_aug - kappa_base
    if abs(delta_ph) < 0.005:
        print(f"\n  PH-inspired features add delta_kappa={delta_ph:+.3f} — negligible.")
        print(f"  The torus features already capture the geometric information.")
    elif delta_ph > 0.005:
        print(f"\n  PH-inspired features add delta_kappa={delta_ph:+.3f} — positive contribution.")
        print(f"  Persistence captures complementary geometric information.")
    else:
        print(f"\n  PH-inspired features add delta_kappa={delta_ph:+.3f} — slight hurt.")

    # Save report
    report = {
        "kappa_torus_128": float(kappa_base),
        "kappa_mlp_test": float(kappa_test),
        "kappa_torus_on_cnn": float(kappa_torus_cnn),
        "kappa_128_plus_ph": float(kappa_aug),
        "mlp_iterations": mlp.n_iter_,
        "intrinsic_dim": float(dim_all),
        "ph_cnn": {s: ph_results[s]["betti"] for s in STAGES},
        "ph_control_shuf": ph_shuf["betti"],
        "ph_control_gauss": ph_gauss["betti"],
        "gradient_original": gradient_original,
        "gradient_cnn": gradient_cnn,
    }
    with open(results_dir / "reverse_engineering_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: results/reverse_engineering_report.json")

    elapsed = time.time() - t_start
    print(f"\n  Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
