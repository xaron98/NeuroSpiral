#!/usr/bin/env python3
"""NeuroSpiral — Geometric Embedding Comparison.

Why Clifford torus and not another geometry?
Test 4 projections of the SAME 4D Takens vector on raw EEG signals,
extract 8 analogous features from each, classify, compare kappa.

Usage:
    python scripts/embedding_comparison.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import mne
from scipy.signal import argrelextrema
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGES = ["W", "N1", "N2", "N3", "REM"]
HMC_LABELS = {
    "Sleep stage W": "W", "Sleep stage N1": "N1", "Sleep stage N2": "N2",
    "Sleep stage N3": "N3", "Sleep stage R": "REM", "Sleep stage ?": None,
}
STAGE_INT = {s: i for i, s in enumerate(STAGES)}

TAUS = [10, 25, 40]
SFREQ = 100
EPOCH_SAMPLES = 3000


def _wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def takens_embed(signal, d=4, tau=25):
    n_emb = len(signal) - (d - 1) * tau
    if n_emb < 50:
        return None
    emb = np.zeros((n_emb, d))
    for i in range(d):
        emb[:, i] = signal[i * tau: i * tau + n_emb]
    if np.std(emb) < 1e-15 or not np.all(np.isfinite(emb)):
        return None
    return emb


# ─────────────────────────────────────────────────────────────
# Feature extractors for each geometry
# ─────────────────────────────────────────────────────────────
def features_torus(emb):
    """8 features from Clifford torus projection: theta=atan2(v2,v1), phi=atan2(v4,v3)."""
    theta = np.arctan2(emb[:, 1], emb[:, 0])
    phi = np.arctan2(emb[:, 3], emb[:, 2])
    dtheta = _wrap(np.diff(theta))
    dphi = _wrap(np.diff(phi))
    N = len(dtheta)
    if N < 5:
        return None
    f = []
    f.append(np.mean(np.abs(dtheta)))
    f.append(np.mean(np.abs(np.diff(dtheta))) if N >= 2 else 0.0)
    f.append(np.var(dtheta))
    f.append(np.sum(np.sqrt(dtheta**2 + dphi**2)))
    counts, _ = np.histogram(theta, bins=16, range=(-np.pi, np.pi))
    c = counts.astype(np.float64); total = c.sum()
    if total > 0:
        p = c / total; p = p[p > 0]
        f.append(-np.sum(p * np.log2(p)))
    else:
        f.append(0.0)
    pd = theta - phi
    R = np.abs(np.mean(np.exp(1j * pd)))
    f.append(np.sqrt(-2 * np.log(max(R, 1e-10))) if R < 1 else 0.0)
    f.append(R)
    signs = (emb >= 0).astype(int)
    verts = signs[:, 0]*8 + signs[:, 1]*4 + signs[:, 2]*2 + signs[:, 3]
    f.append(np.sum(np.diff(verts) != 0) / max(len(verts)-1, 1))
    return np.array(f, dtype=np.float64)


def features_sphere(emb):
    """8 features from S3 spherical projection."""
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    v = emb / norms
    r = norms.ravel()
    # Spherical angles in 4D
    theta1 = np.arccos(np.clip(v[:, 0], -1, 1))
    theta2 = np.arctan2(v[:, 2], v[:, 1])
    theta3 = np.arctan2(v[:, 3], np.sqrt(v[:, 1]**2 + v[:, 2]**2 + 1e-15))
    dt1 = np.diff(theta1)
    dt2 = _wrap(np.diff(theta2))
    N = len(dt1)
    if N < 5:
        return None
    f = []
    f.append(np.mean(np.abs(dt2)))                  # angular velocity
    f.append(np.mean(np.abs(np.diff(dt2))) if N >= 2 else 0.0)  # curvature
    f.append(np.var(dt2))                            # acceleration
    f.append(np.sum(np.sqrt(dt1**2 + dt2**2)))       # geodesic distance
    counts, _ = np.histogram(theta2, bins=16, range=(-np.pi, np.pi))
    c = counts.astype(np.float64); total = c.sum()
    if total > 0:
        p = c / total; p = p[p > 0]
        f.append(-np.sum(p * np.log2(p)))
    else:
        f.append(0.0)
    f.append(np.std(r))                              # radial variability
    f.append(np.mean(np.abs(dt1)))                   # polar velocity
    f.append(np.mean(np.abs(np.diff(theta3))) if N >= 2 else 0.0)  # azimuthal curvature
    return np.array(f, dtype=np.float64)


def features_cylinder(emb):
    """8 features from cylindrical projection: r=|(v1,v2)|, theta=atan2(v2,v1), z=|(v3,v4)|."""
    r = np.sqrt(emb[:, 0]**2 + emb[:, 1]**2)
    theta = np.arctan2(emb[:, 1], emb[:, 0])
    z = np.sqrt(emb[:, 2]**2 + emb[:, 3]**2)
    dtheta = _wrap(np.diff(theta))
    dz = np.diff(z)
    dr = np.diff(r)
    N = len(dtheta)
    if N < 5:
        return None
    f = []
    f.append(np.mean(np.abs(dtheta)))                # angular velocity
    f.append(np.mean(np.abs(np.diff(dtheta))) if N >= 2 else 0.0)  # curvature
    f.append(np.var(dtheta))                          # acceleration
    f.append(np.sum(np.sqrt(dtheta**2 + dz**2)))     # path length
    counts, _ = np.histogram(theta, bins=16, range=(-np.pi, np.pi))
    c = counts.astype(np.float64); total = c.sum()
    if total > 0:
        p = c / total; p = p[p > 0]
        f.append(-np.sum(p * np.log2(p)))
    else:
        f.append(0.0)
    f.append(np.std(r))                               # radial std
    f.append(np.mean(z))                              # mean height
    f.append(np.std(dz))                              # vertical variability
    return np.array(f, dtype=np.float64)


def features_pca(emb):
    """8 features from PCA of the 4D embedding (no geometric projection)."""
    from sklearn.decomposition import PCA as _PCA
    if len(emb) < 20:
        return None
    pca = _PCA(n_components=2, random_state=42)
    pc = pca.fit_transform(emb)
    pc1, pc2 = pc[:, 0], pc[:, 1]
    dpc1 = np.diff(pc1)
    dpc2 = np.diff(pc2)
    N = len(dpc1)
    if N < 5:
        return None
    f = []
    f.append(np.mean(np.abs(dpc1)))                   # PC1 velocity
    f.append(np.mean(np.abs(np.diff(dpc1))) if N >= 2 else 0.0)
    f.append(np.var(dpc1))
    f.append(np.sum(np.sqrt(dpc1**2 + dpc2**2)))     # path length
    f.append(float(pca.explained_variance_ratio_[0]))  # var explained
    f.append(np.std(pc1))
    f.append(np.std(pc2))
    f.append(np.corrcoef(pc1, pc2)[0, 1] if np.std(pc1) > 1e-10 and np.std(pc2) > 1e-10 else 0.0)
    return np.array(f, dtype=np.float64)


GEOMETRIES = {
    "Torus":    features_torus,
    "Sphere":   features_sphere,
    "Cylinder": features_cylinder,
    "PCA_4D":   features_pca,
}


def main():
    t_start = time.time()

    print()
    print("=" * 70)
    print("  NEUROSPIRAL — Geometric Embedding Comparison")
    print("  Same 4D Takens vector, 4 different projections")
    print("=" * 70)

    data_dir = PROJECT_ROOT / "data" / "hmc"
    results_dir = PROJECT_ROOT / "results"

    # Find subjects
    subj_files = []
    for i in range(1, 200):
        sid = f"SN{i:03d}"
        psg = data_dir / f"{sid}.edf"
        hyp = data_dir / f"{sid}_sleepscoring.edf"
        if psg.exists() and hyp.exists() and psg.stat().st_size > 1_000_000:
            subj_files.append((sid, psg, hyp))

    print(f"  Found {len(subj_files)} subjects")
    print(f"  Geometries: {list(GEOMETRIES.keys())}")
    print(f"  Taus: {TAUS}")
    print(f"  Features per geometry: 8 × {len(TAUS)} taus = {8 * len(TAUS)}")

    # Extract features for all geometries simultaneously
    all_features = {g: [] for g in GEOMETRIES}
    all_stages = []
    all_subjects = []
    n_ok = 0

    for idx, (sid, psg, hyp) in enumerate(subj_files):
        if idx % 20 == 0:
            print(f"  [{idx+1}/{len(subj_files)}] {sid}...", flush=True)

        raw = mne.io.read_raw_edf(str(psg), preload=True, verbose=False)
        ch = None
        for name in ["EEG C4-M1", "C4-M1", "EEG C4"]:
            if name in raw.ch_names:
                ch = name
                break
        if ch is None:
            continue

        raw.pick([ch])
        if raw.info["sfreq"] != SFREQ:
            raw.resample(SFREQ, verbose=False)
        raw.filter(0.5, 30.0, verbose=False)
        signal = raw.get_data()[0]

        annots = mne.read_annotations(str(hyp))
        n_ep = int(len(signal) / SFREQ // 30)
        labels = [None] * n_ep
        for onset, dur, desc in zip(annots.onset, annots.duration, annots.description):
            stage = HMC_LABELS.get(str(desc).strip())
            if stage is None:
                continue
            s = int(onset // 30)
            for e in range(s, min(s + max(1, int(dur // 30)), n_ep)):
                labels[e] = stage

        for i in range(n_ep):
            if labels[i] is None:
                continue
            start = i * EPOCH_SAMPLES
            end = start + EPOCH_SAMPLES
            if end > len(signal):
                break
            epoch = signal[start:end]
            if np.max(np.abs(epoch)) > 500e-6:
                continue

            # For each tau, embed and extract features for all geometries
            epoch_ok = True
            geom_feats = {g: [] for g in GEOMETRIES}

            for tau in TAUS:
                emb = takens_embed(epoch, d=4, tau=tau)
                if emb is None:
                    epoch_ok = False
                    break

                for g_name, g_func in GEOMETRIES.items():
                    f = g_func(emb)
                    if f is None:
                        epoch_ok = False
                        break
                    geom_feats[g_name].extend(f)

                if not epoch_ok:
                    break

            if not epoch_ok:
                continue

            for g_name in GEOMETRIES:
                all_features[g_name].append(geom_feats[g_name])
            all_stages.append(STAGE_INT[labels[i]])
            all_subjects.append(n_ok)

        n_ok += 1

    y = np.array(all_stages, dtype=np.int8)
    subjects = np.array(all_subjects, dtype=np.int16)

    print(f"\n  Epochs: {len(y):,}, Subjects: {n_ok}")
    for s_int, s_name in enumerate(STAGES):
        print(f"    {s_name}: {(y == s_int).sum():,}")

    # Classify each geometry
    print(f"\n{'=' * 70}")
    print(f"  CLASSIFICATION COMPARISON (StratifiedGroupKFold, 5-fold)")
    print(f"{'=' * 70}\n")

    cv = StratifiedGroupKFold(n_splits=5)
    clf = RandomForestClassifier(n_estimators=200, max_depth=15,
                                  class_weight="balanced", random_state=42, n_jobs=-1)

    results = {}
    for g_name in GEOMETRIES:
        X_g = np.array(all_features[g_name], dtype=np.float64)
        # Clean
        valid = np.all(np.isfinite(X_g), axis=1)
        X_g = X_g[valid]
        y_g = y[valid]
        subj_g = subjects[valid]

        X_s = StandardScaler().fit_transform(X_g)
        y_pred = cross_val_predict(clf, X_s, y_g, groups=subj_g, cv=cv)
        kappa = cohen_kappa_score(y_g, y_pred)
        f1 = f1_score(y_g, y_pred, average="macro", zero_division=0)

        results[g_name] = {"kappa": kappa, "f1": f1, "n_feat": X_g.shape[1],
                            "n_epochs": len(y_g)}

        print(f"  {g_name:<12} kappa={kappa:.3f}  F1={f1:.3f}  ({X_g.shape[1]} features)")

    # Omega1 gradient check for each geometry
    print(f"\n{'=' * 70}")
    print(f"  OMEGA1 GRADIENT (feature 0 = angular velocity)")
    print(f"{'=' * 70}\n")

    for g_name in GEOMETRIES:
        X_g = np.array(all_features[g_name], dtype=np.float64)
        valid = np.all(np.isfinite(X_g), axis=1)
        X_g = X_g[valid]
        y_g = y[valid]

        means = {}
        for s_int, s_name in enumerate(STAGES):
            mask = y_g == s_int
            if mask.sum() > 0:
                means[s_name] = float(X_g[mask, 0].mean())
        ordered = sorted(means.items(), key=lambda x: x[1])
        gradient = " < ".join(f"{k}" for k, _ in ordered)
        expected = "N3 < N2 < REM < N1 < W"
        match = gradient == expected
        marker = "MATCH" if match else ""
        print(f"  {g_name:<12}: {gradient}  {marker}")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}\n")

    best = max(results, key=lambda g: results[g]["kappa"])
    print(f"  {'Embedding':<12} {'kappa':>7} {'F1':>7} {'n_feat':>7}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*7}")
    for g in ["Torus", "Sphere", "Cylinder", "PCA_4D"]:
        r = results[g]
        marker = " <-- BEST" if g == best else ""
        print(f"  {g:<12} {r['kappa']:>7.3f} {r['f1']:>7.3f} {r['n_feat']:>7}{marker}")

    torus_k = results["Torus"]["kappa"]
    spread = max(r["kappa"] for r in results.values()) - min(r["kappa"] for r in results.values())

    print(f"\n{'=' * 70}")
    print(f"  VERDICT")
    print(f"{'=' * 70}\n")

    if best == "Torus":
        print(f"  [+++] Torus WINS (kappa={torus_k:.3f})")
        print(f"        Empirical justification: torus outperforms sphere, cylinder, and PCA.")
    elif results[best]["kappa"] - torus_k < 0.01:
        print(f"  [++] Torus ties with {best} (delta < 0.01)")
        print(f"       Torus preferred for interpretability (theta/phi decomposition).")
    else:
        delta = results[best]["kappa"] - torus_k
        print(f"  [!] {best} beats Torus by {delta:+.3f}")
        print(f"      This must be acknowledged in the paper.")
        print(f"      However, the torus provides unique interpretability")
        print(f"      (omega1 gradient, REM decomposition, per-channel paradox).")

    print(f"\n  Spread across geometries: {spread:.3f}")
    if spread < 0.03:
        print(f"  All geometries perform similarly — the Takens embedding")
        print(f"  matters more than the projection geometry.")

    # Save
    np.savez_compressed(
        results_dir / "embedding_comparison.npz",
        results_kappa=np.array([(g, results[g]["kappa"], results[g]["f1"])
                                 for g in GEOMETRIES], dtype=object),
        stages=y,
        subjects=subjects,
    )
    print(f"\n  Saved: results/embedding_comparison.npz")
    elapsed = time.time() - t_start
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
