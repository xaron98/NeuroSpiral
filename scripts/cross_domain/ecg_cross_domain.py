#!/usr/bin/env python3
"""NeuroSpiral — Cross-Domain Test: Torus on Pathological ECG (PTB-XL).

Does the Clifford torus discriminate cardiac pathologies?
Same geometry, different domain.

Usage:
    python scripts/ecg_cross_domain.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import wfdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (cohen_kappa_score, f1_score, roc_auc_score,
                             classification_report)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TAUS = [10, 20, 40]  # Multi-scale for 500Hz ECG


def _wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def takens(sig, d=4, tau=20):
    n = len(sig) - (d - 1) * tau
    if n < 50:
        return None
    e = np.zeros((n, d))
    for i in range(d):
        e[:, i] = sig[i * tau: i * tau + n]
    if np.std(e) < 1e-15 or not np.all(np.isfinite(e)):
        return None
    return e


def torus8(emb):
    if emb is None or emb.shape[0] < 20:
        return None
    th = np.arctan2(emb[:, 1], emb[:, 0])
    ph = np.arctan2(emb[:, 3], emb[:, 2])
    dt = _wrap(np.diff(th))
    dp = _wrap(np.diff(ph))
    N = len(dt)
    if N < 5:
        return None
    f = [np.mean(np.abs(dt)),
         np.mean(np.abs(np.diff(dt))) if N >= 2 else 0,
         np.var(dt),
         np.sum(np.sqrt(dt**2 + dp**2))]
    c, _ = np.histogram(th, 16, (-np.pi, np.pi))
    c = c.astype(float)
    t = c.sum()
    if t > 0:
        p = c / t
        p = p[p > 0]
        f.append(-np.sum(p * np.log2(p)))
    else:
        f.append(0)
    pd = th - ph
    R = np.abs(np.mean(np.exp(1j * pd)))
    f.append(np.sqrt(-2 * np.log(max(R, 1e-10))) if R < 1 else 0)
    f.append(R)
    s = (emb >= 0).astype(int)
    v = s[:, 0] * 8 + s[:, 1] * 4 + s[:, 2] * 2 + s[:, 3]
    f.append(np.sum(np.diff(v) != 0) / max(len(v) - 1, 1))
    return np.array(f)


def parse_scp_codes(scp_str):
    """Parse SCP codes string without using eval/ast.literal_eval."""
    try:
        return json.loads(scp_str.replace("'", '"'))
    except Exception:
        return {}


def main():
    t_start = time.time()

    print()
    print("=" * 70)
    print("  NEUROSPIRAL — Cross-Domain: Torus on Pathological ECG")
    print("  PTB-XL: 21,837 ECGs, 5 classes, 10s @ 500Hz")
    print("=" * 70)

    data_dir = PROJECT_ROOT / "data" / "ptb-xl"
    data_dir.mkdir(parents=True, exist_ok=True)

    # ── Download PTB-XL metadata ──────────────────────────────
    print(f"\n[1/5] Loading PTB-XL from PhysioNet...")

    import pandas as pd
    import urllib.request

    meta_path = data_dir / "ptbxl_database.csv"
    if not meta_path.exists():
        print(f"  Downloading metadata...")
        url = "https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv"
        urllib.request.urlretrieve(url, meta_path)

    df = pd.read_csv(meta_path)
    print(f"  Total records: {len(df)}")

    # Parse SCP codes to get primary diagnostic code
    def get_primary(scp_str):
        codes = parse_scp_codes(scp_str)
        if codes:
            return max(codes, key=codes.get)
        return None

    df["primary_code"] = df["scp_codes"].apply(get_primary)

    # Download SCP statements mapping
    scp_path = data_dir / "scp_statements.csv"
    if not scp_path.exists():
        url = "https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv"
        urllib.request.urlretrieve(url, scp_path)

    scp_df = pd.read_csv(scp_path, index_col=0)

    def get_superclass(code):
        if code and code in scp_df.index:
            sc = scp_df.loc[code, "diagnostic_class"]
            if pd.notna(sc):
                return sc
        return None

    df["superclass"] = df["primary_code"].apply(get_superclass)
    df_valid = df.dropna(subset=["superclass"])
    print(f"  With diagnostic class: {len(df_valid)}")
    print(f"  Classes: {df_valid['superclass'].value_counts().to_dict()}")

    # Sample up to 400 per class for speed
    max_per_class = 400
    sampled = df_valid.groupby("superclass").apply(
        lambda x: x.sample(min(len(x), max_per_class), random_state=42)
    ).reset_index(drop=True)
    print(f"  Sampled: {len(sampled)} records")

    # ── Download and process ECG records ──────────────────────
    print(f"\n[2/5] Downloading ECGs and extracting torus features...")

    all_features = []
    all_labels = []
    n_ok = 0
    n_fail = 0

    for idx, row in sampled.iterrows():
        if n_ok % 200 == 0 and n_ok > 0:
            print(f"    {n_ok} records OK...", flush=True)

        filename = row["filename_hr"]  # 500Hz version
        label = row["superclass"]

        try:
            # wfdb needs the subdirectory in pn_dir, not in record name
            # filename = 'records500/XXXXX/YYYYY_hr'
            parts = filename.split("/")
            subdir = "/".join(parts[:-1])  # records500/XXXXX
            rec_name = parts[-1]           # YYYYY_hr
            record = wfdb.rdrecord(rec_name,
                                   pn_dir=f"ptb-xl/1.0.3/{subdir}",
                                   channels=[1])  # Lead II

            sig = record.p_signal[:, 0]

            if len(sig) < 4000 or not np.all(np.isfinite(sig)):
                n_fail += 1
                continue

            # Multi-scale torus features
            feats = []
            ok = True
            for tau in TAUS:
                emb = takens(sig, d=4, tau=tau)
                f = torus8(emb)
                if f is None:
                    ok = False
                    break
                feats.extend(f)

            if not ok or not np.all(np.isfinite(feats)):
                n_fail += 1
                continue

            all_features.append(feats)
            all_labels.append(label)
            n_ok += 1

        except Exception:
            n_fail += 1
            continue

    X = np.array(all_features, dtype=np.float64)
    y_str = np.array(all_labels)

    print(f"\n  Processed: {n_ok} OK, {n_fail} failed")
    print(f"  Feature matrix: {X.shape}")

    valid = np.all(np.isfinite(X), axis=1)
    X = X[valid]
    y_str = y_str[valid]

    le = LabelEncoder()
    y = le.fit_transform(y_str)
    classes = list(le.classes_)
    print(f"  Classes: {dict(zip(classes, np.bincount(y)))}")

    # ── Classification ────────────────────────────────────────
    print(f"\n[3/5] Classification...")

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = RandomForestClassifier(n_estimators=300, max_depth=15,
                                  class_weight="balanced", random_state=42, n_jobs=-1)

    # Binary: Normal vs Pathological
    y_bin = (y_str != "NORM").astype(int)
    y_pred_bin = cross_val_predict(clf, X_s, y_bin, cv=cv)
    y_prob_bin = cross_val_predict(clf, X_s, y_bin, cv=cv, method="predict_proba")[:, 1]
    kappa_bin = cohen_kappa_score(y_bin, y_pred_bin)
    auc_bin = roc_auc_score(y_bin, y_prob_bin)
    sens_bin = float((y_pred_bin[y_bin == 1] == 1).mean())
    spec_bin = float((y_pred_bin[y_bin == 0] == 0).mean())

    print(f"\n  Binary (Normal vs Pathological):")
    print(f"    kappa={kappa_bin:.3f}  AUC={auc_bin:.3f}  "
          f"sens={sens_bin:.3f}  spec={spec_bin:.3f}")

    # Multi-class
    y_pred_mc = cross_val_predict(clf, X_s, y, cv=cv)
    kappa_mc = cohen_kappa_score(y, y_pred_mc)
    f1_mc = f1_score(y, y_pred_mc, average="macro", zero_division=0)

    print(f"\n  Multi-class ({len(classes)} classes):")
    print(f"    kappa={kappa_mc:.3f}  F1={f1_mc:.3f}")
    print()
    print(classification_report(y, y_pred_mc, target_names=classes,
                                 digits=3, zero_division=0))

    # ── Omega1 gradient ───────────────────────────────────────
    print(f"[4/5] Omega1 gradient across pathologies...")

    class_omega1 = {}
    for cls in classes:
        mask = y_str == cls
        if mask.sum() > 0:
            class_omega1[cls] = float(X[mask, 0].mean())

    ordered = sorted(class_omega1.items(), key=lambda x: x[1])
    gradient = " < ".join(f"{k}({v:.4f})" for k, v in ordered)
    print(f"  omega1: {gradient}")
    ratio = ordered[-1][1] / max(ordered[0][1], 1e-10)
    print(f"  Ratio (max/min): {ratio:.2f}")

    # ── Feature importance ────────────────────────────────────
    print(f"\n[5/5] Feature importance...")

    clf.fit(X_s, y)
    imp = clf.feature_importances_

    feat_names = []
    base = ["omega1", "curvature", "accel", "geodesic",
            "entropy", "phase_std", "coherence", "transition"]
    for tau in TAUS:
        for fn in base:
            feat_names.append(f"{fn}_t{tau}")

    top10 = np.argsort(imp)[::-1][:10]
    print(f"\n  Top 10 features:")
    for rank, idx in enumerate(top10):
        print(f"    {rank+1}. {feat_names[idx]:<25} {imp[idx]:.4f}")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}\n")
    print(f"  Binary (Normal vs Path):  kappa={kappa_bin:.3f}  AUC={auc_bin:.3f}")
    print(f"  Multi-class ({len(classes)}):       kappa={kappa_mc:.3f}  F1={f1_mc:.3f}")
    print(f"  omega1 gradient: {' < '.join(k for k, _ in ordered)}")
    print(f"  omega1 ratio: {ratio:.2f}")

    print(f"\n{'=' * 70}")
    print(f"  VERDICT")
    print(f"{'=' * 70}\n")

    if auc_bin > 0.70:
        print(f"  [+++] Torus WORKS for ECG pathology (AUC={auc_bin:.3f})")
        print(f"        Cross-domain transfer confirmed.")
    elif auc_bin > 0.60:
        print(f"  [++] Torus shows MODERATE ECG discrimination (AUC={auc_bin:.3f})")
    elif auc_bin > 0.55:
        print(f"  [+] Torus shows WEAK ECG discrimination (AUC={auc_bin:.3f})")
    else:
        print(f"  [---] Torus does NOT discriminate ECG pathology (AUC={auc_bin:.3f})")

    if ratio > 1.5:
        print(f"  omega1 gradient exists (ratio={ratio:.2f})")
    else:
        print(f"  No clear omega1 gradient (ratio={ratio:.2f})")

    np.savez_compressed(
        PROJECT_ROOT / "results" / "ecg_cross_domain.npz",
        features=X.astype(np.float32),
        labels=y_str,
        classes=np.array(classes),
        kappa_binary=kappa_bin,
        auc_binary=auc_bin,
        importances=imp,
        feat_names=np.array(feat_names),
    )
    print(f"\n  Saved: results/ecg_cross_domain.npz")
    elapsed = time.time() - t_start
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed / 60:.1f}m)")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
