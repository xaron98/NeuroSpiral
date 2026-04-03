#!/usr/bin/env python3
"""NeuroSpiral — Cross-Domain Tests (Finance + Climate + Solar + Traffic).

Does the Clifford torus work outside biology?
Test on 4 fundamentally different domains.

Usage:
    python scripts/cross_domain_all.py
"""

from __future__ import annotations

import time
import warnings
from datetime import datetime

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def _wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def takens(sig, d=4, tau=3):
    n = len(sig) - (d - 1) * tau
    if n < 8:
        return None
    e = np.zeros((n, d))
    for i in range(d):
        e[:, i] = sig[i * tau: i * tau + n]
    if np.std(e) < 1e-15 or not np.all(np.isfinite(e)):
        return None
    return e


def torus8(emb):
    if emb is None or emb.shape[0] < 8:
        return None
    th = np.arctan2(emb[:, 1], emb[:, 0])
    ph = np.arctan2(emb[:, 3], emb[:, 2])
    dt = _wrap(np.diff(th))
    dp = _wrap(np.diff(ph))
    N = len(dt)
    if N < 3:
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


def run_domain(X, y, class_names, domain_name, taus_used):
    """Classify with torus features and PCA, report comparison."""
    valid = np.all(np.isfinite(X), axis=1)
    X, y = X[valid], y[valid]

    if len(X) < 20 or len(np.unique(y)) < 2:
        print(f"  SKIPPED: insufficient data ({len(X)} samples, {len(np.unique(y))} classes)")
        return None

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    n_splits = min(5, min(np.bincount(y)))
    n_splits = max(2, n_splits)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                  class_weight="balanced", random_state=42, n_jobs=-1)

    yp = cross_val_predict(clf, Xs, y, cv=cv)
    kappa_torus = cohen_kappa_score(y, yp)

    # PCA comparison
    pca = PCA(n_components=min(2, X.shape[1]), random_state=42)
    Xpca = pca.fit_transform(Xs)
    yp_pca = cross_val_predict(clf, Xpca, y, cv=cv)
    kappa_pca = cohen_kappa_score(y, yp_pca)

    # Omega1 gradient
    omega1_means = {}
    for c in np.unique(y):
        mask = y == c
        omega1_means[class_names[c]] = float(X[mask, 0].mean())
    ordered = sorted(omega1_means.items(), key=lambda x: x[1])
    gradient = " < ".join(f"{k}" for k, _ in ordered)

    # Beta (if 3+ classes: position of middle class on extreme axis)
    beta_str = "N/A"
    if len(np.unique(y)) >= 3:
        centroids = {c: Xs[y == c].mean(axis=0) for c in np.unique(y)}
        classes_sorted = [c for c, _ in sorted(
            [(c, omega1_means[class_names[c]]) for c in np.unique(y)],
            key=lambda x: x[1])]
        lo, hi = classes_sorted[0], classes_sorted[-1]
        mid = classes_sorted[len(classes_sorted) // 2]
        ax = centroids[hi] - centroids[lo]
        sq = np.dot(ax, ax)
        if sq > 1e-15:
            rv = centroids[mid] - centroids[lo]
            beta = float(np.dot(rv, ax) / sq)
            beta_str = f"{beta:.3f}"

    print(f"\n  {domain_name}:")
    print(f"    Samples: {len(y)}, Classes: {len(np.unique(y))}, Features: {X.shape[1]}")
    print(f"    kappa_torus = {kappa_torus:.3f}  |  kappa_PCA = {kappa_pca:.3f}")
    print(f"    omega1: {gradient}")
    print(f"    beta: {beta_str}")

    return {
        "domain": domain_name,
        "kappa_torus": kappa_torus,
        "kappa_pca": kappa_pca,
        "gradient": gradient,
        "beta": beta_str,
        "n_samples": len(y),
        "n_features": X.shape[1],
    }


def main():
    t_start = time.time()

    print()
    print("=" * 70)
    print("  NEUROSPIRAL — Cross-Domain Tests (4 domains)")
    print("  Does the Clifford torus work outside biology?")
    print("=" * 70)

    results = []

    # ── DOMAIN 1: FINANCE ─────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  DOMAIN 1: FINANCE (AAPL+MSFT+GOOGL+AMZN, 2020-2024)")
    print(f"{'=' * 70}")

    try:
        import yfinance as yf
        import pandas as pd

        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        start = "2020-01-01"
        end = "2024-12-31"

        # Download
        dfs = {}
        for t in tickers:
            dfs[t] = yf.download(t, start=start, end=end, progress=False)

        # Compute signals: close, volume, volatility(20d), RSI(14), MACD
        # Use AAPL as primary, others as additional signals
        df = dfs["AAPL"].copy()
        df["vol20"] = df["Close"].rolling(20).std()
        # RSI
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        df["RSI"] = 100 - 100 / (1 + gain / loss.clip(lower=1e-10))
        # MACD
        df["MACD"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
        df = df.dropna()

        SIGNALS = ["Close", "Volume", "vol20", "RSI", "MACD"]
        EPOCH = 20  # trading days
        TAUS = [2, 3, 5]

        all_feats, all_labels = [], []
        for i in range(len(df) // EPOCH):
            s = i * EPOCH
            e = s + EPOCH
            chunk = df.iloc[s:e]
            if len(chunk) < EPOCH:
                continue

            # Label: bull (>5%), bear (<-5%), lateral
            ret = (chunk["Close"].iloc[-1] - chunk["Close"].iloc[0]) / chunk["Close"].iloc[0]
            if ret > 0.05:
                label = 0  # bull
            elif ret < -0.05:
                label = 1  # bear
            else:
                label = 2  # lateral

            ep_f = []
            ok = True
            for sig in SIGNALS:
                vals = chunk[sig].values.astype(float)
                if np.any(~np.isfinite(vals)):
                    ok = False
                    break
                for tau in TAUS:
                    emb = takens(vals, 4, tau)
                    f = torus8(emb)
                    if f is None:
                        ok = False
                        break
                    ep_f.extend(f)
                if not ok:
                    break

            if ok:
                all_feats.append(ep_f)
                all_labels.append(label)

        if all_feats:
            X = np.array(all_feats)
            y = np.array(all_labels)
            r = run_domain(X, y, ["Bull", "Bear", "Lateral"], "Finance", TAUS)
            if r:
                results.append(r)
    except Exception as e:
        print(f"  Finance FAILED: {e}")

    # ── DOMAIN 2: CLIMATE (already done — load results) ───────
    print(f"\n{'=' * 70}")
    print(f"  DOMAIN 2: CLIMATE (Barcelona 2024, from previous analysis)")
    print(f"{'=' * 70}")

    try:
        cd = np.load("results/climate_cross_domain.npz")
        X_clim = cd["features"]
        y_clim = cd["labels"]
        r = run_domain(X_clim, y_clim,
                        ["Winter", "Spring", "Summer", "Autumn"],
                        "Climate", [1, 2, 3])
        if r:
            results.append(r)
    except Exception as e:
        print(f"  Climate FAILED: {e}")

    # ── DOMAIN 3: SOLAR ACTIVITY ──────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  DOMAIN 3: SOLAR ACTIVITY (sunspot number, 1970-2024)")
    print(f"{'=' * 70}")

    try:
        import urllib.request
        import csv
        import io

        # Download monthly sunspot data from SILSO
        url = "https://www.sidc.be/SILSO/INFO/sndtotcsv.php"
        resp = urllib.request.urlopen(url, timeout=30)
        text = resp.read().decode("utf-8")

        # Parse CSV: year, month, date_frac, ssn, std, n_obs, provisional
        rows = []
        for line in text.strip().split("\n"):
            parts = line.split(";")
            if len(parts) >= 4:
                try:
                    year = int(parts[0].strip())
                    month = int(parts[1].strip())
                    ssn = float(parts[3].strip())
                    if year >= 1970 and ssn >= 0:
                        rows.append((year, month, ssn))
                except ValueError:
                    continue

        # Build monthly time series
        ssn_vals = np.array([r[2] for r in rows])
        years = np.array([r[0] for r in rows])
        months = np.array([r[1] for r in rows])

        # Signals: SSN, 13-month smoothed, rate of change, cumulative
        ssn_smooth = np.convolve(ssn_vals, np.ones(13)/13, mode="same")
        ssn_rate = np.gradient(ssn_smooth)
        ssn_cumul = np.cumsum(ssn_vals - ssn_vals.mean())

        SIGNALS_SOL = [ssn_vals, ssn_smooth, ssn_rate, ssn_cumul]
        EPOCH_SOL = 27  # ~1 solar rotation in months (Carrington)
        # Actually use 12 months as epoch (1 year)
        EPOCH_SOL = 12
        TAUS_SOL = [1, 2, 3]

        # Labels: solar cycle phase
        # Cycle 21: 1976-1986, 22: 1986-1996, 23: 1996-2008, 24: 2008-2019, 25: 2019+
        # Phase: minimum (ssn<30), rising (ssn increasing), maximum (ssn>100), declining
        all_feats_sol, all_labels_sol = [], []
        for i in range(len(ssn_vals) // EPOCH_SOL):
            s = i * EPOCH_SOL
            e = s + EPOCH_SOL
            if e > len(ssn_vals):
                break

            mean_ssn = ssn_smooth[s:e].mean()
            mean_rate = ssn_rate[s:e].mean()

            if mean_ssn < 30:
                label = 0  # minimum
            elif mean_ssn >= 30 and mean_rate > 0.5:
                label = 1  # rising
            elif mean_ssn >= 100:
                label = 2  # maximum
            else:
                label = 3  # declining

            ep_f = []
            ok = True
            for sig in SIGNALS_SOL:
                vals = sig[s:e]
                if len(vals) < EPOCH_SOL or np.any(~np.isfinite(vals)):
                    ok = False
                    break
                for tau in TAUS_SOL:
                    emb = takens(vals, 4, tau)
                    f = torus8(emb)
                    if f is None:
                        ok = False
                        break
                    ep_f.extend(f)
                if not ok:
                    break

            if ok:
                all_feats_sol.append(ep_f)
                all_labels_sol.append(label)

        if all_feats_sol:
            X_sol = np.array(all_feats_sol)
            y_sol = np.array(all_labels_sol)
            r = run_domain(X_sol, y_sol,
                            ["Minimum", "Rising", "Maximum", "Declining"],
                            "Solar Activity", TAUS_SOL)
            if r:
                results.append(r)
    except Exception as e:
        print(f"  Solar FAILED: {e}")

    # ── DOMAIN 4: TRAFFIC (synthetic from real patterns) ──────
    print(f"\n{'=' * 70}")
    print(f"  DOMAIN 4: TRAFFIC (synthetic multi-day patterns)")
    print(f"{'=' * 70}")

    try:
        rng = np.random.default_rng(42)
        # Generate realistic traffic patterns: flow, speed, occupancy
        n_days = 365
        hours_per_day = 24

        all_feats_tr, all_labels_tr = [], []
        TAUS_TR = [1, 2, 3]

        for day in range(n_days):
            t = np.arange(hours_per_day)

            # Realistic traffic patterns
            is_weekend = (day % 7) >= 5
            base_flow = 500 if is_weekend else 1000
            # Morning rush 7-9, evening rush 16-19
            rush_morning = 800 * np.exp(-((t - 8)**2) / 4)
            rush_evening = 600 * np.exp(-((t - 17.5)**2) / 6)
            flow = base_flow + rush_morning + rush_evening + rng.normal(0, 50, hours_per_day)
            flow = np.maximum(flow, 10)

            speed = 65 - 0.02 * flow + rng.normal(0, 5, hours_per_day)
            occupancy = 0.001 * flow + rng.normal(0, 0.02, hours_per_day)
            occupancy = np.clip(occupancy, 0, 1)

            # Additional signals
            density = flow / np.maximum(speed, 1)
            variability = np.abs(np.diff(np.append(flow, flow[-1])))

            SIGS = [flow, speed, occupancy, density, variability]

            # Label per hour block
            for h_start in range(0, 24, 6):
                h_end = h_start + 6
                if h_start == 0:
                    label = 0  # night
                elif h_start == 6:
                    label = 1  # morning
                elif h_start == 12:
                    label = 2  # afternoon
                else:
                    label = 3  # evening

                ep_f = []
                ok = True
                for sig in SIGS:
                    vals = sig[h_start:h_end]
                    if len(vals) < 6:
                        ok = False
                        break
                    for tau in TAUS_TR:
                        emb = takens(vals, 4, tau)
                        f = torus8(emb)
                        if f is None:
                            ok = False
                            break
                        ep_f.extend(f)
                    if not ok:
                        break

                if ok:
                    all_feats_tr.append(ep_f)
                    all_labels_tr.append(label)

        if all_feats_tr:
            X_tr = np.array(all_feats_tr)
            y_tr = np.array(all_labels_tr)
            r = run_domain(X_tr, y_tr,
                            ["Night", "Morning", "Afternoon", "Evening"],
                            "Traffic", TAUS_TR)
            if r:
                results.append(r)
    except Exception as e:
        print(f"  Traffic FAILED: {e}")

    # ── SUMMARY TABLE ─────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY TABLE")
    print(f"{'=' * 70}\n")

    print(f"  {'Domain':<20} {'n':>6} {'feat':>5} {'k_torus':>8} {'k_PCA':>7} "
          f"{'torus/PCA':>10} {'beta':>7}")
    print(f"  {'-'*20} {'-'*6} {'-'*5} {'-'*8} {'-'*7} {'-'*10} {'-'*7}")

    for r in results:
        ratio = r["kappa_torus"] / max(r["kappa_pca"], 0.001)
        print(f"  {r['domain']:<20} {r['n_samples']:>6} {r['n_features']:>5} "
              f"{r['kappa_torus']:>8.3f} {r['kappa_pca']:>7.3f} "
              f"{ratio:>9.1f}x {r['beta']:>7}")

    # Include sleep for reference
    print(f"  {'Sleep (HMC)':<20} {'117K':>6} {'128':>5} {'0.607':>8} {'0.264':>7} "
          f"{'2.3x':>10} {'0.569':>7}")

    print(f"\n  Gradient omega1:")
    for r in results:
        print(f"    {r['domain']:<20}: {r['gradient']}")

    # Verdict
    print(f"\n{'=' * 70}")
    print(f"  VERDICT")
    print(f"{'=' * 70}\n")

    n_work = sum(1 for r in results if r["kappa_torus"] > 0.2)
    n_beat_pca = sum(1 for r in results if r["kappa_torus"] > r["kappa_pca"])

    print(f"  Domains tested: {len(results)}")
    print(f"  Torus kappa > 0.2: {n_work}/{len(results)}")
    print(f"  Torus beats PCA: {n_beat_pca}/{len(results)}")

    if n_work >= 3:
        print(f"\n  [+++] Torus works across MULTIPLE non-biological domains.")
        print(f"        It is a GENERAL-PURPOSE geometry for periodic signals.")
    elif n_work >= 1:
        print(f"\n  [++] Torus works in some non-biological domains.")
    else:
        print(f"\n  [---] Torus does not generalize outside biology.")

    np.savez_compressed(
        "results/cross_domain_all.npz",
        summary=np.array([(r["domain"], r["kappa_torus"], r["kappa_pca"],
                            r["gradient"], r["beta"]) for r in results], dtype=object),
    )
    print(f"\n  Saved: results/cross_domain_all.npz")
    elapsed = time.time() - t_start
    print(f"  Elapsed: {elapsed:.0f}s")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
