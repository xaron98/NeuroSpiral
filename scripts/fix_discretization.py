"""Quick fix: add adaptive discretization to tesseract.py and re-run HMC.

Three improvements:
1. Q_adaptive: sgn(x - median(x)) instead of sgn(x) — channel-invariant
2. Q_quantile: split at per-coordinate median of the embedding cloud
3. Filter subjects with <500 epochs (removes truncated PSGs)
"""

import numpy as np
import sys, os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def Q_adaptive(embedding: np.ndarray) -> np.ndarray:
    """Median-centered sign discretizer.
    
    Instead of Q(x) = sgn(x), computes Q(x) = sgn(x - median(x)).
    This centers each coordinate around its own median, making the
    discretization invariant to channel-specific DC offsets.
    
    Parameters
    ----------
    embedding : array of shape (n_epochs, 4)
    
    Returns
    -------
    vertex_indices : array of shape (n_epochs,) with values 0-15
    """
    from src.geometry.tesseract import VERTICES
    
    medians = np.median(embedding, axis=0)  # (4,)
    centered = embedding - medians  # (n_epochs, 4)
    signs = np.sign(centered)
    signs[signs == 0] = 1.0
    
    # Convert signs to vertex index
    indices = np.zeros(len(signs), dtype=int)
    for i, s in enumerate(signs):
        dists = np.linalg.norm(s - VERTICES, axis=1)
        indices[i] = np.argmin(dists)
    
    return indices


def Q_quantile(embedding: np.ndarray, n_bins=2) -> np.ndarray:
    """Quantile-based discretizer.
    
    Splits each coordinate at its 50th percentile (median).
    Equivalent to Q_adaptive but explicitly using quantiles.
    Also supports 3-bin (tertile) discretization for finer resolution.
    
    Parameters
    ----------
    embedding : array of shape (n_epochs, 4)
    n_bins : 2 for binary (16 vertices), 3 for ternary (81 vertices)
    
    Returns
    -------
    vertex_indices : array of shape (n_epochs,)
    """
    from src.geometry.tesseract import VERTICES
    
    n_epochs, n_dim = embedding.shape
    
    if n_bins == 2:
        # Binary: above/below median for each coordinate
        medians = np.median(embedding, axis=0)
        binary = (embedding > medians).astype(float) * 2 - 1  # {-1, +1}
        
        indices = np.zeros(n_epochs, dtype=int)
        for i in range(n_epochs):
            dists = np.linalg.norm(binary[i] - VERTICES, axis=1)
            indices[i] = np.argmin(dists)
        return indices
    else:
        raise ValueError("Only n_bins=2 supported for now")


if __name__ == "__main__":
    """Re-run HMC validation with adaptive discretization."""
    
    import mne
    from pathlib import Path
    from collections import Counter
    from scipy.stats import chi2_contingency
    from sklearn.metrics import cohen_kappa_score, f1_score, mutual_info_score
    
    from src.features.spectral import compute_band_powers, compute_hjorth
    from src.features.takens import time_delay_embedding
    from src.geometry.tesseract import VERTICES, Q_discretize
    from src.geometry.alignment import compute_fixed_tau
    
    HMC_LABELS = {
        "Sleep stage W": "W", "Sleep stage N1": "N1", "Sleep stage N2": "N2",
        "Sleep stage N3": "N3", "Sleep stage R": "REM", "Sleep stage ?": None,
        "0": "W", "1": "N1", "2": "N2", "3": "N3", "4": "REM", "5": None,
        "W": "W", "N1": "N1", "N2": "N2", "N3": "N3", "REM": "REM", "R": "REM",
    }
    
    TARGET_SFREQ = 100
    MIN_EPOCHS = 500  # Filter out truncated PSGs
    
    data_dir = Path("data/hmc")
    fixed_tau = compute_fixed_tau(TARGET_SFREQ, 2.0)
    
    print("=" * 65)
    print("  NeuroSpiral — HMC Adaptive Discretization Test")
    print("  Comparing: sgn(x) vs sgn(x-median) vs quantile")
    print("=" * 65)
    print(f"\n  Min epochs filter: {MIN_EPOCHS}")
    print(f"  Fixed τ = {fixed_tau}")
    
    # Collect all data
    all_vertices_original = []
    all_vertices_adaptive = []
    all_vertices_quantile = []
    all_stages = []
    all_embeddings = []
    all_delta = []
    all_omega1 = []
    subject_count = 0
    subject_kappas_orig = []
    subject_kappas_adapt = []
    
    # Find subjects
    subjects = []
    for f in sorted(data_dir.glob("SN*.edf")):
        if "_sleepscoring" not in f.name:
            sid = f.stem
            hyp = data_dir / f"{sid}_sleepscoring.edf"
            if hyp.exists():
                subjects.append((f, hyp, sid))
    
    print(f"  Found {len(subjects)} subjects\n")
    
    for psg_path, hyp_path, sid in subjects:
        try:
            # Load EEG
            raw = mne.io.read_raw_edf(str(psg_path), preload=False, verbose=False)
            ch_names = raw.ch_names
            
            # Pick channels
            target_chs = ["EEG C4-M1", "EEG C3-M2"]
            picks = [c for c in target_chs if c in ch_names]
            if not picks:
                picks = [c for c in ch_names if 'EEG' in c][:2]
            if not picks:
                continue
            
            raw.pick(picks)
            raw.load_data(verbose=False)
            
            # Resample
            if raw.info['sfreq'] != TARGET_SFREQ:
                raw.resample(TARGET_SFREQ, verbose=False)
            
            raw.filter(0.5, 30.0, verbose=False)
            
            # Load annotations
            annotations = mne.read_annotations(str(hyp_path))
            new_descriptions = [str(d) for d in annotations.description]
            annotations = mne.Annotations(
                onset=annotations.onset,
                duration=annotations.duration,
                description=new_descriptions
            )
            raw.set_annotations(annotations)
            
            # Extract events
            events, event_id = mne.events_from_annotations(
                raw, chunk_duration=30.0, verbose=False
            )
            
            stage_names = ["W", "N1", "N2", "N3", "REM"]
            label_map = {}
            for desc, eid in event_id.items():
                mapped = HMC_LABELS.get(desc)
                if mapped and mapped in stage_names:
                    if mapped not in label_map:
                        label_map[mapped] = eid
            
            if not label_map:
                continue
            
            epochs = mne.Epochs(
                raw, events, event_id=label_map,
                tmin=0, tmax=30.0 - 1.0/TARGET_SFREQ,
                baseline=None, preload=True, verbose=False
            )
            
            if len(epochs) < MIN_EPOCHS:
                print(f"  {sid}... skip ({len(epochs)} epochs < {MIN_EPOCHS})")
                continue
            
            data = epochs.get_data()  # (n_epochs, n_channels, n_samples)
            
            # Get labels
            id_to_stage = {}
            for desc, eid in event_id.items():
                mapped = HMC_LABELS.get(desc)
                if mapped:
                    id_to_stage[eid] = mapped
            
            labels = []
            for i in range(len(epochs)):
                eid = epochs.events[i, 2]
                stage = id_to_stage.get(eid)
                if stage and stage in stage_names:
                    labels.append(stage)
                else:
                    labels.append(None)
            
            # Process epochs
            embeddings = []
            vertices_orig = []
            vertices_adapt = []
            deltas = []
            stages = []
            
            for i in range(len(data)):
                if labels[i] is None:
                    continue
                
                epoch = data[i, 0, :]  # First channel
                epoch_uv = epoch * 1e6
                
                if np.max(np.abs(epoch_uv)) > 500:
                    continue
                
                # Takens embedding
                emb = time_delay_embedding(epoch_uv, dimension=4, tau=fixed_tau)[0]
                if len(emb) == 0:
                    continue
                
                mean_emb = np.mean(emb, axis=0)
                embeddings.append(mean_emb)
                
                # Original Q(x) = sgn(x)
                v_orig = Q_discretize(mean_emb)
                vertices_orig.append(v_orig)
                
                # Band powers
                bp = compute_band_powers(epoch_uv, sfreq=TARGET_SFREQ)
                deltas.append(bp.get('delta', 0))
                
                stages.append(labels[i])
            
            if len(embeddings) < MIN_EPOCHS * 0.5:
                continue
            
            emb_array = np.array(embeddings)
            
            # Adaptive Q(x) = sgn(x - median)
            v_adapt = Q_adaptive(emb_array)
            
            # Quantile
            v_quant = Q_quantile(emb_array)
            
            # Per-subject kappa (original)
            stage_to_int = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
            y_true = [stage_to_int[s] for s in stages]
            
            # Simple vertex→stage mapping for kappa
            from collections import defaultdict
            vmap_orig = defaultdict(lambda: Counter())
            vmap_adapt = defaultdict(lambda: Counter())
            for v, s in zip(vertices_orig, stages):
                vmap_orig[v][s] += 1
            for v, s in zip(v_adapt, stages):
                vmap_adapt[v][s] += 1
            
            pred_orig = [stage_to_int[vmap_orig[v].most_common(1)[0][0]] for v in vertices_orig]
            pred_adapt = [stage_to_int[vmap_adapt[v].most_common(1)[0][0]] for v in v_adapt]
            
            k_orig = cohen_kappa_score(y_true, pred_orig)
            k_adapt = cohen_kappa_score(y_true, pred_adapt)
            
            subject_kappas_orig.append(k_orig)
            subject_kappas_adapt.append(k_adapt)
            
            all_vertices_original.extend(vertices_orig)
            all_vertices_adaptive.extend(v_adapt.tolist())
            all_vertices_quantile.extend(v_quant.tolist())
            all_stages.extend(stages)
            all_embeddings.extend(embeddings)
            all_delta.extend(deltas)
            
            subject_count += 1
            n3_count = stages.count("N3")
            print(f"  {sid}... ✓ {len(stages)} epochs, N3={n3_count}, "
                  f"κ_orig={k_orig:.3f}, κ_adapt={k_adapt:.3f}")
            
        except Exception as e:
            print(f"  {sid}... ✗ {e}")
            continue
    
    print(f"\n  Total: {len(all_stages)} epochs from {subject_count} subjects")
    print(f"  (filtered: ≥{MIN_EPOCHS} epochs per subject)")
    
    # === Cramér's V comparison ===
    print("\n" + "=" * 65)
    print("  COMPARISON: Original vs Adaptive vs Quantile Discretization")
    print("=" * 65)
    
    stage_to_int = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
    stage_ints = [stage_to_int[s] for s in all_stages]
    
    for name, verts in [("Original sgn(x)", all_vertices_original),
                         ("Adaptive sgn(x-median)", all_vertices_adaptive),
                         ("Quantile (median split)", all_vertices_quantile)]:
        # Cross-tab
        n_verts = 16
        n_stages = 5
        ct = np.zeros((n_verts, n_stages))
        for v, s in zip(verts, stage_ints):
            ct[v, s] += 1
        
        # Remove empty rows
        nonempty = ct.sum(axis=1) > 0
        ct_clean = ct[nonempty]
        
        if ct_clean.shape[0] < 2:
            print(f"\n  {name}: only {ct_clean.shape[0]} vertices occupied — skip")
            continue
        
        chi2, p, dof, _ = chi2_contingency(ct_clean)
        n_total = ct_clean.sum()
        k = min(ct_clean.shape)
        v_cramer = np.sqrt(chi2 / (n_total * (k - 1))) if n_total > 0 else 0
        
        # Vertex purity
        purities = []
        dominants = []
        for row in ct_clean:
            if row.sum() > 0:
                purities.append(row.max() / row.sum())
                dominants.append(["W","N1","N2","N3","REM"][int(np.argmax(row))])
        
        mean_purity = np.mean(purities) * 100
        unique_dominants = len(set(dominants))
        
        # Merged V
        from collections import defaultdict
        stage_groups = defaultdict(list)
        occupied_indices = np.where(nonempty)[0]
        for idx, dom in zip(occupied_indices, dominants):
            stage_groups[dom].append(idx)
        
        if len(stage_groups) > 1:
            merged_labels = []
            for v in verts:
                for dom, members in stage_groups.items():
                    if v in members:
                        merged_labels.append(dom)
                        break
                else:
                    merged_labels.append("unknown")
            
            # Compute merged V
            merged_ct = np.zeros((len(stage_groups), n_stages))
            dom_list = sorted(stage_groups.keys())
            for ml, si in zip(merged_labels, stage_ints):
                if ml in dom_list:
                    merged_ct[dom_list.index(ml), si] += 1
            
            if merged_ct.shape[0] >= 2:
                chi2_m, _, _, _ = chi2_contingency(merged_ct)
                n_m = merged_ct.sum()
                k_m = min(merged_ct.shape)
                v_merged = np.sqrt(chi2_m / (n_m * (k_m - 1)))
            else:
                v_merged = 0
        else:
            v_merged = 0
        
        print(f"\n  {name}:")
        print(f"    Vertices occupied:  {int(nonempty.sum())}/16")
        print(f"    Unique dominants:   {unique_dominants} stages")
        print(f"    Mean purity:        {mean_purity:.1f}%")
        print(f"    Cramér's V (raw):   {v_cramer:.3f}")
        print(f"    Cramér's V (merged):{v_merged:.3f}")
        print(f"    Dominant stages:    {dict(Counter(dominants))}")
    
    # === Kappa comparison ===
    print(f"\n  Per-subject κ comparison (n={subject_count}):")
    print(f"    Original:  {np.mean(subject_kappas_orig):.3f} ± {np.std(subject_kappas_orig):.3f}")
    print(f"    Adaptive:  {np.mean(subject_kappas_adapt):.3f} ± {np.std(subject_kappas_adapt):.3f}")
    
    # === CMI check ===
    print(f"\n  CMI check (quick, no permutation test):")
    
    # Discretize delta into 10 bins
    delta_arr = np.array(all_delta)
    delta_bins = np.digitize(delta_arr, np.percentile(delta_arr, np.arange(10, 100, 10)))
    
    for name, verts in [("Original", all_vertices_original),
                         ("Adaptive", all_vertices_adaptive)]:
        mi_delta = mutual_info_score(stage_ints, delta_bins)
        mi_vert = mutual_info_score(stage_ints, verts)
        
        joint = [f"{d}_{v}" for d, v in zip(delta_bins, verts)]
        mi_joint = mutual_info_score(stage_ints, joint)
        cmi = mi_joint - mi_delta
        
        print(f"    {name}: MI(vertex,stage)={mi_vert:.4f}, CMI(vertex|delta)={cmi:.4f}")
    
    print("\n  Done.")
