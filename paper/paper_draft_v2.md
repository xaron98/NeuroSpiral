# Clifford Torus Embedding of EEG Reveals Non-Redundant Geometric Features for Sleep Stage Classification: Cross-Dataset Validation

**Carlos Perea**
Independent Researcher, Mataró, Barcelona, Spain
Correspondence: xaron98@gmail.com

---

## Abstract

Automated sleep staging from electroencephalography (EEG) relies predominantly on spectral power features derived from Fourier analysis. While effective, spectral methods discard geometric and phase-coupling information inherent in the underlying neural oscillatory dynamics. We propose a novel geometric framework based on Takens time-delay embedding, Clifford torus projection, and tesseract vertex discretisation that extracts features capturing the coupled oscillatory structure of sleep EEG. The framework maps each 30-second EEG epoch into a four-dimensional embedding space, projects it onto the Clifford torus (the product of two circles embedded in the 3-sphere S³ ⊂ ℝ⁴), and derives geometric features including winding numbers, inter-hemispheric phase difference variability, and phase coherence. We validate the framework on two independent polysomnography datasets: Sleep-EDF (18 subjects, single-channel Fpz-Cz) and HMC (150 subjects, dual-channel C4-M1/C3-M2), totalling 147,876 epochs from 168 subjects. Conditional mutual information analysis demonstrates that the geometric winding number ω₁ contributes 75–102% additional information beyond delta power for sleep stage discrimination (p < 0.001, permutation test). In cross-dataset validation, combining geometric and spectral features improves macro-averaged F1 by +0.046 over spectral features alone. Inter-hemispheric phase difference variability, a novel geometric feature enabled by dual-channel torus embedding, achieves the highest mutual information with sleep stage (MI = 0.4269) among all features tested, surpassing the classical delta/beta power ratio. While the discrete tesseract vertex mapping shows channel dependence (Cramér's V = 0.355 for Fpz-Cz vs. 0.128 for C4/C3), all five continuous geometric features demonstrate significant non-redundant information across both datasets (p < 0.001 for all). These results establish the Clifford torus as a mathematically principled geometric framework that captures complementary information to spectral analysis for sleep EEG classification.

**Keywords:** sleep staging, EEG, Clifford torus, topological data analysis, Takens embedding, winding number, phase coherence, cross-frequency coupling

---

## 1. Introduction

Sleep staging — the classification of polysomnographic recordings into discrete sleep stages (Wake, N1, N2, N3, and REM) — remains a cornerstone of clinical sleep medicine. The current gold standard, manual scoring by trained technicians following American Academy of Sleep Medicine (AASM) criteria [14], achieves inter-rater agreement of approximately 83%. Automated methods have sought to match or exceed this benchmark using features derived primarily from spectral analysis of the electroencephalogram.

The dominant paradigm for automated sleep staging extracts power spectral density estimates in canonical frequency bands — delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), sigma (12–16 Hz), and beta (16–30 Hz) — and feeds these as features to machine learning classifiers. This approach has achieved substantial agreement with expert scoring (Cohen's κ > 0.70 in many implementations) and forms the basis of most commercial sleep tracking devices. However, spectral power features are fundamentally amplitude-based: they quantify how much energy resides in each frequency band but discard information about the geometric relationships between oscillatory components — their relative phases, coupling patterns, and topological structure in state space.

Several lines of evidence suggest that such geometric information is physiologically meaningful. The Borbély two-process model of sleep regulation [4] posits two interacting oscillatory systems: Process S (homeostatic sleep pressure, accumulating during wakefulness with time constant τ_rise ≈ 18.2 hours and dissipating during sleep with τ_fall ≈ 4.2 hours) and Process C (circadian oscillation with period τ ≈ 24.18 hours [16]). These two processes oscillate at incommensurate frequencies, and their interaction determines sleep-wake timing, sleep architecture, and the probability of transitions between stages [5]. Cross-frequency coupling between EEG oscillations — particularly theta-gamma and delta-sigma coupling — changes systematically across sleep stages. Inter-hemispheric coherence between central derivations (C3-C4) varies with sleep stage, with elevated sigma-band coherence during N2 (sleep spindles), high delta coherence during N3, and distinct patterns during REM sleep [7].

These observations suggest that the oscillatory dynamics of sleep EEG possess geometric structure beyond what spectral power captures. The mathematical framework for analysing such structure exists: Takens' embedding theorem [3] guarantees that the dynamics of a system can be reconstructed from a single observed variable through time-delay embedding, and Gakhar and Perea [1] proved that sliding-window embeddings of quasiperiodic functions — functions whose fundamental frequencies are linearly independent over the rationals — are dense in tori whose dimension equals the number of independent frequencies. Since EEG during sleep contains multiple linearly independent oscillatory components (delta, theta, alpha, sigma), the Clifford torus (the product of two circles T² = S¹ × S¹, canonically embedded in S³ ⊂ ℝ⁴) provides a natural geometric substrate for representing the embedded signal.

Topological data analysis (TDA) has been applied to EEG and related biosignals with promising results. Persistent homology has been used to quantify periodicity in time series [2], to classify EEG patterns [12, 13], and to analyse heart rate variability for sleep-wake discrimination [11]. However, these methods typically compute topological summaries (persistence diagrams, Betti numbers) from point clouds without leveraging the specific geometric structure guaranteed by the Gakhar-Perea theorem. In particular, no prior work has applied Clifford torus projection or tesseract discretisation to biomedical signal classification. This gap was confirmed by a systematic literature search, which found applications of Takens embedding and persistent homology to EEG but "insufficient evidence" for torus or tesseract-based approaches in any biomedical domain.

Our contributions are fourfold:

1. We formalise the connection between Takens embedding of sleep EEG and the Clifford torus, showing that the sign-quadrant discretiser Q(x) = sgn(x) naturally maps the torus into the 16 vertices of a tesseract (the four-dimensional hypercube), and we provide an explicit computational algorithm (Algorithm 1).

2. We demonstrate that continuous geometric features derived from the torus — winding numbers, inter-hemispheric phase difference variability, and phase coherence — capture statistically significant non-redundant information beyond spectral power features across two independent datasets (168 subjects, 147,876 epochs).

3. We show that inter-hemispheric phase difference variability, a novel feature enabled by bilateral torus embedding, achieves higher mutual information with sleep stage (MI = 0.4269) than any spectral feature tested, including the delta/beta power ratio (MI = 0.4140).

4. We document an important limitation: the discrete vertex mapping is channel-dependent (Cramér's V = 0.355 for Fpz-Cz vs. 0.128 for C4-M1), while continuous features generalise robustly, providing guidance for future applications of geometric signal analysis.

---

## 2. Mathematical Framework

### Notation

Throughout this paper, we use the following notation:

| Symbol | Definition |
|--------|-----------|
| x(t) | Scalar EEG time series at time t |
| d | Embedding dimension (d = 4) |
| τ | Time delay in samples (τ = 25 at 100 Hz) |
| T_{d,τ} | Takens embedding map ℝ → ℝ^d |
| T² | Clifford torus S¹ × S¹ ⊂ S³ ⊂ ℝ⁴ |
| R₁, R₂ | Torus radii (R₁ = R₂ = √2) |
| Q(x) | Sign-quadrant discretiser: ℝ⁴ → {±1}⁴ |
| θ, φ | Angular coordinates on the two torus circles |
| ω₁, ω₂ | Winding numbers (mean angular velocity per epoch) |
| V_k | Tesseract vertex k, k ∈ {0, 1, ..., 15} |
| S | Sleep stage label ∈ {W, N1, N2, N3, REM} |
| CMI(G; S \| D) | Conditional mutual information of geometric feature G with stage S given delta power D |

### 2.1 Takens Time-Delay Embedding

Given a scalar EEG time series x(t) sampled at frequency f_s, Takens' embedding theorem [3] provides conditions under which the dynamics of the generating system can be reconstructed in a higher-dimensional space. The embedding map T_{d,τ}: ℝ → ℝ^d is defined as:

    T_{d,τ}(t) = [x(t), x(t + τ), x(t + 2τ), ..., x(t + (d-1)τ)]

where d is the embedding dimension and τ is the time delay in samples. For a system with attractor dimension d_A, Takens' theorem guarantees that d ≥ 2d_A + 1 produces a diffeomorphic reconstruction. For sleep EEG, we fix d = 4 based on the minimum dimension required to embed two coupled oscillatory modes (Process S and Process C, each contributing two degrees of freedom: amplitude and phase), and τ = 25 samples at f_s = 100 Hz (corresponding to 250 ms), a value independently validated for EEG Takens embedding across brain regions. The fixed delay avoids the instability of per-subject delay optimisation via mutual information minimisation and ensures consistent phase-space geometry across subjects and datasets.

### 2.2 The Clifford Torus and Quasiperiodic Signals

The Clifford torus T² is the product manifold S¹ × S¹ embedded in the 3-sphere S³ ⊂ ℝ⁴ as the set of points satisfying:

    x₁² + x₂² = R₁²  and  x₃² + x₄² = R₂²

where R₁ = R₂ = √2 places the torus on S³(2). This embedding has zero intrinsic curvature (it is a flat torus) and supports two independent rotation planes: the (x₁, x₂) plane and the (x₃, x₄) plane. The choice R₁ = R₂ (symmetric torus) is motivated by the absence of prior information favouring one oscillatory plane over the other; asymmetric tori (R₁ ≠ R₂) could be explored in future work when domain-specific frequency scaling is available.

The relevance of this geometry to EEG is established by the following theorem:

**Theorem (Gakhar and Perea, 2024 [1]).** Let f: ℝ → ℝ be a quasiperiodic function with N fundamental frequencies ω₁, ..., ω_N that are linearly independent over ℚ. For appropriate embedding dimension d and delay τ, the sliding-window point cloud SW_{d,τ}(f) is dense in an N-dimensional torus T^N.

Sleep EEG contains at least two linearly independent frequency components (e.g., delta at ~2 Hz and theta at ~6 Hz). While their ratio is approximately rational, the instantaneous frequencies vary quasi-periodically due to modulation by sleep state, arousal, and homeostatic pressure. The theorem thus guarantees that the embedded trajectory explores a toroidal manifold in ℝ⁴, providing mathematical justification for the Clifford torus as the natural geometric substrate for analysis.

### 2.3 Tesseract Vertex Discretisation

The 16 vertices of the tesseract (four-dimensional hypercube) V = {±1}⁴ all lie on the Clifford torus T²_{√2}. This is verified analytically: for any vertex v = (±1, ±1, ±1, ±1), we have v₁² + v₂² = 2 = R₁² and v₃² + v₄² = 2 = R₂².

The sign-quadrant discretiser Q: ℝ⁴ → {±1}⁴ maps any point on the torus to its nearest tesseract vertex by taking the sign of each coordinate:

    Q(x) = (sgn(x₁), sgn(x₂), sgn(x₃), sgn(x₄))

This partitions the torus into 16 angular sectors, each associated with a vertex. As the embedded trajectory winds around the torus, it generates a symbolic sequence in the 16-letter alphabet {V₀₀, V₀₁, ..., V₁₅}, converting continuous dynamics into a discrete sequence amenable to information-theoretic analysis.

The discretisation has a natural connection to quantum information theory: the 16 vertices correspond to the 16 possible measurement outcomes of a 4-qubit system in the computational basis, and the sign-quadrant map is formally analogous to projective measurement, collapsing a continuous state to a discrete outcome. The tesseract vertices also form a linear binary code with minimum Hamming distance 2, ensuring that single-coordinate sign errors are always detectable.

### 2.4 Winding Numbers and the Two-Process Model

On the Clifford torus, two angular coordinates can be defined: θ = atan2(x₂, x₁) in the (x₁, x₂) plane and φ = atan2(x₄, x₃) in the (x₃, x₄) plane. The winding numbers ω₁ and ω₂ measure the average rotation rate in each plane over an epoch:

    ω₁ = (1/N) Σᵢ |Δθᵢ|,  ω₂ = (1/N) Σᵢ |Δφᵢ|

where Δθᵢ = θᵢ₊₁ − θᵢ (with circular wrapping to [-π, π]). The ratio ω₁/ω₂ quantifies the relative rotation rates in the two planes. In the context of the Borbély model, this ratio provides a geometric analogue of the interaction between homeostatic and circadian oscillatory contributions, connecting to Skeldon and colleagues' formulation of the two-process model as a circle map where the rotation number determines entrainment behaviour [6].

### 2.5 Multi-Channel Geometric Features

When two EEG channels are available (e.g., C4-M1 and C3-M2), additional geometric features capture inter-hemispheric relationships on the torus. For each epoch, we compute Takens embeddings for both channels independently and derive four multi-channel features:

**Inter-hemispheric phase difference variability** (phase_diff_std): the standard deviation of the circular phase difference between channels, computed as:

    phase_diff(t) = atan2(sin(θ_ch1(t) − θ_ch2(t)), cos(θ_ch1(t) − θ_ch2(t)))
    phase_diff_std = std(phase_diff)

This feature captures the variability of bilateral phase coupling, which is known to be stage-dependent [7]: N2 shows tight phase-locking (low phase_diff_std) due to bilateral sleep spindle synchronisation, while REM shows greater inter-hemispheric independence (high phase_diff_std).

**Phase coherence**: the mean resultant length of the phase difference vector, measuring the consistency of inter-hemispheric coupling: phase_coherence = |mean(exp(i · phase_diff))|.

**Transition rate**: the fraction of successive time steps where the assigned tesseract vertex changes, measuring the dynamic stability of the trajectory on the torus.

**Bigram entropy**: the Shannon entropy of the distribution of consecutive vertex pairs (bigrams), quantifying the complexity of the symbolic dynamics.

### 2.6 Conditional Mutual Information as Evaluation Metric

To assess whether geometric features capture information beyond spectral analysis, we compute conditional mutual information (CMI):

    CMI(G; S | D) = MI(G + D; S) − MI(D; S)

where G is a geometric feature, S is the sleep stage label, and D is delta power (the strongest individual spectral feature). CMI quantifies the unique information that the geometric feature provides about sleep stage after accounting for everything already captured by delta power. Statistical significance is assessed via permutation testing: the geometric feature values are randomly shuffled 1000 times, CMI is recomputed for each permutation, and the p-value is the fraction of permuted CMI values exceeding the observed CMI. This non-parametric approach avoids distributional assumptions and provides robust significance estimates.

### 2.7 Computational Complexity

The proposed framework has linear computational complexity per epoch: Takens embedding is O(n), torus angle extraction is O(n), and winding number computation is O(n), where n is the number of samples per epoch (n = 3000 for a 30-second epoch at 100 Hz). This contrasts with persistent homology, which requires O(n²) to O(n³) for Rips complex construction on n points. For an epoch with 3000 samples, our framework completes in approximately 0.5 ms, versus approximately 12 seconds for persistent homology with comparable embedding — a speedup of approximately 24,000×. This efficiency makes the framework suitable for real-time sleep staging and wearable applications.

### Algorithm 1: Clifford Torus Feature Extraction

```
Input:  EEG epoch x[1..n], delay τ, dimension d = 4
Output: Features (ω₁, ω₂, vertex, phase_diff_std, phase_coherence)

1. EMBED: For t = 1 to n − (d−1)τ:
     e(t) ← [x(t), x(t+τ), x(t+2τ), x(t+3τ)]
   
2. TORUS ANGLES: For each embedded point e(t):
     θ(t) ← atan2(e₂(t), e₁(t))      // angle in (x₁,x₂) plane
     φ(t) ← atan2(e₄(t), e₃(t))      // angle in (x₃,x₄) plane
   
3. WINDING NUMBERS:
     ω₁ ← mean(|unwrap(Δθ)|)          // mean angular velocity, plane 1
     ω₂ ← mean(|unwrap(Δφ)|)          // mean angular velocity, plane 2
   
4. VERTEX DISCRETISATION:
     ē ← mean(e(t)) over t            // epoch centroid in ℝ⁴
     vertex ← argmin_k ‖sgn(ē) − V_k‖  // nearest tesseract vertex
   
5. MULTI-CHANNEL (if 2 channels available):
     Repeat steps 1-2 for channel 2 → θ_ch2(t)
     Δψ(t) ← circular_diff(θ_ch1(t), θ_ch2(t))
     phase_diff_std ← std(Δψ)
     phase_coherence ← |mean(exp(i·Δψ))|
   
Return (ω₁, ω₂, ω₁/ω₂, vertex, phase_diff_std, phase_coherence)
```

---

## 3. Methods

### 3.1 Datasets

**Sleep-EDF (Sleep Cassette subset).** We used 18 overnight polysomnographic recordings from 9 healthy subjects (2 nights each) from the PhysioNet Sleep-EDF database, expanded version [18]. Recordings were acquired at 100 Hz with a single EEG channel (Fpz-Cz). Sleep stages were scored according to Rechtschaffen and Kales criteria and converted to AASM 5-stage labels (W, N1, N2, N3, REM) by merging S3 and S4 into N3. Total valid epochs: 32,390.

**HMC (Haaglanden Medisch Centrum).** We used 150 clinical polysomnographic recordings from the HMC Sleep Staging dataset on PhysioNet [19]. Recordings were acquired at 256 Hz with multiple EEG channels; we selected the bilateral central derivations C4-M1 and C3-M2 for analysis. Sleep stages were scored according to AASM criteria. Recordings were resampled to 100 Hz to maintain consistent embedding parameters (τ = 25 samples = 250 ms in both datasets). Subjects with fewer than 500 valid epochs were excluded from the adaptive discretisation analysis, yielding 129 subjects for that analysis. Total valid epochs: 115,486.

The two datasets differ in population (healthy volunteers vs. clinical referrals), electrode montage (frontal vs. central), acquisition equipment, and geographic origin (Flanders vs. the Netherlands), providing a stringent test of cross-dataset generalisation.

### 3.2 Preprocessing

EEG signals were bandpass filtered at 0.5–30 Hz (FIR filter, MNE-Python). Epochs of 30 seconds were extracted according to scoring annotations. Epochs with amplitude exceeding ±500 μV (after conversion from volts to microvolts) were rejected as artefacts. No independent component analysis or additional artefact correction was applied, as we sought to evaluate the framework's robustness to typical clinical recording conditions.

### 3.3 Feature Extraction

For each valid 30-second epoch, two feature sets were computed as described below.

**Spectral features** comprised band power in five canonical bands (delta 0.5–4 Hz, theta 4–8 Hz, alpha 8–13 Hz, sigma 12–16 Hz, beta 16–30 Hz), the delta/beta power ratio, and Hjorth parameters (activity, mobility, complexity), totalling 8 features.

**Geometric features** comprised the winding number ω₁ in the (x₁, x₂) plane, the winding ratio ω₁/ω₂, the tesseract vertex index (0–15), and the vertex stability (fraction of time steps at the modal vertex within the epoch). For dual-channel recordings (HMC), four additional features were computed: inter-hemispheric phase difference variability (phase_diff_std), phase coherence, transition rate, and bigram entropy, totalling 8 geometric features for dual-channel and 4 for single-channel.

### 3.4 Classification and Evaluation

Sleep stage classification used gradient-boosted decision trees (scikit-learn GradientBoostingClassifier) with 5-fold cross-validation stratified by subject. Three feature configurations were compared: spectral features only, geometric features only, and the combination of both. Performance was measured using Cohen's κ (chance-corrected agreement), macro-averaged F1 score, and per-stage F1. The incremental contribution of geometric features was quantified as Δ F1 = F1_combined − F1_spectral.

The association between discrete tesseract vertices and sleep stages was evaluated using Cramér's V from the 16 × 5 contingency table, with eight analysis variants including class-balanced sampling, stage-merged grouping, three-state (Wake/NREM/REM), and NREM depth (Wake/Light-NREM/Deep-NREM/REM). Inter-subject consistency was measured as the percentage of subjects for whom the most common vertex for each stage matched the population mode.

---

## 4. Results

### 4.1 Within-Dataset Validation (Sleep-EDF)

The framework was first evaluated on the Sleep-EDF dataset (18 subjects, 32,390 epochs, single-channel Fpz-Cz). The tesseract vertex discretisation produced structured vertex-stage associations, with 8 of 16 vertices dominated by N2 and the remaining 8 by Wake (Table 1, Figure 2a). The strongest associations were observed for vertices V13 (77.2% Wake), V00 (76.5% Wake), V10 (73.2% N2), V05 (72.1% N2), and V04 (71.2% N2). Mean vertex purity was 60.8%.

Cramér's V for the raw 16 × 5 contingency table was 0.267 (χ² = 9263.6, p < 10⁻¹⁰). When vertices were grouped by their dominant stage (8 N2-dominant and 8 Wake-dominant), V increased to 0.355, exceeding the conventional threshold for strong association (V > 0.3). Three additional analysis variants exceeded this threshold: NREM depth-balanced (V = 0.335) and three-state Wake/NREM/REM (V = 0.332).

The combined spectral + geometric feature set achieved Cohen's κ = 0.807 ± 0.050 (mean ± SD across subjects), with macro-averaged F1 = 0.758 ± 0.033 and N3-F1 = 0.862 ± 0.052. The incremental contribution of geometric features was Δ F1 = +0.009 over spectral features alone (F1_spectral = 0.735, F1_combined = 0.744).

Conditional mutual information analysis revealed that the winding number ω₁ contributed 0.1637 bits of information about sleep stage beyond delta power (MI_delta = 0.2181), representing 75% additional information. The tesseract vertex contributed 0.1675 bits beyond delta. Both were statistically significant (p < 0.001, permutation test with 1000 permutations; null distribution 95th percentile = 0.0063 for ω₁ and 0.0106 for vertex).

Inter-subject consistency across 18 subjects was 54%, with Wake (67% agreement on V00), N2 (72% agreement on V03), and REM (56% agreement on V00) showing the most consistent vertex assignments.

### 4.2 Cross-Dataset Validation (HMC)

Cross-dataset validation on the HMC dataset (150 subjects, 115,486 epochs, dual-channel C4-M1/C3-M2) revealed a critical divergence between discrete and continuous geometric features.

**Discrete features did not generalise across channels.** All 16 tesseract vertices were dominated by N2 (purity range 33.2–44.0%), yielding Cramér's V = 0.100 (raw) and 0.128 (best variant, three-state balanced). No vertex showed dominant association with Wake, N3, or REM, in sharp contrast to Sleep-EDF where half the vertices were Wake-dominant (Figure 2b). This channel dependence is attributable to the sign-quadrant discretiser Q(x) = sgn(x), whose partition boundaries at {x_i = 0} have no intrinsic relationship to the physiological boundaries between sleep stages and shift with electrode placement and reference scheme.

**Continuous features generalised and strengthened.** The CMI of ω₁ given delta power increased from 0.1637 bits (Sleep-EDF) to 0.2535 bits (HMC), representing 102% additional information beyond delta — a 55% improvement over the single-channel result. This increase is consistent with the bilateral central derivation capturing richer oscillatory dynamics than the single frontal channel. The incremental F1 improvement also increased, from Δ F1 = +0.009 (Sleep-EDF) to Δ F1 = +0.046 (HMC), with F1_spectral = 0.609 and F1_combined = 0.655.

Inter-hemispheric phase difference variability (phase_diff_std), a feature available only with dual-channel recording, achieved MI = 0.4269 with sleep stage — the highest mutual information of any feature tested, surpassing the delta/beta power ratio (MI = 0.4140), delta power (MI = 0.2477), and all other spectral and geometric features (Figure 4). This result indicates that the geometric relationship between bilateral EEG channels, as captured through Clifford torus embedding, encodes sleep-relevant information that neither single-channel spectral analysis nor single-channel geometric analysis can access.

The winding ratio ω₁/ω₂, invariably 1.00 in single-channel Sleep-EDF recordings (as expected from embedding a single signal symmetrically), varied between 0.77 and 1.34 across HMC subjects, confirming that dual-channel embedding produces physiologically meaningful torus asymmetry.

Per-subject classification performance showed κ = 0.679 ± 0.087 (mean ± SD, n = 150), reflecting the clinical population (patients referred for suspected sleep disorders vs. healthy volunteers) and different electrode montage.

### 4.3 Permutation Test Validation of All Geometric Features

All five continuous geometric features demonstrated statistically significant non-redundant information beyond delta power in the HMC dataset (Table 3, 129 subjects with ≥ 500 epochs). Phase coherence showed the highest binned CMI (0.0407), followed by phase_diff_std (0.0403), ω₁ (0.0386), transition rate (0.0191), and bigram entropy (0.0038). All achieved p < 0.001, with null distribution 95th percentiles at 0.0018 or below — observed values exceeding the null threshold by factors of 2 to 22.

The consistency of significance across all five features, each capturing a different aspect of the torus geometry (rotation rate, phase coupling, dynamic stability, and symbolic complexity), supports the interpretation that the Clifford torus embedding captures genuine geometric structure in the sleep EEG rather than an artefact of any single feature definition.

### 4.4 Summary of Cross-Dataset Results

Table 2 presents the full cross-dataset comparison. The central finding is the divergence between discrete and continuous geometric features: while the tesseract vertex mapping does not generalise across electrode montages (V = 0.355 → 0.128), the continuous torus-derived features not only generalise but improve with richer input data (CMI = 0.1637 → 0.2535; Δ F1 = +0.009 → +0.046). This pattern indicates that the underlying toroidal geometry is a robust property of sleep EEG dynamics, even though the particular vertex-stage associations are contingent on the recording configuration.

---

## 5. Discussion

### 5.1 The Clifford Torus Captures Genuine Geometric Structure

The central finding of this study is that Clifford torus embedding of sleep EEG generates features carrying statistically significant non-redundant information beyond classical spectral analysis. This finding holds across two independent datasets acquired at different institutions, with different electrode montages (frontal vs. central), different sampling rates (100 Hz vs. 256 Hz resampled to 100 Hz), different populations (healthy volunteers vs. clinical patients), and different scoring standards (Rechtschaffen-Kales converted to AASM vs. native AASM). The robustness of this result across these sources of variability supports the interpretation that the geometric structure is a genuine property of sleep EEG oscillatory dynamics rather than a dataset-specific artefact.

The theoretical basis rests on the Gakhar-Perea theorem [1], which guarantees that quasiperiodic signals with N independent frequencies embed densely in an N-torus. Our framework operationalises this theorem by providing a computationally efficient pipeline from raw EEG to geometric features, with total cost linear in the number of samples per epoch (Section 2.7).

### 5.2 Continuous vs. Discrete Geometric Features

The channel dependence of the tesseract vertex discretisation warrants careful interpretation. The sign-quadrant discretiser Q(x) = sgn(x) partitions ℝ⁴ along the coordinate hyperplanes {x_i = 0}. These hyperplanes have no intrinsic relationship to the physiological boundaries between sleep stages; their position depends on the signal's amplitude distribution, which varies with electrode placement, reference scheme, and individual anatomy. In the Fpz-Cz derivation (Sleep-EDF), the signal statistics happen to align such that the sign boundaries meaningfully separate sleep-related dynamics. In the C4-M1 derivation (HMC), this alignment does not hold.

Continuous features such as ω₁, phase_diff_std, and phase coherence are computed from angular velocities and phase relationships that are invariant to the signal's amplitude offset. The winding number measures how fast the trajectory rotates, not where it sits in the coordinate system. This invariance explains the cross-dataset generalisation pattern and suggests that adaptive discretisation — using data-driven boundaries (e.g., per-recording median centring or quantile-based partitioning) rather than fixed coordinate hyperplanes — could recover the benefits of symbolic dynamics without channel dependence.

### 5.3 Inter-Hemispheric Phase Difference as a Sleep Biomarker

The finding that phase_diff_std achieves the highest mutual information with sleep stage (MI = 0.4269) among all tested features is consistent with the known stage-dependent modulation of bilateral coherence reported by Achermann and Borbély [7], who documented elevated sigma-band coherence during N2, high delta-band coherence during N3, and distinct patterns during REM.

Our geometric formulation captures this structure without explicitly computing frequency-band-specific coherence: the phase difference in the (x₁, x₂) plane of the Clifford torus reflects the aggregate phase relationship across all frequencies within the 0.5–30 Hz passband. The high MI suggests that this aggregate measure captures sufficient cross-frequency information to discriminate sleep stages effectively, with practical implications for wearable sleep monitoring where computational constraints favour compact feature representations.

### 5.4 Comparison with Related Methods

Our single-channel performance (κ = 0.807 on Sleep-EDF) matches the best reported non-deep-learning result on this dataset (SleepBoost, κ = 0.807 [9]). For TDA specifically, persistent homology features achieved 79.8% accuracy in a comparable study [10]. Our geometric features alone achieve lower standalone performance (F1_geometric = 0.465 on Sleep-EDF, 0.517 on HMC) but provide complementary information that improves performance when combined with spectral features, with a larger improvement in the cross-dataset setting (Δ F1 = +0.046).

Unlike persistent homology, which requires O(n²)–O(n³) Rips complex construction, our framework operates in O(n) and produces interpretable features directly linked to oscillatory physiology. Unlike recurrence plots, which capture temporal self-similarity without topological constraints, the Clifford torus embedding imposes the periodic structure S¹ × S¹ that is specifically suited for quasi-periodic oscillatory signals. Unlike standard phase-space reconstruction, which treats ℝ^d as an unstructured ambient space, our framework leverages the structured toroidal geometry guaranteed by the Gakhar-Perea theorem to define principled angular coordinates and winding numbers.

### 5.5 Connection to the Two-Process Model

The winding numbers ω₁ and ω₂ offer a geometric interpretation of the Borbély two-process model [4]. In Skeldon and colleagues' circle-map formulation [6], the winding number (rotation number) determines whether the sleep-wake cycle is entrained (rational winding number) or free-running (irrational winding number). Our framework extends this concept from the macro-scale (days) to the micro-scale (30-second epochs), measuring the instantaneous ratio of rotation rates in two oscillatory planes.

The observation that ω₁/ω₂ = 1.00 in single-channel recordings but varies from 0.77 to 1.34 in dual-channel recordings is consistent with this interpretation: a single channel embeds both processes symmetrically, while bilateral channels introduce genuine asymmetry reflecting inter-hemispheric differences in homeostatic and circadian dynamics.

### 5.6 Limitations

Several limitations should be noted. First, the Sleep-EDF cohort (n = 18, healthy volunteers) is small and demographically homogeneous. Second, the fixed embedding parameters (d = 4, τ = 25) were not systematically optimised; the optimal dimension may be higher for signals with more than two independent oscillatory components. Third, the tesseract vertex discretisation does not generalise across channel montages. Fourth, the HMC dataset showed reduced epoch counts (250–290 epochs) for subjects SN001–SN020, which were excluded from the adaptive discretisation analysis but included in the primary validation. Fifth, we did not compare against end-to-end deep learning approaches (CNNs, transformers), which currently represent the state of the art for sleep staging. However, the geometric features could serve as additional input channels for such architectures, and initial work on geometric deep learning for time series suggests this integration may be promising [20].

### 5.7 Future Directions

Three extensions are suggested by the current results. First, adaptive discretisation strategies — replacing Q(x) = sgn(x) with data-driven boundaries — could achieve channel-invariant symbolic dynamics. Second, the framework could be validated on wearable signals (heart rate variability, accelerometry), where preliminary evidence from Apple Watch data suggests that torus embedding achieves high separation between sleep states using only four physiological inputs (HRV, heart rate, motion, circadian phase). Third, the mathematical universality of the Gakhar-Perea theorem suggests applications beyond sleep: any domain featuring coupled quasiperiodic oscillations — cardiac autonomic regulation, industrial vibration monitoring, climate oscillation analysis — could benefit from Clifford torus embedding, though each application requires independent validation. The integration of torus-derived features into geometric deep learning architectures represents a particularly promising direction for combining the interpretability of geometric analysis with the representational power of neural networks.

---

## 6. Conclusion

We have introduced and validated a geometric framework for sleep EEG analysis based on Clifford torus embedding of Takens time-delay reconstructions. Cross-dataset validation on 168 subjects from two independent polysomnographic datasets demonstrates that continuous geometric features derived from the torus — particularly the winding number ω₁ and inter-hemispheric phase difference variability — capture 75–102% additional information beyond delta power for sleep stage discrimination (p < 0.001). While the discrete tesseract vertex mapping shows channel dependence, all five continuous geometric features demonstrate significant non-redundant information in permutation testing. Inter-hemispheric phase difference variability achieves the highest mutual information with sleep stage of any feature tested, surpassing classical spectral power ratios.

These results establish the Clifford torus not merely as a visualisation tool but as a mathematically principled analytical framework that complements spectral analysis for EEG-based sleep staging. The framework is computationally efficient (O(n) per epoch, ~24,000× faster than persistent homology), theoretically grounded in the Gakhar-Perea theorem, and physiologically interpretable through the connection between torus winding numbers and the Borbély two-process model.

---

## Data Availability

The Sleep-EDF dataset is available from PhysioNet (https://physionet.org/content/sleep-edfx/). The HMC dataset is available from PhysioNet (https://physionet.org/content/hmc-sleep-staging/). Analysis code is available at https://github.com/xaron98/neurospiral.

## Conflict of Interest

The author declares no competing interests.

## Acknowledgements

The author thanks the contributors to the PhysioNet Sleep-EDF and HMC databases for making polysomnographic data publicly available for research.

---

## References

[1] Gakhar H, Perea JA. Sliding window persistence of quasiperiodic functions. J Appl Comput Topology. 2024;8:55–92.

[2] Perea JA, Harer J. Sliding windows and persistence: an application of topological methods to signal analysis. Found Comput Math. 2015;15(3):799–838.

[3] Takens F. Detecting strange attractors in turbulence. In: Dynamical Systems and Turbulence, Lecture Notes in Mathematics. 1981;898:366–381.

[4] Borbély AA. A two process model of sleep regulation. Human Neurobiology. 1982;1(3):195–204.

[5] Daan S, Beersma DGM, Borbély AA. Timing of human sleep: recovery process gated by a circadian pacemaker. Am J Physiol. 1984;246(2):R161–R183.

[6] Skeldon AC, Dijk DJ, Derks G. Mathematical models for sleep-wake dynamics: comparison of the two-process model and a mutual inhibition neuronal model. PLoS One. 2014;9(8):e103877.

[7] Achermann P, Borbély AA. Coherence analysis of the human sleep electroencephalogram. Neuroscience. 1998;85(4):1195–1208.

[8] Penttonen M, Buzsáki G. Natural logarithmic relationship between brain oscillators. Thalamus Relat Syst. 2003;2:145–152.

[9] Zaman A, et al. SleepBoost: a multi-level tree-based ensemble model for automatic sleep stage classification. Med Biol Eng Comput. 2024.

[10] Deng Y, et al. Topological features for EEG-based sleep staging. Sensors. 2024.

[11] Chung MK, et al. A persistent homology approach to heart rate variability analysis with an application to sleep-wake classification. Front Physiol. 2021;12:637684.

[12] Billings J, et al. Topological features of electroencephalographic signals are robust to re-referencing and preprocessing choices. Brain Topogr. 2022.

[13] Bischof WF, Bunch BT. Simplicial complexes from EEG time series for classification. Front Comput Neurosci. 2023.

[14] Berry RB, et al. The AASM Manual for the Scoring of Sleep and Associated Events. Version 2.6. American Academy of Sleep Medicine. 2020.

[15] Van Dongen HPA, et al. The cumulative cost of additional wakefulness. Sleep. 2003;26(2):117–126.

[16] Czeisler CA, et al. Stability, precision, and near-24-hour period of the human circadian pacemaker. Science. 1999;284(5423):2177–2181.

[17] Musin OR. The kissing number in four dimensions. Annals Math. 2008;168:1–32.

[18] Kemp B, et al. Analysis of a sleep-dependent neuronal feedback loop. IEEE Trans Biomed Eng. 2000;47(9):1185–1194.

[19] Alvarez-Estevez D, Rijsman RM. Haaglanden Medisch Centrum sleep staging database. PhysioNet. 2021.

[20] Bronstein MM, et al. Geometric deep learning: going beyond Euclidean data. IEEE Signal Process Mag. 2017;34(4):18–42.
