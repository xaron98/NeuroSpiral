Dear Editor,

We are pleased to submit our manuscript entitled "Clifford Torus Embedding of EEG Reveals Non-Redundant Geometric Features for Sleep Stage Classification: Cross-Dataset Validation" for consideration in Computers in Biology and Medicine.

This manuscript introduces a novel geometric framework for sleep EEG analysis based on Clifford torus embedding of Takens time-delay reconstructions. The framework extracts geometric features — winding numbers, inter-hemispheric phase difference variability, and phase coherence — that capture information about oscillatory coupling structure that traditional spectral analysis does not access.

We believe this work is suitable for Computers in Biology and Medicine for the following reasons.

First, the approach is genuinely novel. A systematic literature search confirmed that no prior work has applied Clifford torus projection or tesseract discretisation to biomedical signal classification. The mathematical foundation rests on a recent theorem by Gakhar and Perea (2024, Journal of Applied and Computational Topology) proving that quasiperiodic signals embed densely in tori — a result we operationalise for the first time in a biomedical context.

Second, the validation is rigorous. We evaluate the framework on two independent polysomnography datasets totalling 168 subjects and 147,876 epochs from different institutions (PhysioNet Sleep-EDF and Haaglanden Medisch Centrum), with different electrode montages, populations, and scoring standards. Conditional mutual information analysis with permutation testing (1000 permutations) demonstrates that geometric features contribute 75–102% additional information beyond delta power (p < 0.001 for all five features tested).

Third, we report both strengths and limitations transparently. While continuous torus features generalise robustly across datasets (CMI increases from 0.164 to 0.254), the discrete tesseract vertex mapping shows channel dependence (Cramér's V = 0.355 for frontal vs. 0.128 for central derivations). We discuss the geometric reasons for this divergence and propose solutions for future work.

Fourth, the strongest finding — that inter-hemispheric phase difference variability (a geometric feature) achieves higher mutual information with sleep stage (MI = 0.427) than any classical spectral feature including the delta/beta ratio (MI = 0.414) — suggests that torus embedding captures clinically relevant information about bilateral neural dynamics that spectral analysis alone cannot access.

The manuscript has not been published elsewhere and is not under consideration by any other journal. All data used in this study are publicly available through PhysioNet, and the analysis code is available at https://github.com/xaron98/neurospiral.

We suggest the following reviewers who have expertise in the relevant areas:

1. Prof. Jose A. Perea (Northeastern University) — expert in topological data analysis and sliding window embeddings of quasiperiodic functions
2. Prof. Anne C. Skeldon (University of Surrey) — expert in mathematical models of sleep-wake dynamics and circle maps
3. Dr. Mathieu Guillot — expert in topological data analysis applied to biosignals

Thank you for your consideration. We look forward to your response.

Sincerely,

Carlos Perea
Independent Researcher
Mataró, Barcelona, Spain
xaron98@gmail.com
