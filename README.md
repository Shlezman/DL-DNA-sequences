# Identifying Regulatory DNA Sequences Using Machine Learning: A Comparative Study of 1D-CNN and SVM Approaches

**Authors:** [Author Names]
**Course:** Data Analysis and Machine Learning
**Date:** February 2026

---

## Abstract

Understanding the regulatory architecture of the genome is a central challenge in computational biology. Promoters, enhancers, and intergenic regions serve fundamentally distinct biological roles, yet their reliable identification from raw sequence alone remains non-trivial. This study addresses a three-class classification problem: given a DNA sequence of variable length, predict whether it belongs to a **promoter**, **enhancer**, or **intergenic** region. We assembled 234,783 labeled sequences (154,842 enhancers; 43,810 intergenic; 36,131 promoters) and addressed class imbalance via stratified downsampling to 36,131 samples per class, yielding 108,393 balanced sequences. Two modeling paradigms were evaluated: (1) a k-mer-based Support Vector Machine (SVM) with Optuna-tuned hyperparameters, and (2) a one-dimensional convolutional neural network (1D-CNN) with multi-layer regularization, data augmentation, and 5-fold stratified cross-validation. The SVM achieved 73.07% test accuracy, while the CNN achieved **91.70%** accuracy (macro F1 = 0.917), a +25.50% relative improvement. The CNN achieved near-perfect recall for promoter sequences (100%), consistent with the strong conservation of core promoter motifs. Gradient-based saliency analysis suggests the CNN learns biologically meaningful positional signals. These results demonstrate that hierarchical, position-sensitive convolutional representations substantially outperform position-agnostic k-mer statistics for regulatory element classification.

---

## 1. Introduction

The functional annotation of genomic sequences is one of the foundational tasks of modern computational genomics. The human genome encodes not only protein-coding genes (~2% of 3 billion base pairs) but an extensive repertoire of regulatory elements that orchestrate when, where, and how much each gene is expressed. Among the most critical are **promoters** — sequences located immediately upstream of transcription start sites (TSSs) that recruit the RNA polymerase II pre-initiation complex and serve as the mandatory platform for transcription initiation — and **enhancers** — distal cis-regulatory elements that contact promoters through chromatin looping to amplify transcription in a cell-type-specific, signal-responsive manner. The remaining majority of the genome is comprised of **intergenic** regions — sequences between annotated genes that largely lack direct transcriptional regulatory function, though they may harbor transposable elements, non-coding RNAs, or unannotated regulatory activity.

Accurate computational distinction among these three classes has profound implications: the interpretation of genome-wide association study (GWAS) variants, over 90% of which map to non-coding sequences, depends critically on knowing which elements are functionally active. Misclassification of an intergenic region as an enhancer — or vice versa — can lead to erroneous prioritization of disease variants. In synthetic biology, engineering gene circuits demands precise control over promoter and enhancer activity. And at a basic science level, understanding how the genome's regulatory grammar is encoded in primary sequence remains an open question.

**Core promoter architecture.** Mammalian core promoters are defined by a cluster of short sequence motifs within roughly 50 bp of the TSS. The TATA box (consensus TATAWAWR, ~30 bp upstream) recruits TATA-binding protein (TBP); the Initiator (Inr) element spans the TSS; the downstream promoter element (DPE) occurs ~30 bp downstream of the TSS. CpG islands — stretches of CG dinucleotide enrichment — mark roughly 70% of human promoters. These features are highly conserved, compact (8–20 bp), and positionally constrained relative to the TSS, making them in principle detectable by convolutional filters scanning fixed-length windows.

**Enhancer biology.** Enhancers are markedly more heterogeneous than promoters. They function as combinatorial platforms: clusters of transcription factor binding sites (TFBSs) of 6–12 bp each, interspersed with unbound linker sequence, collectively determine cell-type specificity and signal responsiveness. Unlike promoters, enhancers have no fixed positional relationship to their target genes (they may act from hundreds of kilobases away), and their primary sequence is less constrained across evolution. This biological heterogeneity is reflected in the lower classification F1 (0.88) observed for the enhancer class relative to promoters.

Traditional approaches rely on position weight matrices (PWMs) derived from known TFBS databases (Stormo, 2000) or on chromatin accessibility assays (ATAC-seq, DNase-seq) combined with histone modification profiling (ChIP-seq for H3K4me3, a promoter mark, or H3K27ac, an active enhancer mark). Sequence-only machine learning bypasses the need for such epigenomic experiments and can therefore make predictions for any genomic locus from primary sequence alone. K-mer frequency SVMs (Leslie et al., 2002) and, more recently, 1D-CNNs (Kelley et al., 2016; Zhou & Troyanskaya, 2015) have both been applied to this problem, with deep learning models consistently showing superior performance. This paper presents a rigorous head-to-head comparison on a balanced three-class dataset, with full methodological transparency and statistical validation.

**Why sequence-based classification matters.** Experimental characterization of regulatory elements via reporter assays, ChIP-seq, or CRISPR perturbation is expensive, time-consuming, and inherently cell-type-specific. A reliable sequence-based classifier could rapidly annotate millions of genomic loci — including those in newly sequenced genomes — without requiring any functional assay. Moreover, for clinical variant interpretation, knowing whether a single nucleotide polymorphism (SNP) falls in a promoter versus an intergenic region has direct implications for its likely pathogenic mechanism. The sequence-level models evaluated here represent a step toward scalable, experiment-free regulatory annotation.

---

## 2. Dataset and Preprocessing

### 2.1 Raw Data and Class Imbalance

The dataset *DNA_multiclass.parquet* contained 234,783 labeled sequences across three classes: Enhancer (154,842; 65.9%), Intergenic (43,810; 18.7%), and Promoter (36,131; 15.4%). This severe imbalance — enhancers over-represented 4.3× relative to the minority promoter class — reflects the biological reality that the human genome harbors an estimated 400,000–1,000,000 active enhancers (depending on the cell type surveyed) compared with roughly 20,000–30,000 protein-coding gene promoters. Uncorrected training on such a distribution would bias predictions toward the majority class, inflating macro accuracy while suppressing minority-class recall.

We applied stratified downsampling to n = 36,131 per class (the minority size), yielding 108,393 balanced sequences. The three sub-samples were concatenated and globally shuffled (random_state=42). A 70/15/15 stratified split produced:

| Partition | Samples |
|---|---|
| Train | 75,918 |
| Validation | 16,216 |
| Test (held-out) | 16,259 |

The test set was strictly held out and never used during model development or selection.

### 2.2 Sequence Length Distribution

Sequence lengths in the balanced dataset follow a mildly right-skewed distribution. Shapiro-Wilk tests on 5,000-sample subsets rejected normality for all three classes (p < 10⁻¹⁶ for Enhancer and Intergenic). A Kruskal-Wallis test confirmed that length distributions differ significantly across classes (H = 2332, p ≈ 0), with all pairwise Mann-Whitney U comparisons remaining significant after Bonferroni correction (adjusted α = 0.0167). However, the modest absolute differences in median lengths confirm that sequence length alone is insufficient for reliable classification — the primary discriminative signal must reside in nucleotide composition and motif arrangement.

**Table 1. Sequence Length Statistics (Balanced Dataset, n = 108,393)**

| Statistic | Value (bp) |
|---|---|
| Mean | 249.5 |
| Std. Dev. | 110.5 |
| Minimum | 2 |
| 25th Percentile | 194 |
| Median | 251 |
| 75th Percentile | 300 |
| 95th Percentile | 452 |
| Maximum | 573 |

For model input, all sequences were padded with a sentinel 'N' token to FINAL_LEN = 452 bp (the 95th-percentile length). Post-padding was applied; the ~5% of sequences exceeding 452 bp were truncated from the 3′ end. The choice of 452 bp preserves the entire biological sequence for 95% of samples while bounding computational cost. Global Average Pooling in the CNN architecture further mitigates padding artifacts by averaging activations across all positions, diluting the contribution of uninformative N-padded regions.

---

## 3. Methods

### 3.1 SVM with K-mer Features

The SVM pipeline decomposes each DNA sequence into a vector of k-mer frequencies — counts of all length-k sub-strings — creating a fixed-length representation amenable to linear classifiers. This is conceptually equivalent to a bag-of-words model in NLP: rich enough to capture local composition, but entirely blind to the order and spatial arrangement of those k-mers. For regulatory sequences, this is a fundamental limitation: a TATA box at position −30 bp and the same hexamer at position +200 bp carry completely different functional meanings, yet contribute identically to the k-mer vector.

Features were extracted using scikit-learn's *CountVectorizer* (analyzer='char', ngram_range=(k,k)). Hyperparameter optimization over k ∈ {3,4,5,6} and C ∈ [0.001,10] was performed with Optuna (15 trials, Tree-structured Parzen Estimator) on 15,000 training samples. Best configuration: k = 4, C ≈ 0.1. Final training used *LinearSVC* (dual=False, max_iter=2,000) on the full 75,918-sample training set.

### 3.2 1D-CNN Architecture

DNA sequences were integer-encoded (A→0, C→1, G→2, T→3, N→4) then one-hot encoded via tf.one_hot(depth=5), yielding tensors of shape (452, 5). One-hot encoding treats all nucleotides as equidistant — biologically appropriate since there is no natural ordinal relationship among bases — and allows convolutional filters to learn arbitrary position-specific base preferences, functionally equivalent to PWMs.

**Table 2. 1D-CNN Architecture**

| Layer | Filters | Kernel | Output Shape | Notes |
|---|---|---|---|---|
| Input | — | — | (452, 5) | One-hot encoded |
| GaussianNoise | — | — | (452, 5) | σ=0.1, train only |
| Conv1D Block 1 | 32 | 12 | (441, 32) | ReLU, L1+L2 |
| BatchNorm + SpatialDrop | — | — | (441, 32) | p=0.20 |
| MaxPool1D | — | 4 | (110, 32) | |
| Conv1D Block 2 | 64 | 8 | (103, 64) | ReLU, L1+L2 |
| BatchNorm + SpatialDrop | — | — | (103, 64) | p=0.30 |
| MaxPool1D | — | 4 | (25, 64) | |
| Conv1D Block 3 | 32 | 6 | (20, 32) | ReLU, L1+L2 |
| BatchNorm | — | — | (20, 32) | |
| GlobalAvgPool1D | — | — | (32,) | Positional averaging |
| Dense + BatchNorm + Dropout | 64 | — | (64,) | p=0.50 |
| Dense (output) | 3 | — | (3,) | Softmax |

*L1=1e-4, L2=1e-3 on all Conv and Dense kernels.*

The first convolutional layer uses a kernel of size 12 bp — matching the typical length of transcription factor binding motifs (8–20 bp) — so individual filters can learn to recognize specific sequence motifs in a position-sensitive manner, analogous to scanning a PWM along the sequence. Deeper layers with smaller kernels (8 and 6 bp) integrate pairwise and higher-order motif co-occurrences. GlobalAveragePooling over the final convolutional feature maps provides translational robustness while retaining sensitivity to the co-occurrence of patterns at any position.

### 3.3 Regularization Strategy

Five complementary regularization mechanisms combat overfitting: (1) **GaussianNoise (σ=0.1)** on the input layer perturbs one-hot vectors during training, acting as data augmentation; (2) **on-the-fly batch augmentation** (σ=0.02 Gaussian noise added per-batch via tf.data.Dataset) prevents exact sequence memorization; (3) **L1+L2 kernel regularization** on all convolutional and dense weights; (4) **SpatialDropout1D** (p=0.2–0.3) drops entire feature maps rather than individual units, which is more effective for correlated sequence data than standard dropout; and (5) **BatchNormalization** after each convolutional block stabilizes gradient flow. Training used Adam (lr=3×10⁻⁴, clipnorm=1.0) with categorical cross-entropy loss, batch size 128, for 10 epochs per fold.

### 3.4 Cross-Validation and Model Selection

5-fold stratified cross-validation was performed on the combined train+validation pool (n = 92,134). For each fold a fresh model was instantiated and trained; validation accuracy was recorded. The fold yielding the highest validation accuracy was retained as the final model for test evaluation. This framework prevents data leakage: the held-out test set (n = 16,259) was never touched during any stage of development or selection.

---

## 4. Results

### 4.1 SVM Baseline — 73.07% Accuracy

The k-mer SVM (k=4, C≈0.1) achieved 73.07% test accuracy. While above random chance (33.3%), the SVM's ceiling reflects the fundamental information loss of treating sequences as unordered bags of tetranucleotides. The model conflates identical k-mer compositions appearing in different positional arrangements, which is particularly damaging for regulatory sequence classification: a promoter's TATA box is only functional because it appears ~30 bp upstream of the TSS, not because the TATAAAA heptamer is abundant in the sequence. This positional blindness is the SVM's intrinsic limitation, irrespective of hyperparameter choice.

### 4.2 CNN Cross-Validation

**Table 3. 5-Fold Stratified Cross-Validation Results**

| Fold | Val Accuracy | Val AUC |
|---|---|---|
| 1 | 0.9131 | ~0.985 |
| 2 | 0.9105 | ~0.985 |
| 3 | 0.9104 | ~0.985 |
| 4 | 0.9078 | ~0.985 |
| **5 ★** | **0.9132** | **~0.986** |
| **Mean ± SD** | **0.9110 ± 0.0020** | — |

*★ = model selected for final evaluation.*

Cross-validation variance is remarkably low (SD = 0.002), confirming that the model's performance is stable across data splits and is not an artifact of a particularly favorable partition. The near-identical AUC values across folds (~0.985) indicate excellent discriminative ability across the full precision-recall trade-off curve for all three classes simultaneously.

### 4.3 Final Evaluation on Held-Out Test Set

**Table 4. CNN Classification Report (Test Set, n = 16,259)**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| ENHANCER | 0.879 | 0.881 | 0.880 | 5,419 |
| **PROMOTER** | **0.983** | **1.000** | **0.991** | **5,420** |
| INTERGENIC | 0.888 | 0.870 | 0.879 | 5,420 |
| Macro Avg | 0.916 | 0.917 | 0.917 | 16,259 |
| **Overall Acc.** | — | — | **91.70%** | 16,259 |

*Test loss: 0.2929.*

The CNN's most biologically striking result is its **perfect recall for PROMOTER sequences** (recall = 1.000; precision = 0.983). Every single promoter in the test set was correctly identified. This is interpretable in terms of the biology described above: core promoter motifs (TATA box, CpG island, Inr element) are highly conserved, compact, and appear at fixed positions relative to the TSS. Convolutional filters with kernel size 12 bp are precisely calibrated to detect such motifs, and their positional consistency within the sequence window provides an additional discriminative signal unavailable to the k-mer SVM.

The lower F1 scores for ENHANCER (0.880) and INTERGENIC (0.879) are also biologically interpretable. Enhancers lack conserved positional motif arrangements — their TFBSs are combinatorial and context-dependent — making them harder to distinguish from intergenic regions, which may harbor unannotated regulatory elements or diverged enhancers from other cell types.

**Table 5. Model Comparison (Held-Out Test Set)**

| Model | Accuracy | Macro F1 | McNemar p |
|---|---|---|---|
| SVM (k=4, LinearSVC) | 73.07% | ~0.730 | — |
| 1D-CNN (Fold 5) | **91.70%** | **0.917** | <0.001 |
| Relative Improvement | **+25.50%** | — | — |

### 4.4 Model Interpretability

Two post-hoc interpretability analyses were conducted. First, **first-layer filter weights** (32 filters × 12 bp × 5 channels) were visualized as position weight matrices. Several filters exhibited characteristic nucleotide preference patterns consistent with known regulatory motifs: GC-rich patterns reminiscent of CpG islands (a promoter hallmark), AT-rich patterns consistent with TATA boxes (consensus: TATAWAWR), and purine-rich stretches matching SP1 binding sites (GC boxes: GGGCGG). This qualitative correspondence suggests the network has learned biologically relevant features through gradient descent alone, without any prior knowledge of known TFBSs.

Second, **gradient-based saliency maps** were computed for individual test sequences using tf.GradientTape. The gradient of the predicted class score with respect to the one-hot input tensor identifies nucleotide positions whose perturbation would most strongly alter model confidence. For promoter sequences, saliency is concentrated at discrete, compact windows — consistent with the localized TATA box and Inr element. For enhancer sequences, saliency is distributed more broadly across multiple positions, consistent with the combinatorial TFBS architecture of enhancers. This interpretability analysis is exploratory; rigorous motif discovery would require JASPAR database alignment and statistical enrichment testing.

### 4.5 Statistical Validation of Model Differences

To confirm that the performance gap between the CNN and SVM is not attributable to sampling variability, we applied **McNemar's test** to the paired binary correctness vectors on the shared test set. McNemar's test is appropriate here because the two classifiers operate on the same test samples, making errors correlated rather than independent. The test statistic compares the number of samples correctly classified by the CNN but not the SVM (n₀₁) against those correctly classified by the SVM but not the CNN (n₁₀). The highly significant result (p < 0.001, with n₀₁ >> n₁₀) confirms that the CNN's improved accuracy reflects a genuine difference in learned representations, not statistical noise.

The Bonferroni-corrected pairwise Mann-Whitney U tests on sequence length (Section 2.2) serve a complementary role: by confirming that length is statistically discriminative yet practically insufficient for classification, they establish that the primary predictive signal must come from nucleotide composition and motif arrangement — the exact features targeted by both the k-mer SVM and the CNN. The CNN's advantage therefore reflects its superior exploitation of that same informational substrate.

---

## 5. Discussion

### 5.1 Representation is the Key Variable

The 18.6 percentage-point absolute improvement (73.07% → 91.70%) achieved by the CNN over the SVM is attributable almost entirely to the difference in sequence representation. The k-mer SVM collapses a DNA sequence into an unordered frequency histogram — destroying all positional and contextual information. The CNN, by contrast, applies learned position-sensitive filters across the sequence, integrating short motif detections (Layer 1, kernel=12 bp) into longer-range motif-motif interactions (Layers 2–3). This hierarchical composition mirrors the actual biology of regulatory sequences: individual TFBSs are recognized by specific DNA-binding domains (PWM-like, local), while functional regulatory elements emerge from the combinatorial arrangement of multiple TFBSs in a defined spatial grammar.

The 5-fold CV results (91.10% ± 0.20%) closely matching held-out test accuracy (91.70%) confirm genuine generalization. The multi-pronged regularization strategy — input noise, batch augmentation, SpatialDropout, L1+L2, BatchNormalization, gradient clipping — collectively prevented the model from overfitting the 452 × 5 = 2,260 input features per sequence.

### 5.2 Limitations

- **Downsampling.** Discarding ~77% of enhancer samples reduces training diversity for the majority class. A class-weighted loss approach on the full 234,783-sample dataset would be informative.

- **Padding artifacts.** N-token padding introduces artificial sequence content. Variable-length architectures (e.g., Transformers with attention masking) would handle sequence length variation more naturally.

- **Cell-type and organism generalizability.** Regulatory elements — especially enhancers — are highly cell-type-specific. A model trained on bulk annotations from one cell type may fail in another context. Cross-organism evaluation is needed.

- **SVM baseline ceiling.** LinearSVC is a lower bound on k-mer-based performance. Kernel SVMs using spectrum or weighted-degree string kernels would be more competitive but were excluded due to O(n²) computational cost.

- **Interpretability depth.** Saliency maps are exploratory. Formal motif discovery requires systematic filter-weight alignment against JASPAR/ENCODE databases with statistical enrichment testing.

### 5.3 Future Directions

Transformer-based genomic language models pre-trained on large sequence corpora (DNABERT, Nucleotide Transformer, HyenaDNA) represent the current state of the art and would be a natural next architecture to evaluate on this benchmark. Incorporating epigenomic auxiliary inputs (H3K4me3, H3K27ac, ATAC-seq signal) could improve enhancer classification specifically, given the context-dependence of enhancer activity. Multi-organism transfer learning — pre-training on vertebrate genomes and fine-tuning on human sequences — could further improve generalizability. Finally, extending the task to finer-grained regulatory annotation (e.g., distinguishing strong vs. weak enhancers, or classifying promoter subtypes such as CpG-island vs. TATA-containing) would provide a more clinically relevant benchmark.

---

## 6. Conclusion

We have demonstrated that a three-block 1D-CNN with multi-layer regularization achieves 91.70% accuracy and macro F1 = 0.917 on three-class DNA regulatory element classification, outperforming an Optuna-tuned k-mer SVM (73.07%) by a 25.50% relative margin. The CNN's perfect recall for promoter sequences (100%) validates the biological hypothesis that strong, conserved positional motifs are detectable by convolutional filters. The modest but consistent difficulty with enhancer/intergenic discrimination mirrors the genuine biological heterogeneity of enhancer sequences. Gradient-based saliency analysis provides qualitative evidence that the learned representations are biologically interpretable. The low cross-validation variance (SD = 0.002) and close agreement with held-out test accuracy confirm robust generalization. These results contribute to the growing body of evidence that positional sequence context — captured by deep convolutional architectures — is the decisive factor in regulatory element classification from primary DNA sequence.

---

## References

Andersson, R., & Sandelin, A. (2020). Determinants of enhancer and promoter activities of regulatory elements. *Nature Reviews Genetics*, 21(2), 71–87.

Ji, Y., Zhou, Z., Liu, H., & Davuluri, R. V. (2021). DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome. *Bioinformatics*, 37(15), 2112–2120.

Kelley, D. R., Snoek, J., & Rinn, J. L. (2016). Basset: learning the regulatory code of the accessible genome with deep convolutional neural networks. *Genome Research*, 26(7), 990–999.

Leslie, C., Eskin, E., & Noble, W. S. (2002). The spectrum kernel: a string kernel for SVM protein classification. *Pacific Symposium on Biocomputing*, 7, 564–575.

Quang, D., & Xie, X. (2016). DanQ: a hybrid convolutional and recurrent deep neural network for quantifying the function of DNA sequences. *Nucleic Acids Research*, 44(11), e107.

Stormo, G. D. (2000). DNA binding sites: representation and discovery. *Bioinformatics*, 16(1), 16–23.

Zhou, J., & Troyanskaya, O. G. (2015). Predicting effects of noncoding variants with deep learning–based sequence model. *Nature Methods*, 12(10), 931–934.