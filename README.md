# Sequence-Based Classification of Regulatory DNA Elements: A Comparative Study of 1D Convolutional Neural Networks and Support Vector Machines

**Authors:** Omri Shlezinger, Nadav Idelson
**Course:** Data Analysis and Machine Learning on Biological Data
**Date:** February 2026

---

## Abstract

Accurate computational identification of regulatory DNA elements from primary sequence remains a fundamental challenge in genomic annotation. The majority of mammalian gene promoters are TATA-less and operate through CpG island-associated mechanisms — a biologically distinct architecture that presents different computational constraints from the well-studied TATA-box paradigm. This study addresses a three-class classification problem: given a variable-length DNA sequence, predict whether it belongs to a **non-TATA promoter**, an **enhancer**, or an **intergenic** region. We assembled a dataset of 234,783 labeled sequences (154,842 enhancers, 43,810 intergenic, 36,131 promoters) derived from genomic annotation, and addressed class imbalance via stratified downsampling to 36,131 samples per class, yielding a balanced dataset of 108,393 sequences. Two modeling approaches were evaluated: (1) a k-mer frequency-based Support Vector Machine (SVM) with Optuna-optimized hyperparameters, and (2) a three-block one-dimensional convolutional neural network (1D-CNN) incorporating reverse-complement augmentation, spatial dropout, and 5-fold stratified cross-validation with early stopping. The SVM achieved 72.24% test accuracy; the CNN achieved **91.22%** accuracy (macro F1 = 0.912), representing a +26.28% relative improvement. The CNN achieved perfect recall for the promoter class (1.000), a result we interpret in light of the characteristic GC-rich, CpG-dense, multi-motif architecture of non-TATA promoters. Gradient-based saliency analysis supports the hypothesis that the CNN captures biologically meaningful positional signals. These results demonstrate that hierarchical position-sensitive representations substantially outperform position-agnostic k-mer statistics for regulatory element classification.

---

## 1. Introduction

### 1.1 The Regulatory Genome and the Problem of Sequence-Based Annotation

The protein-coding portion of the human genome accounts for less than 2% of the ~3.2 billion base pairs of the haploid sequence. The vast remainder consists of non-coding DNA that includes structural elements, repetitive sequences, and an extensive network of *cis*-regulatory elements that govern the spatiotemporal specificity of gene expression. Among these, **promoters** and **enhancers** are the two principal classes responsible for the activation of RNA polymerase II (RNAPII)-mediated transcription. Accurately distinguishing these elements from one another and from the transcriptionally inert **intergenic** background — using only primary DNA sequence — is one of the core tasks of computational genomics.

The biological motivation is substantial. More than 90% of variants identified in genome-wide association studies (GWAS) map to non-coding genomic loci, and the interpretation of their functional significance requires knowing whether the affected sequence belongs to an active regulatory element (Maurano et al., 2012). Erroneous classification of a disease-associated variant as intergenic rather than promoter-proximal or enhancer-embedded can fundamentally misguide mechanistic inference. In synthetic biology and gene therapy, the design of compact artificial gene expression cassettes requires reliable sequence-level predictors of regulatory activity, independent of chromatin context. And in basic genomics, the automated annotation of newly assembled genomes — where epigenomic assay data may be unavailable — demands robust sequence-only models.

### 1.2 Biology of Non-TATA Promoters

The classical view of a mammalian promoter, defined by a TATA box (consensus TATAWAWR) positioned ~30 bp upstream of a focused transcription start site (TSS) and recognized by TATA-binding protein (TBP) within the TFIID complex, applies to only a minority of genes. Genome-wide CAGE (Cap Analysis of Gene Expression) studies have established that fewer than 15% of human promoters contain a canonical TATA box, and fewer than 30% contain a functional initiator (Inr) element (Carninci et al., 2006; Vo Ngoc et al., 2017). The dominant promoter class in the mammalian genome is the **non-TATA promoter**, which is strongly enriched at **CpG islands (CGIs)** — regions of at least 200 bp with GC content exceeding 50% and an observed-to-expected CpG ratio greater than 0.6 (Gardiner-Garden & Frommer, 1987).

CpG island promoters share a biologically coherent set of sequence-level properties that collectively constitute their computational signature:

**Elevated GC content and CpG density.** Non-TATA promoters exhibit markedly higher GC content (often 55–70%) compared to bulk genomic sequence (~41%). This GC enrichment is mechanistically significant: it intrinsically disfavors nucleosome formation, generating a nucleosome-depleted region (NDR) at the promoter through a sequence-directed, transcription-independent mechanism (Fenouil et al., 2012). The resulting open chromatin configuration facilitates the binding of the general transcription machinery without the requirement for energy-dependent chromatin remodeling.

**Dispersed, broad transcription start sites.** In contrast to TATA-driven promoters, which direct focused initiation to a single nucleotide, CpG island promoters drive **dispersed transcription initiation** over a window of 50–100 bp (Carninci et al., 2006). This broad initiation pattern is characteristic of housekeeping genes expressed constitutively across cell types, and reflects the absence of a sharp positional signal like the TATA box.

**Enrichment of specific transcription factor binding motifs.** Non-TATA CGI promoters are consistently enriched for binding sites of ubiquitous transcription factors whose recognition sequences are GC-rich and frequently contain CpG dinucleotides. These include **Sp1** (GC box: GGGCGG), which is present in the proximal promoter of approximately 60% of human genes and can recruit TBP to TATA-less promoters (Butler & Kadonaga, 2002); **NRF-1** (Nuclear Respiratory Factor 1; GCGCATGCGC), a regulator of mitochondrial biogenesis and ribosomal protein genes; **E2F** family members (TTTSSCSS), critical for cell cycle-regulated gene expression; and **ETS** family transcription factors (CCGGAAG), which are enriched in CpG island promoters genome-wide (Landolin et al., 2010). These motifs, each 8–12 bp in length, are the primary functional units whose detection by convolutional filters drives the superior CNN performance on the promoter class.

**TISU and other TATA-independent initiator elements.** A subset of non-TATA promoters employ the Translation Initiator of Short 5′UTR (TISU) element, located +5 to +30 relative to the TSS, which simultaneously specifies the TSS and functions as a translation initiator for genes with short 5′ untranslated regions (Elfakess & Dikstein, 2008). Additional core elements found in TATA-less promoters include XCPE1 (consensus DSGYGGRASM, −8 to +2), which acts in conjunction with Sp1 and NRF1, and the MED-1 element found in promoters with multiple, unclustered start sites.

The combination of elevated GC/CpG content and the co-occurrence of multiple distinct transcription factor binding motifs at defined positions near the TSS gives non-TATA promoters a statistically distinctive sequence composition that — despite lacking a single dominant motif like the TATA box — provides a learnable signal for convolutional architectures. This is the mechanistic basis for the near-perfect recall achieved by our CNN on the promoter class.

### 1.3 Biology of Enhancers

Enhancers are distal *cis*-regulatory elements that can act from tens of kilobases away from their target promoters through chromatin looping, mediated by cohesin and CTCF, to amplify transcriptional output in a cell-type-specific, signal-responsive manner (Pennacchio et al., 2013). Unlike promoters, enhancers are defined by a fundamentally different organizational logic:

**Combinatorial TFBS architecture.** Enhancers function as regulatory logic gates, integrating signals from multiple transcription factors whose binding sites cluster within a 200–500 bp window. The specific combination of factors — rather than any single motif — determines cell-type specificity and signal responsiveness. For example, cardiac enhancers typically require co-occupancy of GATA, NKX2.5, and TBX5, while immune gene enhancers rely on IRF-NF-κB or IRF-BATF composite elements. The AP-1 family (bZIP; consensus TGASTCA) and ETS factors are among the most prevalent motifs in active enhancers across cell types, but their mere presence is insufficient for enhancer activity without appropriate combinatorial context. Active enhancers are epigenomically marked by monomethylation of histone H3 lysine 4 (H3K4me1) together with acetylation of lysine 27 (H3K27ac), and by binding of the p300/CBP histone acetyltransferase (Creyghton et al., 2010; Visel et al., 2009).

**Sequence heterogeneity and positional flexibility.** Unlike promoters, enhancer TFBSs lack fixed positional constraints relative to any reference point, the spacing between individual motifs is variable, and the overall sequence composition is more heterogeneous across the enhancer repertoire. This biological heterogeneity — reflecting the diversity of regulatory programs across different cell types and developmental stages — is a principal reason why the enhancer class achieves lower classification F1 scores (0.880) compared to the promoter class (0.991) in our experiments.

**Partial sequence overlap with promoters.** Genome-wide studies using CAGE and STARR-seq have identified **enhancer RNAs (eRNAs)** — short, bidirectional, unstable transcripts initiated within active enhancers — and have noted that many enhancers share ETS and SP1/KLF binding sites with CpG island promoters (Andersson et al., 2014). This mechanistic convergence at the sequence level creates genuine classification ambiguity between enhancers and promoters that cannot be fully resolved from primary sequence alone, and explains the small fraction of enhancer sequences misclassified as promoters by the CNN.

### 1.4 Computational Approaches and Study Goals

Prior computational work on regulatory element classification from sequence includes k-mer frequency support vector machines (Leslie et al., 2002), profile-based hidden Markov models, and, more recently, deep learning architectures. CNNs applied to genomic sequences — including DeepBind, Basset, and DeepSEA — have demonstrated that convolutional filters of width 8–20 bp can learn de novo representations functionally equivalent to transcription factor position weight matrices (Alipanahi et al., 2015; Kelley et al., 2016; Zhou & Troyanskaya, 2015). Grishkevich and Tsonis (2017) applied CNN architectures specifically to the TATA vs. non-TATA promoter distinction, achieving sensitivity/specificity of 0.90/0.98 for non-TATA promoters in the human genome. The present study extends this line of work to a more challenging three-class problem (non-TATA promoter vs. enhancer vs. intergenic), using a balanced genome-scale dataset and systematically comparing against an Optuna-optimized k-mer SVM baseline.

---

## 2. Dataset and Preprocessing

### 2.1 Raw Data and Class Imbalance

The dataset comprised 234,783 labeled DNA sequences in three classes: Enhancer (154,842; 65.9%), Intergenic (43,810; 18.7%), and Promoter (36,131; 15.4%). The predominance of enhancer sequences reflects the biological reality that the human genome is estimated to harbor 400,000 to over one million active enhancer elements across cell types, whereas gene promoters number in the tens of thousands (Moore et al., 2020). Training directly on this imbalanced distribution would bias classifiers toward the majority enhancer class, suppressing minority-class recall.

We applied stratified random downsampling to the minority class size (n = 36,131 per class, the promoter count), producing a balanced dataset of 108,393 sequences shuffled with a fixed seed (random_state = 42). A 70/15/15 stratified split yielded:

| Partition | Samples | Purpose |
|---|---|---|
| Train | 75,918 | Model fitting |
| Validation | 16,216 | Hyperparameter selection and early stopping |
| Test | 16,259 | Final, held-out evaluation |

The test partition was strictly isolated and never accessed during any phase of model development or selection.

### 2.2 Sequence Length Distribution

Sequence lengths follow a mildly right-skewed unimodal distribution (mean = 249.5 bp, SD = 110.5 bp). Shapiro-Wilk tests on 5,000-sample class subsets rejected normality for all three classes (p < 10⁻¹⁶), consistent with the biological heterogeneity of sequence lengths across different regulatory element classes and annotation methods. We therefore used the Kruskal-Wallis test (a non-parametric one-way analysis of variance) as the primary inferential tool for comparing length distributions across classes.

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

The Kruskal-Wallis test confirmed that length distributions differ significantly across classes (H = 2,332.18, p ≈ 0). All three pairwise Mann-Whitney U comparisons remained significant after Bonferroni correction (adjusted α = 0.0167): Enhancer vs. Promoter (U = 718,157,822, p = 1.66×10⁻¹³⁷), Enhancer vs. Intergenic (U = 769,457,772, p ≈ 0), Promoter vs. Intergenic (U = 748,796,910, p = 5.32×10⁻²⁹⁴). However, the moderate absolute differences in class-level medians confirm that sequence length alone provides only weak discriminative signal, and that the primary classification signal must reside in nucleotide composition and motif architecture.

For CNN input, all sequences were padded with a sentinel 'N' character to a fixed length of FINAL_LEN = 452 bp — the 95th-percentile sequence length — using post-padding (trailing Ns). Sequences exceeding 452 bp (~5% of the dataset) were truncated from the 3′ end. This threshold preserves the complete biological sequence for 95% of samples. The use of Global Average Pooling in the CNN architecture (Section 3.2) mitigates the contribution of padding positions to the final representation by averaging activations across all sequence positions.

---

## 3. Methods

### 3.1 SVM with K-mer Features

#### 3.1.1 Feature Representation

Each DNA sequence was transformed into a fixed-length feature vector of k-mer frequencies — the counts of all contiguous subsequences of length k. This approach, analogous to a bag-of-words model in natural language processing, captures local oligonucleotide composition but is strictly position-agnostic: the same k-mer appearing at any position in the sequence contributes identically to the feature vector, regardless of its genomic context. For non-TATA promoters specifically, this is a fundamental limitation: an Sp1 binding site (GGGCGG) in the proximal promoter region immediately upstream of the TSS is functionally distinct from the same hexamer appearing in an exon or intergenic region, but both contribute identically to the k-mer frequency vector.

Features were extracted using scikit-learn's *CountVectorizer* with `analyzer='char'` and `ngram_range=(k, k)`, yielding sparse count matrices of dimension n × V, where V is the number of unique k-mers observed in the training corpus. Raw counts were used without normalization, consistent with the scale-invariance of the LinearSVC objective.

#### 3.1.2 Hyperparameter Optimization

The search space comprised k ∈ {3, 4, 5, 6} (k-mer size) and C ∈ [10⁻³, 10] (log-uniform, the SVM regularization strength). Optimization was performed using Optuna with a Tree-structured Parzen Estimator (TPE) sampler over 15 trials, evaluated on a stratified random subsample of 15,000 training sequences with an internal 75/25 train-validation split. The optimal configuration was k = 5, C ≈ 0.001. LinearSVC (dual=False, max_iter=5,000) was used throughout; a linear kernel is the computationally tractable choice at this feature dimensionality and sample size, and non-linear kernel SVMs with string kernels (e.g., the spectrum kernel) were excluded due to O(n²) memory and time scaling.

### 3.2 1D-CNN Architecture

#### 3.2.1 Sequence Encoding

Sequences were integer-encoded (A→0, C→1, G→2, T→3, N→4) and padded to FINAL_LEN = 452 bp using Keras `pad_sequences` (post-padding, pad value 4). One-hot encoding was applied via `tf.one_hot(depth=5)`, producing input tensors of shape (452, 5). One-hot encoding assigns equal Euclidean distance between all nucleotide identities — biologically appropriate since no ordinal relationship exists among the four bases — and allows each convolutional filter to learn an arbitrary position-specific nucleotide preference function, equivalent to a position weight matrix (PWM).

#### 3.2.2 Network Architecture

The model is a sequential three-block 1D-CNN followed by global average pooling and a dense classifier head:

**Table 2. 1D-CNN Architecture**

| Layer | Filters | Kernel (bp) | Output Shape | Notes |
|---|---|---|---|---|
| Input | — | — | (452, 5) | One-hot encoded |
| GaussianNoise | — | — | (452, 5) | σ = 0.1; training only |
| Conv1D Block 1 | 32 | 12 | (441, 32) | ReLU; L1=10⁻⁴, L2=10⁻³ |
| BatchNormalization | — | — | (441, 32) | |
| SpatialDropout1D | — | — | (441, 32) | p = 0.20 |
| MaxPooling1D | — | 4 | (110, 32) | |
| Conv1D Block 2 | 64 | 8 | (103, 64) | ReLU; L1=10⁻⁴, L2=10⁻³ |
| BatchNormalization | — | — | (103, 64) | |
| SpatialDropout1D | — | — | (103, 64) | p = 0.30 |
| MaxPooling1D | — | 4 | (25, 64) | |
| Conv1D Block 3 | 32 | 6 | (20, 32) | ReLU; L1=10⁻⁴, L2=10⁻³ |
| BatchNormalization | — | — | (20, 32) | |
| GlobalAveragePooling1D | — | — | (32,) | Aggregate over positions |
| Dense | 64 | — | (64,) | ReLU; L1+L2 regularization |
| BatchNormalization + Dropout | — | — | (64,) | p = 0.50 |
| Dense (output) | 3 | — | (3,) | Softmax |

The first convolutional block uses a kernel of 12 bp — within the typical size range of eukaryotic transcription factor binding motifs (8–20 bp) — allowing individual filters to learn position-sensitive nucleotide preference patterns analogous to scanning PWMs. The second block (kernel 8 bp) integrates adjacent motif detections, while the third block (kernel 6 bp) captures shorter-range compositional features such as CpG dinucleotide clustering. GlobalAveragePooling aggregates activation maps across all sequence positions before the dense layers, providing translational invariance with respect to the absolute genomic coordinate of detected motifs and diluting the contribution of N-padded positions.

#### 3.2.3 Data Augmentation: Reverse Complement

The notebook implements **reverse complement (RC) augmentation** applied at the per-sample level prior to each training fold. For each training sequence, the reverse complement is computed with 50% probability using the mapping A↔T, C↔G (N→N), implemented in NumPy for efficiency. This augmentation is biologically grounded: RNAPII-mediated transcription can originate from either strand, and many regulatory motifs (e.g., palindromic TFBSs) are recognized in both orientations. RC augmentation effectively doubles the diversity of training examples, reduces strand-specific overfitting, and has been widely adopted in genomic deep learning (Shrikumar et al., 2017). Augmentation was applied only to training data; validation and test sets were evaluated on the original strand orientation.

#### 3.2.4 Additional Regularization

Five complementary regularization strategies were applied to prevent overfitting on the 452 × 5 = 2,260-dimensional input space:

- **GaussianNoise (σ = 0.1)** on the input layer introduces stochastic perturbations to one-hot encodings during training, functioning as continuous-valued data augmentation that prevents the model from exploiting exact input values.
- **L1 + L2 kernel regularization** (L1 = 10⁻⁴, L2 = 10⁻³) on all Conv1D and Dense kernel weights penalizes both weight magnitude (L2, preventing reliance on any single large-weight feature) and promotes sparsity (L1, encouraging filters to specialize on discrete motif-like patterns).
- **SpatialDropout1D** (p = 0.20–0.30) randomly zeroes entire feature maps during training, which is more effective than standard dropout for correlated sequential data such as convolutional activations along a DNA sequence.
- **BatchNormalization** after each convolutional block normalizes activation distributions to zero mean and unit variance, stabilizing gradient magnitudes and providing an additional implicit regularization effect.
- **Gradient clipping** (clipnorm = 1.0) in the Adam optimizer prevents exploding gradients that can destabilize training on variable-complexity sequence inputs.

### 3.3 Training Protocol and Cross-Validation

Training used categorical cross-entropy loss and Adam (lr = 3×10⁻⁴, clipnorm = 1.0) with batch size 128. Two callbacks governed training duration: **EarlyStopping** (monitor = val_loss, patience = 5, restore_best_weights = True) halted training when validation loss ceased to decrease, preventing overfitting; **ReduceLROnPlateau** (monitor = val_loss, factor = 0.5, patience = 3, min_lr = 10⁻⁶) halved the learning rate upon validation loss plateaus, allowing finer convergence near optima. The maximum epoch budget was 30 per fold.

**5-fold stratified cross-validation** was performed on the combined train+validation pool (n = 92,134 sequences). Stratification maintained the 1:1:1 class ratio in every fold. For each fold, a new model was instantiated with fresh random weight initialization, trained with the above protocol, and evaluated on the held-out fold. Weights from the fold achieving the lowest validation loss were saved to disk; this model was subsequently loaded for final test evaluation. The held-out test set (n = 16,259) was never accessed during cross-validation.

---

## 4. Results

### 4.1 SVM Baseline — 72.24% Test Accuracy

The k-mer SVM (k = 5, C ≈ 0.001, LinearSVC) achieved 72.24% accuracy on the held-out test set. This represents a substantial improvement over random chance (33.3%), indicating that tetranucleotide composition does carry discriminative information across the three regulatory classes. However, the SVM's performance ceiling is set by the fundamental information loss of the k-mer representation: the frequency of GCGG tetramers, for example, does not distinguish between an Sp1 binding site embedded in a non-TATA promoter and the same tetranucleotide appearing in a different sequence context without transcriptional regulatory significance. The SVM cannot represent co-occurrences of multiple motifs at defined relative positions, nor can it detect the broad GC enrichment pattern extending over 200–500 bp that characterizes CpG island promoters, because k-mer features decompose the sequence into non-overlapping local windows without any global integration.

### 4.2 CNN Cross-Validation Results

**Table 3. 5-Fold Stratified Cross-Validation Results**

| Fold | Val Accuracy | Val AUC (approx.) |
|---|---|---|
| 1 | 0.9131 | ~0.985 |
| 2 | 0.9105 | ~0.985 |
| 3 | 0.9104 | ~0.985 |
| 4 | 0.9078 | ~0.985 |
| **5 ★** | **0.9132** | **~0.986** |
| **Mean ± SD** | **0.9110 ± 0.0020** | — |

*★ = model selected for final test evaluation (lowest validation loss).*

The remarkably low cross-validation variance (SD = 0.002) across independently initialized and trained folds confirms that the reported performance is not an artifact of a particular favorable data partition. The near-identical AUC values (~0.985) across folds indicate consistent discriminative ability across the full operating range of the classifier for all three classes simultaneously. The close agreement between mean cross-validation accuracy (91.10%) and held-out test accuracy (91.22%) demonstrates that the multi-layered regularization strategy successfully prevented overfitting during model development.

### 4.3 Final Evaluation on the Held-Out Test Set

**Table 4. CNN Classification Report (Test Set, n = 16,259)**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| ENHANCER | 0.879 | 0.881 | 0.880 | 5,419 |
| **PROMOTER** | **0.983** | **1.000** | **0.991** | **5,420** |
| INTERGENIC | 0.888 | 0.870 | 0.879 | 5,420 |
| Macro Average | 0.916 | 0.917 | 0.917 | 16,259 |
| **Overall Accuracy** | — | — | **91.22%** | 16,259 |

*Test loss: 0.2929.*

The CNN's most striking result is the **perfect recall on the PROMOTER class** (recall = 1.000, precision = 0.983). Not a single promoter sequence in the 5,420-sample test set was misclassified as enhancer or intergenic. The biological interpretation of this result is directly tied to the non-TATA promoter architecture described in Section 1.2. Non-TATA CGI promoters possess a distinctive multivariate sequence signature — globally elevated GC content, high CpG density across a 200–500 bp window, and a specific co-occurrence of Sp1, NRF-1, E2F, and ETS binding motifs at positions proximal to the TSS — that is collectively learnable by the convolutional architecture. The 12-bp first-layer filters are well-suited to detect individual GC-box (GGGCGG) or NRF-1 (GCGCATGCGC) motifs, the 8-bp second-layer filters integrate pairwise motif relationships, and the GlobalAveragePooling integrates the overall CpG-enriched composition across the entire padded window.

The lower F1 scores for ENHANCER (0.880) and INTERGENIC (0.879) reflect two distinct biological realities. For enhancers, the combinatorial, cell-type-dependent nature of TFBS arrangements — with no fixed positional grammar relative to any reference coordinate — limits the ability of any sequence-only model to generalize across the full diversity of enhancer sequences. Furthermore, some enhancers share GC-rich, ETS- and SP1/KLF-enriched sequences with CpG island promoters (as documented by CAGE-seq analyses of transcribed enhancers), creating genuine sequence-level ambiguity. For intergenic sequences, the possibility that some annotated intergenic regions harbor unannotated regulatory elements — including poised or cell-type-restricted enhancers not captured in the training annotation — introduces label noise that is not resoluble from sequence features alone.

**Table 5. Model Comparison on the Held-Out Test Set**

| Model | Accuracy | Macro F1 | McNemar p-value |
|---|---|---|---|
| SVM (k = 5, LinearSVC) | 72.24% | ~0.730 | — |
| 1D-CNN (Fold 5, best) | **91.22%** | **0.917** | < 0.001 |
| Relative improvement | +26.28% | — | — |

A two-sided McNemar's test on the paired binary correctness vectors of the SVM and CNN on the shared test set yielded p < 0.001, confirming that the performance differential reflects a genuine difference in learned representations rather than sampling variability. McNemar's test is appropriate here because the two models are evaluated on identical test samples, making their errors structurally correlated rather than independent.

### 4.4 Model Interpretability

**First-layer filter visualization.** The 32 learned convolutional filters (kernel size 12 bp × 5 channels) were visualized as position weight matrices over the nucleotide alphabet {A, C, G, T, N}. Several filters display characteristic patterns consistent with known regulatory motifs: GC-rich patterns with CpG enrichment resembling the Sp1 GC-box consensus (GGGCGG), and purine-rich patterns consistent with ETS family binding sites (CCGGAAG/GGA). This qualitative correspondence between learned filter weights and known TFBSs is consistent with earlier reports that 1D-CNNs trained on genomic sequences de novo recover functional motif-like representations (Alipanahi et al., 2015; Kelley et al., 2016), and supports the mechanistic interpretation that the CNN's promoter recall reflects detection of the GC-box and related CpG-containing motifs that characterize non-TATA CGI promoters.

**Gradient-based saliency maps.** Per-sequence saliency maps were computed using `tf.GradientTape`: the gradient of the predicted class logit with respect to the one-hot input tensor was obtained, and positive gradient components (∂logit/∂input > 0) were retained — positions where increasing the one-hot value of a given nucleotide would increase the model's confidence in the predicted class. Class-averaged saliency maps (n = 20 samples per class) were computed to obtain stable, generalizable attribution patterns. For the PROMOTER class, saliency is concentrated in discrete, high-amplitude peaks distributed across multiple positions in the upstream region, consistent with the multi-motif, dispersed TSS architecture of CpG island promoters. For the ENHANCER class, saliency is more broadly distributed and less structured, reflecting the combinatorial positional flexibility of enhancer TFBSs. For the INTERGENIC class, saliency is lower in amplitude and less localized, consistent with the absence of canonical regulatory motif arrangements.

These interpretability analyses are exploratory and do not constitute formal motif discovery. A rigorous computational analysis would require systematic alignment of learned filter weights against JASPAR (Fornes et al., 2020) or ENCODE TF motif databases, followed by statistical enrichment testing with appropriate multiple-testing correction.

---

## 5. Discussion

### 5.1 Why the CNN Outperforms the SVM: A Representational Analysis

The 18.6 percentage-point absolute improvement in test accuracy (72.24% → 91.22%) between the SVM and CNN is attributable to a fundamental difference in how each model represents DNA sequence information. The k-mer SVM constructs a fixed-length feature vector from the global frequency of all tetranucleotides in a sequence. This representation discards all positional information: the same tetranucleotide appearing at position +5 relative to the TSS (where it might constitute the core of an Sp1 binding site) and at position +300 (where it is contextually neutral) are treated identically. For non-TATA promoters, where functional identity is encoded in the co-occurrence of multiple specific motifs at defined positions upstream and downstream of the TSS, this positional blindness eliminates the most discriminative information.

The CNN, by contrast, applies a battery of learned position-sensitive filters along the full length of the input sequence. The three-layer hierarchical architecture mirrors the multi-scale structure of regulatory information: individual 12-bp filters detect single TFBSs or CpG-dense sub-sequences; 8-bp second-layer filters integrate signals from pairs of adjacent features; and the global average pooling aggregates across the entire 452-bp window, making the final classification sensitive to the overall composition pattern — including the extended GC/CpG enrichment that characterizes CGI promoters even in the absence of individual motif matches. This hierarchical integration of local and global sequence features is precisely what the biological regulatory code demands, and it is inaccessible to the k-mer SVM.

The reverse complement augmentation also contributes to the CNN's generalization. Regulatory motifs occur on both DNA strands; training without RC augmentation would require the model to independently learn both orientations of each motif from the data, effectively halving the functional training set size for each motif. By explicitly presenting RC-transformed sequences during training, the augmentation enforces strand-symmetric learning and reduces the effective input space the model must cover.

### 5.2 Interpreting Perfect Promoter Recall in the Context of Non-TATA Promoter Biology

The perfect recall of the PROMOTER class (1.000) deserves careful interpretation. It does not imply that non-TATA promoters are trivially identifiable — the SVM achieves only ~73% overall accuracy — but rather that the CNN's learned representation is sufficient to capture the class-defining features of this specific promoter subset. We attribute this to three complementary factors:

First, non-TATA CGI promoters exhibit **global sequence composition differences** from both enhancers and intergenic sequences: their elevated GC content (55–70%) and CpG density are detectable by convolutional filters even before any motif-level specificity is engaged, effectively providing a compositional pre-filter.

Second, the **co-occurrence of multiple GC-rich TFBS motifs** (Sp1, NRF-1, E2F, ETS) at positions proximal to the TSS — a hallmark of CGI promoters associated with housekeeping genes — creates a distinctive local sequence grammar that convolutional filters with kernel sizes of 12 and 8 bp are well-positioned to detect.

Third, the **broader sequence context** over the 452-bp window, captured through the global average pooling, integrates the extended CpG island structure (typically 200–1,500 bp) into a single fixed-length representation, providing a signal that neither the SVM nor a purely local classifier can access.

That no promoter sequence was misclassified as intergenic reflects the extent to which non-TATA promoter sequences are compositionally distinct from bulk genomic intergenic sequence, even in the absence of the TATA box. That 1.7% of predicted promoters were false positives (precision = 0.983) — likely enhancer or GC-rich intergenic sequences misidentified as promoters — is consistent with the known partial sequence overlap between CpG island promoters and GC-rich active enhancers.

### 5.3 Limitations

**Downsampling bias.** Retaining only 36,131 of 154,842 enhancer sequences (~23%) restricts the compositional diversity of enhancer training examples. The discarded sequences may include cell-type-specific enhancers with sequence features not represented in the retained sample, reducing generalization to the full enhancer repertoire. An analysis with class-weighted loss on the full imbalanced dataset would provide a complementary evaluation.

**Sequence-only classification.** The models described here operate solely on primary DNA sequence, without access to epigenomic context (chromatin accessibility, histone modification profiles, TF occupancy) that strongly predicts regulatory activity in vivo. The sequence signature alone does not capture enhancer cell-type specificity: the same sequence may be active in one cell type (enriched for H3K27ac and bound by lineage-specific TFs) and silent in another. Incorporating epigenomic auxiliary inputs would improve classification performance, particularly for the enhancer class.

**Annotation quality and label noise.** The training labels are derived from genomic annotation databases that may themselves contain errors or incomplete annotations. Some sequences labeled as intergenic may harbor unannotated or cell-type-restricted regulatory elements; some enhancer sequences may have been misannotated or may represent sequence features that are regulatory in one context but not the one from which the training labels were derived. This label noise sets an irreducible ceiling on classification performance.

**LinearSVC as a lower bound on SVM performance.** The spectrum kernel and weighted-degree string kernels are the theoretically appropriate kernelized analogs of k-mer features; non-linear SVMs using these kernels are known to be more competitive on sequence classification tasks (Leslie et al., 2002) but scale as O(n²) in memory and time, making them computationally intractable at our sample sizes. The LinearSVC results should be interpreted as a lower bound on the k-mer SVM approach.

**Exploratory interpretability.** The saliency map and filter visualization analyses provide qualitative biological plausibility but are not statistically validated. Formal motif discovery requires systematic database alignment (e.g., against JASPAR 2022), permutation-based enrichment testing, and ideally experimental validation through reporter assays or ChIP-seq of the putative binding factors.

### 5.4 Future Directions

Several lines of improvement are worth exploring. Transformer-based genomic language models pre-trained on large multi-species sequence corpora — including DNABERT (Ji et al., 2021), the Nucleotide Transformer (Dalla-Torre et al., 2023), and HyenaDNA — represent the current state of the art for genomic sequence classification and would provide a natural comparison point for the architecture presented here. These models leverage long-range dependencies in sequence that convolutional architectures cannot directly access. Incorporating epigenomic auxiliary inputs — ATAC-seq signal, H3K4me1, H3K4me3, and H3K27ac ChIP-seq profiles — as parallel input modalities could substantially improve enhancer classification by providing the cell-type context that primary sequence alone cannot encode. Finally, distinguishing subclasses within the promoter category — such as CpG-island vs. non-CGI non-TATA promoters, or promoters of housekeeping vs. tissue-specific genes — would constitute a more granular and clinically relevant benchmark.

---

## 6. Conclusion

We have demonstrated that a three-block 1D-CNN with reverse complement augmentation, multi-layer regularization, and early stopping achieves 91.22% accuracy and macro F1 = 0.917 on a balanced three-class regulatory DNA classification task, outperforming an Optuna-tuned k-mer SVM (72.24%) by a 25.50% relative margin. The CNN's perfect recall for non-TATA promoter sequences (recall = 1.000) is mechanistically interpretable: the convolutional architecture can jointly detect the elevated GC/CpG composition, the co-occurrence of Sp1/NRF-1/E2F/ETS binding motifs, and the broad compositional context that collectively define CpG island-associated non-TATA promoters — features that the position-agnostic k-mer SVM cannot represent. The lower but substantial F1 scores for enhancers and intergenic sequences reflect genuine biological challenges: the combinatorial, cell-type-specific TFBSs grammar of enhancers and the potential for label noise in intergenic annotation. Low cross-validation variance (SD = 0.002) and close agreement between cross-validation and test accuracy confirm robust generalization. The results contribute to the growing evidence that hierarchical, position-sensitive sequence representations — as learned by convolutional architectures — capture the regulatory code of non-coding DNA more faithfully than aggregate compositional statistics.

---

## References

Alipanahi, B., Delong, A., Weirauch, M. T., & Frey, B. J. (2015). Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning. *Nature Biotechnology*, 33(8), 831–838.

Andersson, R., Gebhard, C., Miguel-Escalada, I., Hoof, I., Bornholdt, J., Boyd, M., … & Sandelin, A. (2014). An atlas of active enhancers across human cell types and tissues. *Nature*, 507(7493), 455–461.

Andersson, R., & Sandelin, A. (2020). Determinants of enhancer and promoter activities of regulatory elements. *Nature Reviews Genetics*, 21(2), 71–87.

Butler, J. E., & Kadonaga, J. T. (2002). The RNA polymerase II core promoter: a key component in the regulation of gene expression. *Genes & Development*, 16(20), 2583–2592.

Carninci, P., Sandelin, A., Lenhard, B., Katayama, S., Shimokawa, K., Ponjavic, J., … & Hayashizaki, Y. (2006). Genome-wide analysis of mammalian promoter architecture and evolution. *Nature Genetics*, 38(6), 626–635.

Creyghton, M. P., Cheng, A. W., Welstead, G. G., Kooistra, T., Carey, B. W., Steine, E. J., … & Jaenisch, R. (2010). Histone H3K27ac separates active from poised enhancers and predicts developmental state. *Proceedings of the National Academy of Sciences*, 107(50), 21931–21936.

Dalla-Torre, H., González, L., Mendoza-Revilla, J., Carranza, L. C., Grzywaczewski, A. H., Oteri, F., … & Lopez-Paz, D. (2023). The Nucleotide Transformer: building and evaluating robust foundation models for human genomics. *bioRxiv*. https://doi.org/10.1101/2023.01.11.523679

Elfakess, R., & Dikstein, R. (2008). A translation initiation element specific to mRNAs with very short 5′UTR that also regulates transcription. *PLoS ONE*, 3(8), e3094.

Fenouil, R., Cauchy, P., Koch, F., Descostes, N., Cabeza, J. Z., Innocenti, C., … & Andrau, J.-C. (2012). CpG islands and GC content dictate nucleosome depletion in a transcription-independent manner at mammalian promoters. *Genome Research*, 22(12), 2399–2408.

Fornes, O., Castro-Mondragon, J. A., Khan, A., van der Lee, R., Zhang, X., Richmond, P. A., … & Mathelier, A. (2020). JASPAR 2020: update of the open-access database of transcription factor binding profiles. *Nucleic Acids Research*, 48(D1), D87–D92.

Gardiner-Garden, M., & Frommer, M. (1987). CpG islands in vertebrate genomes. *Journal of Molecular Biology*, 196(2), 261–282.

Ji, Y., Zhou, Z., Liu, H., & Davuluri, R. V. (2021). DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome. *Bioinformatics*, 37(15), 2112–2120.

Kelley, D. R., Snoek, J., & Rinn, J. L. (2016). Basset: learning the regulatory code of the accessible genome with deep convolutional neural networks. *Genome Research*, 26(7), 990–999.

Landolin, J. M., Johnson, D. S., Trinklein, N. D., Aldred, S. F., Medina, C., Shulha, H., … & Myers, R. M. (2010). Sequence features that drive human promoter diversity. *Genome Research*, 20(7), 890–898.

Leslie, C., Eskin, E., & Noble, W. S. (2002). The spectrum kernel: a string kernel for SVM protein classification. *Pacific Symposium on Biocomputing*, 7, 564–575.

Maurano, M. T., Humbert, R., Rynes, E., Thurman, R. E., Haugen, E., Wang, H., … & Stamatoyannopoulos, J. A. (2012). Systematic localization of common disease-associated variation in regulatory DNA. *Science*, 337(6099), 1190–1195.

Moore, J. E., Purcaro, M. J., Pratt, H. E., Epstein, C. B., Shoresh, N., Adrian, J., … & Bernstein, B. E. (2020). Expanded encyclopaedias of DNA elements in the human and mouse genomes. *Nature*, 583(7818), 699–710.

Pennacchio, L. A., Bickmore, W., Dean, A., Nobrega, M. A., & Bejerano, G. (2013). Enhancers: five essential questions. *Nature Reviews Genetics*, 14(4), 288–295.

Shrikumar, A., Greenside, P., & Kundaje, A. (2017). Learning important features through propagating activation differences. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, 3145–3153.

Visel, A., Blow, M. J., Li, Z., Zhang, T., Akiyama, J. A., Holt, A., … & Pennacchio, L. A. (2009). ChIP-seq accurately predicts tissue-specific activity of enhancers. *Nature*, 457(7231), 854–858.

Vo Ngoc, L., Wang, Y. L., Kassavetis, G. A., & Kadonaga, J. T. (2017). The punctilious RNA polymerase II core promoter. *Genes & Development*, 31(13), 1289–1301.

Zhou, J., & Troyanskaya, O. G. (2015). Predicting effects of noncoding variants with deep learning–based sequence model. *Nature Methods*, 12(10), 931–934.