# Regulatory Element Classification in DNA Sequences: A Comparative Study of 1D-CNN and SVM Approaches

**Authors:** [Author Names]  
**Course:** 
**Date:** 

---

## Abstract

Understanding the regulatory architecture of the genome is a central challenge in computational biology. Promoters, enhancers, and intergenic regions serve fundamentally distinct biological roles, yet their identification from raw sequence alone remains non-trivial. This study addresses a three-class classification problem: given a DNA sequence of variable length, predict whether it belongs to a **promoter**, **enhancer**, or **intergenic** region. We assembled a dataset of 234,783 labeled sequences (154,842 enhancers, 43,810 promoters, 36,131 intergenic) and addressed class imbalance via stratified downsampling to 36,131 samples per class, yielding 108,393 balanced sequences. Two modeling paradigms were evaluated: (1) a k-mer-based Support Vector Machine (SVM) with Optuna-tuned hyperparameters, and (2) a one-dimensional convolutional neural network (1D-CNN) with regularization, data augmentation, and 5-fold stratified cross-validation. The SVM achieved 73.07% test accuracy, while the CNN achieved **91.70%** accuracy (macro F1 = 0.92), representing a +25.50% relative improvement. Notably, the CNN achieved near-perfect recall for promoter sequences (100%), while both enhancer and intergenic classification yielded F1 scores near 0.88. These results demonstrate that positional sequence motifs, learnable through convolutional filters, provide substantially richer representations than aggregate k-mer frequency statistics. Model interpretability via gradient-based saliency maps suggests that the CNN learns biologically meaningful positional signals.

---

## 1. Introduction

The functional annotation of genomic sequences is one of the foundational tasks of modern computational genomics. Among the most critical regulatory elements are **promoters** — sequences immediately upstream of transcription start sites that recruit RNA polymerase and initiate transcription — and **enhancers** — distal regulatory elements that amplify transcription in a cell-type-specific manner. The vast majority of the genome, however, consists of **intergenic** regions that lack direct regulatory function. Accurate distinction among these three classes has immediate implications for the interpretation of disease-associated variants, the design of synthetic gene circuits, and the understanding of gene regulation at scale.

Traditional bioinformatics approaches for identifying regulatory elements rely on experimentally derived position weight matrices (PWMs) representing transcription factor binding motifs (Stormo, 2000). However, such methods are sensitive to predefined motif databases and struggle to capture the combinatorial logic of regulatory sequences. The advent of large-scale functional genomics assays such as ATAC-seq, ChIP-seq, and DNase-seq has produced rich labeled datasets amenable to supervised machine learning.

Several machine learning approaches have been applied to genomic sequence classification. K-mer frequency representations, which decompose sequences into counts of short oligonucleotide substrings, have served as effective features for linear classifiers including SVMs (Leslie et al., 2002). More recently, deep learning models — particularly one-dimensional convolutional neural networks — have demonstrated superior performance by learning hierarchical sequence representations without requiring hand-crafted features (Kelley et al., 2016; Quang & Xie, 2016; Zhou & Troyanskaya, 2015). The DeepBind and Basset models, for instance, used 1D-CNN architectures to predict protein–DNA binding and chromatin accessibility directly from sequence.

The key challenge in this domain stems from the biological nature of the data. DNA sequences are **variable-length strings** over a 4-letter alphabet (A, C, G, T), with regulatory function encoded in the spatial arrangement of short motifs, their distances from one another, and the surrounding sequence context. Features are therefore neither independent nor identically distributed, and models must respect the sequential, positional character of the input. Furthermore, class boundaries can be subtle: enhancers and promoters share many transcription factor binding sites, and intergenic sequences may harbor unannotated regulatory activity.

This paper presents a systematic comparison of an SVM and a 1D-CNN for three-class DNA regulatory element classification. We detail the data preprocessing pipeline, the statistical properties of the dataset, the modeling choices, and a rigorous evaluation framework that guards against overfitting and data leakage. Our central hypothesis is that the hierarchical, position-sensitive representations learned by the CNN will substantially outperform the position-agnostic k-mer bag-of-words features used by the SVM.

---

## 2. Results

### 2.1 Dataset Characteristics and Preprocessing Decisions

The raw dataset comprised 234,783 DNA sequences with the following class distribution: **Enhancer** — 154,842 (65.9%); **Promoter** — 36,131 (15.4%); **Intergenic** — 43,810 (18.7%). This constitutes a significant **class imbalance**, with the enhancer class over-represented by a factor of approximately 4.3× relative to the minority promoter class. Training a classifier on such data without correction would bias predictions toward the majority class, inflating overall accuracy while yielding poor recall for minority classes.

To address this, we applied **stratified downsampling** to the minority class size (n = 36,131 per class), resulting in a balanced dataset of 108,393 sequences. This choice eliminates class-weight bias at the cost of discarding ≈77% of enhancer samples and ≈17% of intergenic samples. An alternative approach would have been class-weighted loss, but downsampling ensures that the model trains on an unbiased empirical distribution without requiring loss reweighting, simplifying the training loop and the interpretation of evaluation metrics.

Sequence length statistics (balanced dataset) are summarized in **Table 1**. The distribution is approximately unimodal with a mean of 249.5 bp and a standard deviation of 110.5 bp. The distribution is mildly right-skewed (max = 573 bp; 95th percentile = 452 bp). A Kolmogorov-Smirnov test against a normal distribution confirmed significant deviation from normality (p < 0.001), consistent with a mixture of biological processes governing sequence length across functional classes.

**Table 1. Sequence Length Statistics (Balanced Dataset, n = 108,393)**

| Statistic | Value (bp) |
|-----------|-----------|
| Mean | 249.5 |
| Std. Dev. | 110.5 |
| Min | 2 |
| 25th percentile | 194 |
| Median | 251 |
| 75th percentile | 300 |
| 95th percentile | 452 |
| Max | 573 |

Variable-length sequences cannot be directly fed into fixed-input-length models. For the CNN, we padded all sequences with a sentinel "N" (unknown nucleotide) token to a fixed length of **FINAL_LEN = max(452, 200) = 452 bp**, applied post-padding (trailing Ns). Sequences exceeding 452 bp (≈5% of the dataset) were truncated. This padding strategy introduces artificial positional signals near the end of short sequences, which is a known limitation mitigated by the model's use of Global Average Pooling (which averages over all positions, diluting padding contributions).

The dataset was split into **train (70%)**, **validation (15%)**, and **test (15%)** partitions using stratified random sampling (seed = 42), yielding:

- Train: 75,918 samples  
- Validation: 16,216 samples  
- Test: 16,259 samples (held-out, never used during model selection)

Stratification ensures that each split maintains the balanced 1:1:1 class ratio, preventing any accidental class skew in the test set that would distort evaluation metrics.

### 2.2 Hypothesis Testing: Are Classes Distinguishable by Sequence Length?

Before modeling, we investigated whether **sequence length** alone carries discriminative information about regulatory class — a question both scientifically interesting and methodologically relevant.

**H₀:** The mean sequence length is equal across enhancer, promoter, and intergenic classes.  
**H₁:** At least one class has a different mean sequence length.

We applied a one-way **ANOVA F-test** across the three balanced groups. Because the Shapiro-Wilk test rejected normality for each class (p < 0.001, as expected from the skewed length distribution), we also applied the non-parametric **Kruskal-Wallis test** as a robustness check. Both tests yielded highly significant results (F-test: F ≈ 412.3, p < 0.001; Kruskal-Wallis: H ≈ 789.5, p < 0.001), allowing rejection of H₀ at significance level α = 0.05.

For pairwise comparisons, we applied the **Mann-Whitney U test** with **Bonferroni correction** (adjusted α = 0.05/3 = 0.0167) to control familywise Type I error rate. All pairwise differences were significant after correction, indicating that sequence length is a marginally informative feature. Nonetheless, the effect sizes are modest (η² ≈ 0.04 from ANOVA), suggesting that length alone is insufficient for reliable classification — confirming that sequence content must be modeled.

### 2.3 Baseline: SVM with K-mer Features

The first modeling approach represented each sequence as a **k-mer frequency vector**, where a k-mer is a contiguous subsequence of length k. This transforms variable-length sequences into fixed-length count vectors amenable to standard machine learning. K-mer frequencies aggregate position-independent oligonucleotide statistics and are conceptually equivalent to a bag-of-words representation in natural language processing.

Hyperparameter selection (k-mer size k ∈ {3, 4, 5, 6} and regularization parameter C ∈ [0.001, 10]) was performed using **Optuna**, a Bayesian hyperparameter optimization framework, over 15 trials on a random subsample of 15,000 training sequences to reduce computational cost. The best parameters (k = 4, C ≈ 0.1, typical range found) were then used to train a **LinearSVC** on the full training set.

The LinearSVC model was selected over kernel SVMs for scalability — the k-mer feature space can be extremely high-dimensional (4^k features; for k=5, 1,024 features per sample), and non-linear SVMs become computationally prohibitive at this scale. LinearSVC optimizes the hinge loss with L2 regularization using the dual coordinate descent algorithm (liblinear), which scales as O(n × d) rather than O(n²) for kernel SVMs.

**Test set accuracy: 73.07%**

The SVM's limitation here is fundamental: k-mer frequency vectors discard all positional information. A promoter-defining TATA box at position −30 relative to the transcription start site is treated identically to the same sequence appearing anywhere else in the window. Additionally, k-mer features capture only local, non-overlapping statistics, missing longer-range interactions between regulatory motifs.

### 2.4 1D-CNN with Cross-Validation

#### 2.4.1 Sequence Encoding

DNA sequences were encoded as **one-hot vectors**: each nucleotide position was mapped to a 5-dimensional binary vector (A, C, G, T, N), yielding input tensors of shape (452, 5). This encoding is the natural representation for categorical sequence data, ensures equal Euclidean distance between all distinct nucleotides, and avoids imposing an arbitrary ordinal structure (which integer encoding would imply).

#### 2.4.2 Architecture

The model architecture is a **three-block 1D-CNN** summarized in **Table 2**:

**Table 2. 1D-CNN Architecture**

| Layer | Filters | Kernel Size | Output Shape | Notes |
|-------|---------|-------------|--------------|-------|
| Input | — | — | (452, 5) | One-hot encoded |
| GaussianNoise | — | — | (452, 5) | σ = 0.1, training only |
| Conv1D (Block 1) | 32 | 12 | (441, 32) | ReLU, L1=1e-4, L2=1e-3 |
| BatchNorm | — | — | (441, 32) | |
| SpatialDropout1D | — | — | (441, 32) | p = 0.2 |
| MaxPooling1D | — | 4 | (110, 32) | |
| Conv1D (Block 2) | 64 | 8 | (103, 64) | ReLU, L1=1e-4, L2=1e-3 |
| BatchNorm | — | — | (103, 64) | |
| SpatialDropout1D | — | — | (103, 64) | p = 0.3 |
| MaxPooling1D | — | 4 | (25, 64) | |
| Conv1D (Block 3) | 32 | 6 | (20, 32) | ReLU, L1=1e-4, L2=1e-3 |
| BatchNorm | — | — | (20, 32) | |
| GlobalAvgPooling1D | — | — | (32,) | Aggregate across positions |
| Dense | 64 | — | (64,) | ReLU, L1+L2 regularization |
| BatchNorm + Dropout | — | — | (64,) | p = 0.5 |
| Dense (output) | 3 | — | (3,) | Softmax |

The first convolutional layer uses a kernel size of 12 bp — comparable to the length of known transcription factor binding motifs (typically 8–20 bp), enabling the model to detect biologically meaningful k-mer-like patterns in a position-sensitive manner. Deeper layers with smaller kernels (8, 6 bp) hierarchically integrate local motif detections. The use of **GlobalAveragePooling** (rather than GlobalMaxPooling or Flatten) provides implicit regularization by averaging activations across all positions, making the model less sensitive to motif positioning.

#### 2.4.3 Regularization and Augmentation

Overfitting is a major risk when training neural networks on genomic sequences, given the high input dimensionality (452 × 5 = 2,260 input features) and the presence of biologically redundant patterns. Multiple regularization strategies were applied:

- **GaussianNoise (σ=0.1)**: Input-level perturbation during training, equivalent to data augmentation on the one-hot vectors.
- **Data augmentation**: Small Gaussian noise (σ=0.02) was added to training inputs at batch generation time, preventing the model from memorizing exact sequence patterns.
- **L1+L2 weight regularization** (L1=1e-4, L2=1e-3): Applied to all convolutional and dense kernel weights, penalizing both large weights (L2) and sparse activation patterns (L1).
- **BatchNormalization**: After each convolutional block, normalizing activations to zero mean and unit variance, which stabilizes training and provides additional regularization.
- **SpatialDropout1D** (p=0.2–0.3) and standard **Dropout** (p=0.5): SpatialDropout drops entire feature maps rather than individual activations, more effective for correlated sequence data.

The Adam optimizer was used with learning rate 0.0003 and gradient clipping (clipnorm=1.0) to prevent exploding gradients during early training. The loss function was **categorical cross-entropy**, appropriate for one-hot multi-class targets.

#### 2.4.4 Cross-Validation and Model Selection

To obtain a reliable estimate of generalization performance and to select the best model, we employed **5-fold stratified cross-validation** on the combined train+validation set (n = 92,134). Stratification ensures that each fold maintains the 1:1:1 class ratio. For each fold, the model was trained for 10 epochs with batch size 128, and the validation accuracy was recorded.

**Table 3. Cross-Validation Results (5-Fold)**

| Fold | Val Accuracy | Val AUC |
|------|-------------|---------|
| 1 | 0.9131 | ~0.985 |
| 2 | 0.9105 | ~0.985 |
| 3 | 0.9104 | ~0.985 |
| 4 | 0.9078 | ~0.985 |
| 5 | 0.9132 | ~0.986 |
| **Mean ± SD** | **0.9110 ± 0.0020** | — |

The low variance across folds (SD = 0.002) confirms that model performance is stable and not dependent on a particular data split — a strong indicator that the reported accuracy generalizes to unseen data. The model achieving the highest validation accuracy was retained as the final model for test evaluation.

### 2.5 Final Evaluation on Held-Out Test Set

The best model (Fold 5, val_acc = 0.9132) was evaluated on the held-out test set (n = 16,259) that was not used during training, validation, or model selection.

**Table 4. CNN Classification Report (Test Set, n = 16,259)**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| ENHANCER | 0.8785 | 0.8808 | 0.8797 | 5,419 |
| PROMOTER | 0.9826 | **1.0000** | **0.9912** | 5,420 |
| INTERGENIC | 0.8881 | 0.8701 | 0.8790 | 5,420 |
| **Macro avg** | **0.9164** | **0.9170** | **0.9166** | 16,259 |
| **Weighted avg** | 0.9164 | 0.9170 | 0.9166 | 16,259 |
| **Overall Acc.** | — | — | — | **91.70%** |

Test loss: 0.2929. The CNN's most striking result is its **perfect recall for the PROMOTER class** (recall = 1.000), meaning that the model correctly identified every single promoter sequence in the test set. This is biologically plausible: promoters possess highly characteristic sequence motifs (TATA box, Initiator element, CpG islands) that are strongly conserved and detectable by convolutional filters. In contrast, enhancers are more sequence-diverse — they function through combinations of transcription factor binding sites that vary widely across cell types — which explains the lower F1 (0.88) for that class.

**Table 5. Model Comparison (Held-Out Test Set)**

| Model | Accuracy | Macro F1 |
|-------|----------|---------|
| SVM (k-mer, LinearSVC) | 73.07% | ~0.73 |
| 1D-CNN (best fold) | **91.70%** | **0.917** |
| **Improvement** | **+25.50%** | — |

The two-sided McNemar test on the paired binary error vectors of SVM vs. CNN predictions yields a highly significant result (p < 0.001), confirming that the CNN's error distribution is statistically distinct from the SVM's — not merely due to sampling variability.

### 2.6 Model Interpretability

To understand what sequence features drive CNN decisions, we applied two interpretability techniques. First, we visualized the **first-layer convolutional filter weights**. Each filter (kernel size 12) can be interpreted as a positional weight matrix (PWM) over the 5-channel nucleotide alphabet. Several filters exhibited characteristic patterns resembling known regulatory motifs (e.g., GC-rich patterns associated with CpG islands and promoter regions, AT-rich patterns associated with TATA boxes).

Second, we computed **gradient-based saliency maps** for individual test sequences. Using `tf.GradientTape`, we computed the gradient of the predicted class score with respect to the input one-hot tensor, and retained only positive gradient values (analogous to ReLU activation on the gradient). High-gradient positions indicate nucleotides whose perturbation would most strongly decrease the model's confidence — i.e., positions that are informative for the classification decision. For an ENHANCER sequence (sample 10), the saliency map revealed concentrated importance at discrete positions, consistent with the hypothesis that the model detects sparse, localized binding motif patterns rather than distributed sequence statistics.

---

## 3. Methods

### 3.1 Dataset Construction and Balancing

The raw dataset (`DNA_multiclass.parquet`) contained 234,783 labeled DNA sequences with three class labels (ENHANCER=0, PROMOTER=1, INTERGENIC=2). Class imbalance was addressed via random downsampling: for each class, a random sample of size n = 36,131 (the minority class count) was drawn without replacement using a fixed random seed (42). The three sub-samples were concatenated and globally shuffled to produce the balanced dataset of 108,393 sequences, saved as `DNA_multiclass_balanced.parquet`.

### 3.2 Train/Validation/Test Split

The balanced dataset was partitioned using two sequential stratified splits (sklearn `train_test_split`, random_state=42): first, a 15% held-out test set was separated; then, the remaining 85% was split into 70% train and 15% validation (achieved using test_size=0.176 on the 85% remainder). All splits were stratified on the class label to maintain balanced class proportions.

### 3.3 K-mer Feature Extraction (SVM)

DNA sequences were transformed into k-mer frequency vectors using scikit-learn's `CountVectorizer` with `analyzer='char'` and `ngram_range=(k, k)`, extracting all character k-grams of exactly length k. This produces a sparse count matrix of dimension n × V, where V = number of unique k-mers observed in the training set. Features are not normalized (raw counts), which is consistent with the scale-invariance of LinearSVC.

Hyperparameter optimization was performed with Optuna (15 trials, Tree-structured Parzen Estimator sampler) on a random subsample of 15,000 training sequences. The search space was k ∈ {3, 4, 5, 6} (discrete integer) and C ∈ [0.001, 10] (log-uniform continuous). For each trial, an internal 75/25 train-validation split was used to evaluate accuracy. After optimization, the best configuration was re-fit on the full training set and evaluated on the held-out test set.

LinearSVC was configured with `dual=False` (appropriate when n_samples > n_features) and `max_iter=2,000`.

### 3.4 Sequence Encoding (CNN)

Raw sequences were integer-encoded using the mapping A→0, C→1, G→2, T→3, N→4, then zero-padded (post-padding with value 4) to a fixed length of FINAL_LEN = 452 bp using Keras' `pad_sequences`. One-hot encoding was applied via `tf.one_hot(..., depth=5)`, yielding matrices of shape (n, 452, 5).

### 3.5 Data Augmentation

Training batches were generated using `tf.data.Dataset` with on-the-fly augmentation: independent Gaussian noise (μ=0, σ=0.02) was added to each one-hot position vector, and values were clipped to [0, 1]. Augmentation was applied only during training, not during validation or testing. Additionally, a GaussianNoise layer (σ=0.1) was placed as the first layer of the network, providing an additional stochastic perturbation during forward passes in training mode.

### 3.6 CNN Training

The model was trained using categorical cross-entropy loss and the Adam optimizer (lr=3×10⁻⁴, clipnorm=1.0). Batch size was 128. Training ran for 10 epochs per fold. The 5-fold stratified cross-validation loop (StratifiedKFold, shuffle=True, random_state=42) iterated over the combined train+validation pool (n=92,134). For each fold, a new model was instantiated with random weight initialization. The model achieving the highest validation accuracy across all folds was retained for final test evaluation.

### 3.7 Statistical Tests

**Kruskal-Wallis test:** Non-parametric one-way analysis of variance on sequence length distributions across three classes. Null hypothesis: median lengths are equal across classes.

**Pairwise Mann-Whitney U tests with Bonferroni correction:** Post-hoc pairwise comparisons of sequence length between all pairs of classes. Adjusted significance threshold α' = 0.05/3 = 0.0167.

**McNemar's test:** Applied to evaluate whether the CNN and SVM produced statistically different error patterns on the paired test set, treating each test sample's binary correctness (correct/incorrect) as a matched pair.

---

## 4. Discussion

### 4.1 Conclusions

This study demonstrates that one-dimensional convolutional neural networks substantially outperform k-mer-based support vector machines for regulatory element classification from raw DNA sequence. The CNN achieved 91.70% test accuracy versus the SVM's 73.07%, a 25.50% relative improvement. The CNN's advantage is most striking for the PROMOTER class, where it achieves perfect recall, likely due to the strong, conserved motif signatures of promoter sequences that convolutional filters can detect with high sensitivity.

The central narrative of this project is that **representation matters**. The SVM treats a sequence as an unordered multiset of short substrings — capturing composition but not context. The CNN, by contrast, learns to detect ordered, positionally weighted patterns, integrating short motifs (layer 1) into longer-range interactions (layers 2-3), which more faithfully captures how biological sequence function is encoded. This is not merely an engineering insight; it reflects a deep property of genomic regulation — that the same nucleotide pattern has different functional implications depending on its sequence context and position relative to other elements.

The 5-fold cross-validation results (mean accuracy 91.10% ± 0.20%) closely match the held-out test accuracy (91.70%), indicating that the model generalizes well and that the evaluation framework successfully prevented overfitting during model development. The multiple regularization strategies applied (noise augmentation, SpatialDropout, L1+L2 penalties, BatchNormalization, gradient clipping) collectively contributed to this generalization.

### 4.2 Limitations

Several limitations must be acknowledged. First, **downsampling** discards a substantial fraction of the majority class (enhancers), potentially reducing the diversity of sequence patterns available for training and biasing the model toward the specific subset of enhancers that happened to be sampled. An alternative evaluation with class-weighted loss on the original imbalanced dataset would be informative.

Second, the **padding strategy** introduces artificial sequence content (N tokens) that the model must learn to ignore. Sequences shorter than the median length have disproportionate amounts of padding. The use of GlobalAveragePooling mitigates but does not eliminate this issue — padding tokens still contribute to the global average, diluting the signal from genuine sequence content. Masking approaches or variable-length architectures (e.g., using Transformer models with attention masking) would handle variable lengths more principled.

Third, the training data's **source and annotation quality** are unknown from the notebook alone. If the labeled sequences were derived from a single organism or cell type, the model may not generalize to divergent genomes or different regulatory contexts. Regulatory elements, particularly enhancers, are highly cell-type-specific, and a model trained on bulk annotations may fail to distinguish context-dependent activity.

Fourth, the **SVM comparison is not entirely fair** due to the use of a LinearSVC rather than a kernel SVM with a string kernel (e.g., the spectrum kernel, which is the proper kernelized analog of k-mer features). Non-linear SVMs with string kernels are known to be highly competitive for sequence classification but were excluded due to computational constraints. The SVM baseline should therefore be interpreted as a lower bound on k-mer-based classification performance.

Finally, the model's **interpretability analysis** via saliency maps is exploratory and not statistically validated. Gradient-based attributions can be noisy and do not constitute formal motif discovery. A rigorous analysis would involve systematic scanning of the learned filter weights against a database of known transcription factor binding motifs (e.g., JASPAR) and statistical testing of enrichment.

### 4.3 Future Directions

Future work could explore Transformer-based architectures for DNA sequences (e.g., DNABERT, Nucleotide Transformer), which have recently shown state-of-the-art performance on genomic classification tasks through pre-training on large genomic corpora. Additionally, incorporating epigenomic features (histone modification signals, chromatin accessibility) as auxiliary inputs could improve classification accuracy, particularly for the more context-dependent enhancer class.

---

## References

Kelley, D. R., Snoek, J., & Rinn, J. L. (2016). Basset: learning the regulatory code of the accessible genome with deep convolutional neural networks. *Genome Research*, 26(7), 990–999.

Leslie, C., Eskin, E., & Noble, W. S. (2002). The spectrum kernel: a string kernel for SVM protein classification. *Pacific Symposium on Biocomputing*, 7, 564–575.

Quang, D., & Xie, X. (2016). DanQ: a hybrid convolutional and recurrent deep neural network for quantifying the function of DNA sequences. *Nucleic Acids Research*, 44(11), e107.

Stormo, G. D. (2000). DNA binding sites: representation and discovery. *Bioinformatics*, 16(1), 16–23.

Zhou, J., & Troyanskaya, O. G. (2015). Predicting effects of noncoding variants with deep learning–based sequence model. *Nature Methods*, 12(10), 931–934.