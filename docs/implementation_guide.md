# ViT-LENS: Comprehensive Implementation Guide

## 1. Theoretical Foundations & Research Objective

### 1.1 The Vision Transformer Residual Stream
In Vision Transformers (ViT-B/16), an image $I \in \mathbb{R}^{3 \times 224 \times 224}$ is partitioned into $N = 196$ non-overlapping patches of spatial size $16 \times 16$. Along with a prepended class token $[CLS]$, the sequence of $N+1 = 197$ tokens passes through 12 stacked Transformer encoder blocks. 

At any intermediate block $l$ (by default Block 7, 0-indexed), the token activations reside in the **residual stream** $\mathbf{x} \in \mathbb{R}^{d_{model}}$, where $d_{model} = 768$. Standard ViT representations are polysemantic: individual neurons do not correspond to clean, human-interpretable visual concepts, but rather superpositions of multiple visual features.

### 1.2 TopK Sparse Autoencoders (TopK SAE)
To disentangle these polysemantic representations into sparse, potentially monosemantic visual directions, ViT-LENS trains a TopK Sparse Autoencoder on the patch-level residual activations.

#### Mathematical Formulation
Given an intermediate patch activation $\mathbf{x} \in \mathbb{R}^{d_{in}}$ ($d_{in} = 768$):

1. **Centering**:
   $$\mathbf{x}_{centered} = \mathbf{x} - \mathbf{b}_{dec}$$
2. **Encoding**:
   $$\mathbf{z}_{pre} = \text{ReLU}\left(\mathbf{x}_{centered} \mathbf{W}_{enc} + \mathbf{b}_{enc}\right) \in \mathbb{R}^{d_{sae}}$$
   where $d_{sae} = 3072$ (expansion factor $4\times$).
3. **Hard TopK Sparsity**:
   $$\mathbf{z}_{topk} = \text{TopK}(\mathbf{z}_{pre}, k)$$
   Only the top $k = 32$ highest activation values are retained; all other elements are strictly set to zero. This enforces an exact $L_0$ sparsity bound without requiring tuning of an $L_1$ penalty coefficient.
4. **Decoding & Reconstruction**:
   $$\hat{\mathbf{x}} = \mathbf{z}_{topk} \mathbf{W}_{dec} + \mathbf{b}_{dec}$$
   where each decoder feature direction $\mathbf{w}_{dec, i} \in \mathbb{R}^{d_{in}}$ is constrained to have unit $L_2$ norm ($\|\mathbf{w}_{dec, i}\|_2 = 1.0$).

---

## 2. Residual-Preserving Causal Intervention

The defining technical contribution of ViT-LENS is the **residual-preserving causal intervention**. Rather than perturbing output logits or retraining models, we perform surgical interventions inside the forward pass of the frozen Vision Transformer.

### 2.1 The Mathematical Formulation
Let the original ViT activation be $\mathbf{x}$, which can be decomposed into the SAE reconstruction and the SAE residual:
$$\mathbf{x} = \hat{\mathbf{x}} + \mathbf{r} = \left(\sum_{j=1}^{d_{sae}} f_j \mathbf{w}_{dec, j} + \mathbf{b}_{dec}\right) + \mathbf{r}$$
where $\mathbf{r} = \mathbf{x} - \hat{\mathbf{x}}$ is the unexplained residual vector.

When intervening on a single target SAE feature $i$ with edited activation $f'_i = \alpha \cdot f_i$ (where $\alpha = 0.0$ for suppression and $\alpha = 2.0$ for amplification):
$$\mathbf{x}_{modified} = \mathbf{x} + (f'_i - f_i) \mathbf{w}_{dec, i}$$

### 2.2 Mathematical Guarantees
1. **Residual Invariance**:
   $$\mathbf{x}_{modified} - \hat{\mathbf{x}}_{modified} = (\mathbf{x} + \Delta f_i \mathbf{w}_{dec, i}) - (\hat{\mathbf{x}} + \Delta f_i \mathbf{w}_{dec, i}) = \mathbf{x} - \hat{\mathbf{x}} = \mathbf{r}$$
   The unexplained residual information is preserved identically.
2. **Strict Identity**:
   At $\alpha = 1.0$, $f'_i = f_i \implies \Delta f_i = 0 \implies \mathbf{x}_{modified} = \mathbf{x}$. The modified forward pass yields the exact original model output within numerical machine precision ($< 10^{-7}$).
3. **Subspace Specificity**:
   The intervention modifies only the 1D subspace spanned by the decoder direction $\mathbf{w}_{dec, i}$, leaving all orthogonal directions unaltered.

---

## 3. Scientific Safeguards and Control Experiments

To guard against misleading conclusions and confirm true causal importance, ViT-LENS enforces:

1. **Random-Feature Controls**:
   Every target feature intervention is benchmarked against 5–10 unrelated control SAE features with comparable activation frequencies.
2. **Cohort Evaluation**:
   Effects are measured across a test cohort of images rather than single cherry-picked examples.
3. **Hypothesis Framework**:
   All assigned semantic labels are explicitly treated as scientific *hypotheses*, not definitive ground truth.

---

## 4. Google Colab & Low-RAM Execution Architecture

ViT-LENS is engineered specifically for Google Colab GPU execution with Google Drive persistence, taking into account low-RAM local machines:

- **Chunked Activation Streaming**: ViT activations are written into discrete chunks (`chunk_*.pt`) of 100 images each, preventing out-of-memory errors.
- **Resumable Extraction & Training**: State checkpoints save the epoch, global step, optimizer state, and loss history.
- **Device-Agnostic Execution**: Automatically switches between CUDA, Apple MPS, and CPU.

