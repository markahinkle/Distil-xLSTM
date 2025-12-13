# Feature-Based Alignment for Cross-Architecture Knowledge Distillation

This document explains the hidden state projection module and its various alignment loss functions for knowledge distillation across different neural network architectures.

## Overview

When distilling knowledge from a Transformer teacher to a recurrent student (LSTM, xLSTM, Mamba), the hidden state representations exist in fundamentally different spaces:

- **Transformer**: Attention-based contextual embeddings where each position attends to all others
- **Recurrent Models**: Sequential state representations where information flows through time

Direct comparison (e.g., Frobenius norm) between these representations is often ineffective because:
1. Different architectures encode the same semantic information differently
2. Averaging all teacher layers loses hierarchical structure
3. Magnitude differences can dominate the loss

**Solution**: A learned projection layer maps student hidden states to teacher space, enabling meaningful alignment.

---

## Architecture

```
Student Hidden States          Teacher Hidden States
     (B, S, d_s)                   (L, B, S, d_t)
         │                              │
         ▼                              ▼
    ┌─────────┐                  ┌─────────────┐
    │ Dropout │                  │ Layer       │
    └────┬────┘                  │ Selection   │
         │                       └──────┬──────┘
         ▼                              │
    ┌─────────┐                         ▼
    │ Linear  │                    (B, S, d_t)
    │ (d_s→d_t)                         │
    └────┬────┘                         │
         │                              │
         ▼                              │
    ┌─────────┐                         │
    │LayerNorm│                         │
    └────┬────┘                         │
         │                              │
         ▼                              ▼
    (B, S, d_t)                    (B, S, d_t)
         │                              │
         └──────────┬───────────────────┘
                    │
                    ▼
              ┌───────────┐
              │ Alignment │
              │   Loss    │
              └───────────┘
```

Where:
- `B` = batch size
- `S` = sequence length  
- `d_s` = student hidden dimension
- `d_t` = teacher hidden dimension
- `L` = number of teacher layers

---

## Teacher Layer Selection Strategies

### 1. Last N Layers (`last`)

Uses only the final N transformer layers, which typically encode higher-level semantic features.

```python
teacher_target = mean(teacher_hidden[-N:])
```

**Rationale**: Later layers contain more task-relevant, abstract representations.

### 2. Uniform Sampling (`uniform`)

Evenly samples N layers across the entire depth of the teacher.

```python
indices = [int(i * L / N) for i in range(N)]
teacher_target = mean([teacher_hidden[i] for i in indices])
```

**Rationale**: Captures both low-level syntax (early layers) and high-level semantics (late layers).

### 3. Learned Weights (`weighted`)

Learns a softmax-weighted combination of all layers.

```python
weights = softmax(layer_weights)  # Learnable parameter
teacher_target = sum(w_i * teacher_hidden[i] for i in range(L))
```

**Rationale**: Allows the model to discover which layers are most useful for the student.

---

## Alignment Loss Functions

### 1. Cosine Similarity Loss

**Equation:**

$$\mathcal{L}_{\text{cosine}} = 1 - \frac{1}{|M|} \sum_{(i,j) \in M} \frac{\mathbf{h}_s^{(i,j)} \cdot \mathbf{h}_t^{(i,j)}}{\|\mathbf{h}_s^{(i,j)}\| \|\mathbf{h}_t^{(i,j)}\|}$$

Where:
- $\mathbf{h}_s^{(i,j)}$ = projected student hidden state at batch $i$, position $j$
- $\mathbf{h}_t^{(i,j)}$ = teacher hidden state target
- $M$ = set of valid (non-masked) positions

**Properties:**
- ✓ Invariant to magnitude (only direction matters)
- ✓ Bounded in [0, 2]
- ✓ Robust to scale differences between architectures
- ✗ Ignores magnitude information

**Best for:** Cross-architecture distillation where representation scales differ.

---

### 2. Mean Squared Error (MSE) Loss

**Equation:**

$$\mathcal{L}_{\text{MSE}} = \frac{1}{|M| \cdot d} \sum_{(i,j) \in M} \|\mathbf{h}_s^{(i,j)} - \mathbf{h}_t^{(i,j)}\|_2^2$$

Where $d$ is the hidden dimension.

**Properties:**
- ✓ Simple and well-understood
- ✓ Preserves magnitude information
- ✗ Sensitive to outliers
- ✗ Scale-dependent

**Best for:** Same-architecture distillation or when magnitude matters.

---

### 3. Smooth L1 (Huber) Loss

**Equation:**

$$\mathcal{L}_{\text{smooth-L1}} = \frac{1}{|M| \cdot d} \sum_{(i,j) \in M} \sum_{k=1}^{d} \text{SmoothL1}(h_{s,k}^{(i,j)} - h_{t,k}^{(i,j)})$$

Where:

$$\text{SmoothL1}(x) = \begin{cases} 0.5x^2 & \text{if } |x| < 1 \\ |x| - 0.5 & \text{otherwise} \end{cases}$$

**Properties:**
- ✓ Robust to outliers (linear penalty for large errors)
- ✓ Smooth gradient near zero
- ✓ Good balance between MSE and L1
- ✗ Has a fixed transition point at 1

**Best for:** When training has high-gradient outliers.

---

### 4. Centered Kernel Alignment (CKA) Loss

**Equation:**

$$\mathcal{L}_{\text{CKA}} = 1 - \frac{\text{HSIC}(K_s, K_t)}{\sqrt{\text{HSIC}(K_s, K_s) \cdot \text{HSIC}(K_t, K_t)}}$$

Where:
- $K_s = X_s X_s^T$ (student Gram matrix after centering)
- $K_t = X_t X_t^T$ (teacher Gram matrix after centering)
- HSIC is the Hilbert-Schmidt Independence Criterion

For linear kernels with centered features:

$$\text{HSIC}(K, L) = \|K^T L\|_F^2$$

**Properties:**
- ✓ **Invariant to orthogonal transformations** (rotations, reflections)
- ✓ **Invariant to isotropic scaling**
- ✓ Measures **structural similarity** not point-wise alignment
- ✓ Excellent for cross-architecture comparison
- ✗ More computationally expensive
- ✗ Requires sufficient batch size for stable estimates

**Best for:** Comparing representations from fundamentally different architectures.

**Reference:** Kornblith et al., "Similarity of Neural Network Representations Revisited", ICML 2019.

---

## Combined Distillation Loss

The total distillation loss combines three components:

$$\mathcal{L}_{\text{total}} = \frac{(1-\alpha-\beta) \cdot \mathcal{L}_{\text{CE}} + \alpha \cdot \mathcal{L}_{\text{KL}} + \beta \cdot \mathcal{L}_{\text{align}}}{\text{weight\_sum}}$$

Where:
- $\mathcal{L}_{\text{CE}}$ = Cross-entropy with ground truth labels
- $\mathcal{L}_{\text{KL}}$ = KL divergence between teacher/student logits
- $\mathcal{L}_{\text{align}}$ = Hidden state alignment (Frobenius or Projection-based)
- $\alpha$ = KL weight (annealed during training)
- $\beta$ = Alignment weight (typically 0.05-0.1)

---

## Configuration

Enable projection-based alignment in your config:

```yaml
loss:
  use_frobenius: true
  use_projection: true
  projection_loss_type: cosine      # cosine, mse, smooth_l1, or cka
  projection_layer_strategy: last   # last, uniform, or weighted
  projection_num_teacher_layers: 4  # N layers to use
  projection_normalize: true        # L2 normalize before loss
  projection_dropout: 0.1
  projection_use_layer_norm: true
  beta: 0.1                         # Alignment loss weight
```

---

## Monitoring

Key metrics to track during training:

| Metric | Good Value | Interpretation |
|--------|-----------|----------------|
| `projection_cosine_sim` | > 0.3 | Student learning teacher's representation direction |
| `projection_alignment_ratio` | > 0.55 | Fraction of dimensions with same sign |
| `frobenius` (alignment loss) | Decreasing | Representations converging |

---

## Research References

1. **FitNets** (Romero et al., 2015) - Introduced hint layers with linear projections
2. **TinyBERT** (Jiao et al., 2020) - MSE loss with learned projections for BERT
3. **DistilBERT** (Sanh et al., 2019) - Cosine embedding loss for representation matching
4. **MiniLM** (Wang et al., 2020) - Attention transfer with dimension alignment
5. **PKD** (Sun et al., 2019) - Patient Knowledge Distillation with layer mapping
6. **CKA** (Kornblith et al., 2019) - Centered Kernel Alignment for representation similarity
