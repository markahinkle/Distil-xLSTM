# DistilVanillaTransformerStudent

## Description

`DistilVanillaTransformerStudent` is a custom decoder-only transformer model implemented in PyTorch. It is designed for use as a student model in distillation experiments, mimicking the architecture and behavior of GPT-style models.

### Key Features

- **Decoder-only architecture:** Processes input sequences in parallel and supports autoregressive generation.
- **Configurable depth and width:** Number of layers, hidden dimension, and attention heads can be set to match or scale down from a teacher model.
- **Efficient implementation:** Uses PyTorch's built-in `nn.MultiheadAttention` and feedforward layers for speed and simplicity.
- **Autoregressive generation:** Implements a `generate` method for token-by-token text generation, similar to HuggingFace models.
- **Distillation compatibility:** Returns logits and hidden states in a HuggingFace-like output format for easy integration with distillation pipelines.

### Usage

- **Training:** Use the `forward` method to obtain logits for all tokens in the input sequence in parallel.
- **Inference:** Use the `generate` method to produce new tokens autoregressively, one at a time, given an initial prompt.

### Example

```python
from src.models.transformer_student import DistilVanillaTransformerStudent

# Instantiate from teacher resources
student = DistilVanillaTransformerStudent.from_teacher(teacher)

# Parallel forward pass (training)
outputs = student(input_ids)

# Autoregressive generation (inference)
generated_ids = student.generate(input_ids, max_new_tokens=16)
```

### Notes

- The model is intended for research and experimentation, not for production deployment.
- The architecture and parameter sizes can be adjusted to balance speed and accuracy.
- For fair speed comparisons, use the `generate` method to match HuggingFace's autoregressive behavior.

