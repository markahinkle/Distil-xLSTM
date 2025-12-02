from __future__ import annotations
import torch.nn as nn

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

TRANSFORMER_STUDENT_MODEL_ID = "Qwen/Qwen1.5-0.5B"


# Qwen transformer student (randomly initialized)
class DistilQwenTransformerStudent(nn.Module):
    """Qwen1.5-0.5B transformer student, randomly initialized."""

    @classmethod
    def from_teacher(cls, teacher, dtype=None):
        device = teacher.device
        dtype = dtype or torch.float32
        tokenizer = AutoTokenizer.from_pretrained(
            TRANSFORMER_STUDENT_MODEL_ID,
            use_fast=True,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            TRANSFORMER_STUDENT_MODEL_ID,
            trust_remote_code=True,
        )
        model = model.to(device=device, dtype=dtype)

        # Reinitialize all weights
        def _init_weights(module):
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
            elif hasattr(module, "weight") and module.weight is not None:
                if module.weight.dim() > 1:
                    nn.init.xavier_uniform_(module.weight)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

        model.apply(_init_weights)
        model.eval()
        model.tokenizer = tokenizer
        return model


# Vanilla transformer block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * dim, dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)
        return x


# Vanilla transformer student
class DistilVanillaTransformerStudent(nn.Module):
    """Custom vanilla transformer student (~0.5B params, best practices)."""

    def __init__(
        self,
        vocab_size=151936,
        num_layers=12,
        hidden_dim=1536,
        num_heads=12,
        max_length=512,
        dropout=0.1,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_length, hidden_dim))
        self.layers = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.max_length = max_length

    @classmethod
    def from_teacher(cls, teacher, dtype=None):
        config = teacher.model.config
        vocab_size = getattr(config, "vocab_size", 151936)
        num_layers = min(getattr(config, "num_hidden_layers", 12), 12)
        hidden_dim = getattr(config, "hidden_size", 1536)
        num_heads = getattr(config, "num_attention_heads", 12)
        max_length = getattr(config, "max_length", 512)
        dropout = 0.1
        device = teacher.device
        dtype = dtype or torch.float32
        model = cls(
            vocab_size=vocab_size,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            max_length=max_length,
            dropout=dropout,
            device=device,
            dtype=dtype,
        ).to(device=device, dtype=dtype)
        return model

    def forward(self, input_ids, attention_mask=None, return_hidden_states=False):
        x = self.embedding(input_ids)
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len]
        hidden_states = [x] if return_hidden_states else None
        for layer in self.layers:
            x = layer(x, attn_mask=attention_mask)
            if return_hidden_states:
                hidden_states.append(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return {
            "logits": logits,
            "hidden_states": hidden_states if return_hidden_states else None,
        }


"""Student Transformer model (Qwen1.5-0.5B architecture, zero-initialized) for distillation."""


@dataclass
class TransformerStudentResources:
    """Container holding the student transformer model, tokenizer, and runtime metadata."""

    model: PreTrainedModel
    tokenizer: AutoTokenizer
    device: torch.device
    dtype: torch.dtype


def load_transformer_student(
    model_id: str = TRANSFORMER_STUDENT_MODEL_ID,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    cache_dir: Optional[str] = None,
) -> TransformerStudentResources:
    """Load the Qwen1.5-0.5B architecture and zero-initialize all weights."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    model = model.to(device=device, dtype=dtype)

    # Reinitialize all weights (train from scratch)
    def _init_weights(module):
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
        elif hasattr(module, "weight") and module.weight is not None:
            if module.weight.dim() > 1:
                torch.nn.init.xavier_uniform_(module.weight)
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if hasattr(module, "bias") and module.bias is not None:
            torch.nn.init.constant_(module.bias, 0.0)

    model.apply(_init_weights)

    model.eval()

    return TransformerStudentResources(
        model=model, tokenizer=tokenizer, device=device, dtype=dtype
    )


def run_transformer_student_smoke_test(
    resources: TransformerStudentResources,
    prompt: str = "Hello from Distil-xLSTM!",
    max_new_tokens: int = 8,
) -> str:
    """Generate a short sample to verify the student loads correctly (output will be degenerate due to zero weights)."""
    tokenizer = resources.tokenizer
    model = resources.model
    device = resources.device

    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}

    generation_config = model.generation_config
    if generation_config.pad_token_id is None and tokenizer.pad_token_id is not None:
        generation_config.pad_token_id = tokenizer.pad_token_id
    if generation_config.eos_token_id is None and tokenizer.eos_token_id is not None:
        generation_config.eos_token_id = tokenizer.eos_token_id

    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text
