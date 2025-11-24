"""Student Transformer model (Qwen1.5-0.5B architecture, zero-initialized) for distillation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

TRANSFORMER_STUDENT_MODEL_ID = "Qwen/Qwen1.5-0.5B"


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
