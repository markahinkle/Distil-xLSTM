"""Utilities for working with the Qwen2.5-1.5B teacher model."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)

TEACHER_MODEL_ID = "Qwen/Qwen2.5-1.5B"


@dataclass
class TeacherResources:
    """Container holding the teacher model, tokenizer, and runtime metadata."""

    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: torch.device
    dtype: torch.dtype


def infer_runtime_device() -> tuple[torch.device, torch.dtype]:
    """Infer the most suitable device and dtype for the local environment.

    Preference order is CUDA > Metal (MPS) > CPU. FP16 is used when the backend
    supports it; otherwise the model falls back to FP32.
    """

    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16

    return torch.device("cpu"), torch.float32


def load_teacher_model(
    model_id: str = TEACHER_MODEL_ID,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    cache_dir: Optional[Path] = None,
) -> TeacherResources:
    """Load the Qwen teacher model and tokenizer with sensible defaults.

    Parameters
    ----------
    model_id:
        Hugging Face model identifier.
    device:
        Optional explicit device override.
    dtype:
        Optional explicit dtype override.
    cache_dir:
        Optional path to cache downloads. Defaults to the HF cache directory.
    """

    runtime_device, runtime_dtype = infer_runtime_device()
    device = device or runtime_device
    dtype = dtype or runtime_dtype

    LOGGER.info("Loading teacher model %s on %s (%s)", model_id, device, dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=True,
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "dtype": dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "cache_dir": str(cache_dir) if cache_dir else None,
    }

    if device.type == "cpu":
        model_kwargs["device_map"] = None
    else:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    if device.type in {"cpu", "mps"}:
        model.to(device)

    model.eval()

    return TeacherResources(model=model, tokenizer=tokenizer, device=device, dtype=dtype)


def run_teacher_smoke_test(
    resources: TeacherResources,
    prompt: str = "Hello from Distil-xLSTM!",
    max_new_tokens: int = 8,
) -> str:
    """Generate a short sample to verify the teacher loads correctly."""

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
    LOGGER.info("Teacher smoke test output: %s", text)
    return text
