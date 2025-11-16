"""Utilities for working with publicly available xLSTM checkpoints."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .teacher import infer_runtime_device

LOGGER = logging.getLogger(__name__)

XLSTM_MODEL_ID = "PatrickHaller/xlstm_wikipedia_110M_500M"
# XLSTM_MODEL_ID = "NX-AI/xLSTM-7b"

@dataclass
class XLSTMTeacherResources:
    """Container holding the xLSTM teacher model, tokenizer, and runtime metadata."""

    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: torch.device
    dtype: torch.dtype


def load_xlstm_teacher(
    model_id: str = XLSTM_MODEL_ID,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    cache_dir: Optional[Path] = None,
) -> XLSTMTeacherResources:
    """Download and load a pretrained xLSTM checkpoint via Hugging Face."""

    runtime_device, runtime_dtype = infer_runtime_device()
    device = device or runtime_device
    if dtype is None:
        dtype = torch.float16 if device.type == "cuda" else torch.float32

    LOGGER.info("Loading xLSTM model %s on %s (%s)", model_id, device, dtype)

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
        "cache_dir": str(cache_dir) if cache_dir else None,
    }
    if device.type == "cpu":
        model_kwargs["device_map"] = None
    else:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    if device.type in {"cpu", "mps"}:
        model.to(device)

    xlstm_inner_cfg = getattr(model.config, "_xlstm_config", None)
    if isinstance(xlstm_inner_cfg, dict):
        num_layers = xlstm_inner_cfg.get("num_blocks")
        hidden_size = xlstm_inner_cfg.get("embedding_dim")
        if num_layers is not None:
            model.config.num_hidden_layers = num_layers
            if not hasattr(model.generation_config, "num_hidden_layers"):
                model.generation_config.num_hidden_layers = num_layers
        if hidden_size is not None:
            model.config.hidden_size = hidden_size
            if not hasattr(model.generation_config, "hidden_size"):
                model.generation_config.hidden_size = hidden_size

    model.generation_config.use_cache = False

    model.eval()

    return XLSTMTeacherResources(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
    )


def run_xlstm_smoke_test(
    resources: XLSTMTeacherResources,
    prompt: str = "Hello from Distil-xLSTM!",
    max_new_tokens: int = 8,
) -> str:
    """Generate a short sample to verify the xLSTM teacher loads correctly."""

    tokenizer = resources.tokenizer
    model = resources.model
    device = resources.device

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)

    generation_config = model.generation_config
    if generation_config.pad_token_id is None and tokenizer.pad_token_id is not None:
        generation_config.pad_token_id = tokenizer.pad_token_id
    if generation_config.eos_token_id is None and tokenizer.eos_token_id is not None:
        generation_config.eos_token_id = tokenizer.eos_token_id

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text
