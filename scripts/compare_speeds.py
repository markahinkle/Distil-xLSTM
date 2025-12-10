import time
import importlib
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to sys.path for src imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Specify which models to compare
models_compared = [
    "transformer_vanilla",
    "mamba",
    "lstm",
    "xlstm",
    # add more if needed
]

# Prompt and test settings
prompt = "The quick brown fox jumps over the lazy dog. What happened next?"
num_runs = 12 # because we delete the first two warmup runs!!!
max_new_tokens = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logarithmic = False  # Set to False for linear scale
dtype = torch.float32

# Load teacher model (always needed)
from src.models.teacher import load_teacher_model
teacher = load_teacher_model(dtype=dtype)
tokenizer = teacher.tokenizer

# Model loading functions
def load_transformer_qwen():
    from src.models.transformer_student import DistilQwenTransformerStudent
    return DistilQwenTransformerStudent.from_teacher(teacher, dtype=dtype)

def load_transformer_vanilla():
    from src.models.transformer_student import DistilVanillaTransformerStudent
    return DistilVanillaTransformerStudent.from_teacher(teacher, dtype=dtype)

def load_mamba():
    from src.models import DistilMambaStudent, build_mamba_student_spec_from_teacher
    spec = build_mamba_student_spec_from_teacher(teacher, context_length=64)
    return DistilMambaStudent.from_teacher(teacher, spec=spec, dtype=dtype)

def load_lstm():
    from src.models import DistilLSTMStudent, build_lstm_student_spec_from_teacher
    spec = build_lstm_student_spec_from_teacher(teacher, context_length=64)
    return DistilLSTMStudent.from_teacher(teacher, spec=spec, dtype=dtype)

def load_xlstm():
    from src.models import DistilXLSTMStudent, build_student_spec_from_teacher
    spec = build_student_spec_from_teacher(teacher, context_length=64)
    return DistilXLSTMStudent.from_teacher(teacher, spec=spec, dtype=dtype)

model_loaders = {
    "transformer_qwen": load_transformer_qwen,
    "transformer_vanilla": load_transformer_vanilla,
    "mamba": load_mamba,
    "lstm": load_lstm,
    "xlstm": load_xlstm,
}

# Inference timing function
def measure_inference_speed(model, tokenizer, prompt, max_new_tokens, num_runs):
    times = []
    model_device = next(model.parameters()).device
    # Warmup run (not timed)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    with torch.inference_mode():
        if hasattr(model, "generate"):
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        else:
            _ = model(inputs["input_ids"])
    # Timed runs
    for run_idx in range(num_runs):
        torch.cuda.empty_cache()
        start = time.time()
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        with torch.inference_mode():
            if hasattr(model, "generate"):
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
                # Calculate number of generated tokens
                if hasattr(output, "shape"):
                    num_generated = output.shape[-1] - inputs["input_ids"].shape[-1]
                else:
                    num_generated = max_new_tokens
            else:
                output = model(inputs["input_ids"])
                num_generated = output.shape[-1] if hasattr(output, "shape") else max_new_tokens
        end = time.time()
        elapsed = end - start
        # Divide by number of generated tokens for fair comparison
        time_per_token = elapsed / max_new_tokens
        print(f"Run {run_idx+1}/{num_runs}: {elapsed:.4f}s ({time_per_token:.6f}s/token), tokens generated: {num_generated}")
        times.append(time_per_token)
    # Remove the first timed run from statistics
    return times[2:]

results = {}
for model_name in models_compared:
    print(f"Loading model: {model_name}")
    loader = model_loaders.get(model_name)
    if loader is None:
        print(f"Model loader for '{model_name}' not found, skipping.")
        continue
    model = loader()
    model.eval()
    # Attach tokenizer if needed
    if not hasattr(model, "tokenizer"):
        model.tokenizer = tokenizer
    print(f"Measuring inference speed for: {model_name}")
    times = measure_inference_speed(model, tokenizer, prompt, max_new_tokens, num_runs)
    results[model_name] = times
    print(f"Median time: {np.median(times):.4f}s")

# Plot boxplot
plt.figure(figsize=(10, 6))
data = [results[m] for m in models_compared if m in results]
plt.boxplot(data, labels=[m for m in models_compared if m in results])
plt.ylabel("Inference Time per Token (seconds)")
plt.title("Student Model Inference Speed per Token Comparison")
plt.grid(True)
if logarithmic:
    plt.yscale("log")
plt.tight_layout()
plot_path = Path(__file__).parent / "inference_speed_comparison.png"
plt.savefig(plot_path)
print(f"Saved boxplot to {plot_path}")
