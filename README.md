# Distil-xLSTM Reproduction

This repository reproduces the key ideas from **“Distil-xLSTM: Learning Attention Mechanisms through Recurrent Structures”** (arXiv: [2503.18565](doc/2503.18565v1.pdf)).  
We distil a Qwen2.5-1.5B transformer teacher into an xLSTM student using the FineWeb dataset, following the architecture and ∆-distillation strategy described in the paper.

---

## Overview

- **Teacher**: Qwen/Qwen2.5-1.5B transformer (outputs logits + hidden states).  
- **Student**: xLSTM stack (alternating sLSTM/mLSTM blocks) with weight reuse for embeddings & LM head.  
- **Training data**: FineWeb streaming dataset (sample-10BT by default).  
- **Distillation**: Cross-entropy + KL + Frobenius alignment with time-varying α (teacher weight) and temperature.

All configuration values live in `config/config.yaml`. The trainer writes metrics, checkpoints, TensorBoard logs, and generated reports to the directory passed via `--output-dir` (defaults to `artifacts/latest`).

---

## Environment Setup

### Option 1 — Use `uv` (recommended)
[`uv`](https://github.com/astral-sh/uv) provides fast dependency resolution and isolated environments.

```bash
# Install uv if needed (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create env & install dependencies
cd distil-xlstm
uv sync

# Activate the virtualenv for the current shell
source .venv/bin/activate
```

### Option 2 — Standard Python + `requirements.txt`

If you cannot install `uv`, a compatible `requirements.txt` is generated via `uv pip compile`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

---

## Key Scripts

Each script lives under `scripts/` and can be run with `uv run python` or from an activated virtual environment.

| Script | Purpose |
| --- | --- |
| `train.py` | End-to-end distillation run. Handles streaming data, optimizer/scheduler, ∆-distillation, checkpointing, and report generation. Use `--output-dir` to select where artifacts go and `--report` to produce plots/markdown from the recorded metrics. |
| `test_teacher.py` | Loads the Qwen2.5-1.5B teacher and optionally runs a short greedy generation to confirm the model weights are accessible on your hardware. |
| `test_xlstm_teacher.py` | Loads an open-source xLSTM checkpoint (PatrickHaller/xlstm_wikipedia_110M_500M by default) to validate the xLSTM inference stack. |
| `test_student.py` | Builds the student from teacher weights, reports parameter counts, and runs a dummy forward pass to ensure the architecture matches the teacher shape. |
| `test_distillation_loss.py` | Exercises the ∆-distillation loss (CE + KL + Frobenius) and prints component values, useful as a unit test for numerical stability. |
| `test_fineweb_stream.py` | Streams a few samples from FineWeb, tokenizes with the teacher tokenizer, and prints sample token IDs—verifies the data pipeline. |

All scripts share the same configuration base (e.g., choice of dataset subset, sequence length). See `src/distillation/trainer.py` for implementation details and logging behaviour.

---

## General Workflow

1. **Validate components**:  
   - `scripts/test_teacher.py --skip-generation`  
   - `scripts/test_student.py`  
   - `scripts/test_fineweb_stream.py`

2. **Run a short training smoke test** (e.g., 100 steps) and watch the console for finite losses.  
   Example:
   ```bash
   uv run python scripts/train.py \
       --config config/config.yaml \
       --output-dir artifacts/test_run \
       --report
   ```
   This produces:
   - `artifacts/test_run/metrics.jsonl` — per-step metrics.
   - `artifacts/test_run/checkpoints/` — checkpoints (if `save_every > 0`).
   - `artifacts/test_run/tensorboard/` — TensorBoard logs.
   - `artifacts/test_run/loss_components.png`, `schedule.png`, `optimization_diagnostics.png`, and `report.md` — generated plots + summary.

3. **Inspect artifacts**: open the Markdown report or load metrics into a notebook. NaN guards will stop training early and dump problematic tensors in `artifacts/*/debug/` for inspection.

4. **Scale up**: once stable, adjust `steps_per_epoch`, `num_epochs`, Frobenius weight `beta`, etc., to match the experiments in the paper.

---

## Repository Structure

```
src/
  data/             # FineWeb streaming + tokenization utilities
  distillation/     # Loss, trainer, configs
  models/           # Teacher/xLSTM student loader utilities
  utils/            # Configuration loader, metrics logger, report generator
scripts/            # CLI entry points described above
config/config.yaml  # Main experiment configuration
artifacts/          # Default output directory for runs (ignored by git)
doc/                # Paper PDF and implementation guide
```

---

## References

- [2503.18565v1.pdf](doc/2503.18565v1.pdf) — official Distil-xLSTM paper.
- `doc/distil_xlstm_implementation_guide.md` — working implementation guide summarizing assumptions & reproduction details.
- FineWeb dataset: <https://huggingface.co/datasets/HuggingFaceFW/fineweb>
- Qwen2.5 models: <https://huggingface.co/Qwen>

---

## Notes

- Training in float16 on Apple MPS can lead to numerical overflow. This is currently being debugged and explored
- All scripts respect `--output-dir`, so you can keep multiple experiment logs side-by-side (e.g. `artifacts/phase5`, `artifacts/ablation_beta01`, etc.).
- Avoid running `scripts/train.py` concurrently with the same output directory; the trainer clears previously generated artifacts to ensure reports reflect the latest run.

Happy distilling!

