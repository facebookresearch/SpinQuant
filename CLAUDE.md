# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SpinQuant is a fork of Meta's SpinQuant LLM quantization framework, adapted for integration with CEVA's LiteML. It applies learned orthogonal rotations to minimize quantization error, then exports the result to LiteML's state dict format.

**Supported models:** Llama 2/3/3.1/3.2, Qwen 2.5, DeepSeek R1-Distill-Qwen variants.

## Environment Setup

SpinQuant uses Python 3.9 / CUDA 11.8 — separate from ailabs_liteml's Python 3.10 / CUDA 12.6 environment.

```bash
conda create -n SpinQuant python=3.9
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirement.txt
```

**`fast-hadamard-transform` requires a manual patch before install:**
```bash
git clone --depth 1 https://github.com/Dao-AILab/fast-hadamard-transform.git
# In fast-hadamard-transform/setup.py ~line 217: replace os.rename() with shutil.move() and add import shutil at the top
pip install --no-build-isolation ./fast-hadamard-transform
```

## Key Scripts

### ptq.py — Post-Training Quantization

Main entry point. Loads a model, optionally applies rotations, quantizes weights/activations/KV-cache with GPTQ or RTN, evaluates perplexity on WikiText2, and saves the quantized state dict.

```bash
python ptq.py \
    --input_model meta-llama/Llama-2-7b-hf \
    --do_eval True \
    --fp16 True \
    --w_bits 4 --a_bits 8 --k_bits 8 --v_bits 8 \
    --w_groupsize 128 --a_groupsize 128 --k_groupsize 128 --v_groupsize 128 \
    --rotate --w_clip \
    --save_qmodel_path "./saved_models/spinquant_gptq_group128.pth"
```

**Key flags:**

| Flag | Description |
|------|-------------|
| `--rotate` | Apply SpinQuant rotations (R1 global + R2 per-layer attention head) |
| `--optimized_rotation_path R.bin` | Use pre-trained rotation matrices instead of Hadamard |
| `--w_bits / --a_bits / --k_bits / --v_bits` | Bit-widths (4, 8, 16) for weights, activations, K/V cache |
| `--w_groupsize / --a_groupsize / --k_groupsize / --v_groupsize` | Group sizes (-1 = per-channel) |
| `--w_clip` | MSE-based weight clipping |
| `--w_rtn` | Use RTN instead of GPTQ for weight quantization |
| `--w_asym / --a_asym / --k_asym / --v_asym` | Asymmetric quantization |
| `--fp32_had` | Apply Hadamard in FP32 precision |
| `--access_token` | HuggingFace token for gated models |

### liteml_state_dict.py — Export to LiteML Format

Converts a SpinQuant checkpoint to the state dict format expected by LiteML's `RetrainerModel`. Key transformation: zero-points are converted from unsigned to signed domain (`convert_spinquant_zp_for_liteml()`).

```bash
# Basic export
python liteml_state_dict.py \
    --spinquant_path saved_models/spinquant_gptq_group128.pth \
    --liteml_path saved_models/liteml_spinquant_gptq_group128.pth \
    --group_size 128

# Fuse lm_head with final RMSNorm
python liteml_state_dict.py ... --fuse_lm_head --group_size 128

# TrueQuantRMSNorm + fused lm_head
python liteml_state_dict.py ... --true_quant --fuse_lm_head --group_size 128
```

### optimize_rotation.py — Train Rotation Matrices

Trains R1/R2 rotation matrices on WikiText2 using Stiefel manifold SGD (orthogonal constraint). Outputs `R.bin` for use with `--optimized_rotation_path` in `ptq.py`.

Multi-GPU training:
```bash
bash scripts/10_optimize_rotation.sh meta-llama/Llama-2-7b-hf 4 8 8
# args: model_name w_bits a_bits k_bits
```

### scripts/

- `eval_ptq_llama_models.sh` — Batch eval across Llama/Qwen/DeepSeek models; pass a model name to eval a single one
- `export_liteml_llama_models.sh` — Batch export to LiteML across model variants and export modes
- `2_eval_ptq.sh` — Single-GPU PTQ eval template
- `10_optimize_rotation.sh` / `11_optimize_rotation_fsdp.sh` — Distributed rotation training (8 GPUs via `torchrun`)

## Architecture

### Data Flow

```
HuggingFace model
    ↓
ptq.py
    ├─ Fuse layer norms
    ├─ Apply rotations: R1 (global, embeddings+head), R2 (per-layer, attention head_dim)
    ├─ Weight quantization: GPTQ (Hessian-based) or RTN
    ├─ Activation/KV-cache quantization wrappers
    ├─ WikiText2 perplexity eval
    └─ Save {model: state_dict, w_quantizers: {...}}
        ↓
liteml_state_dict.py
    ├─ Rekey state dict to LiteML naming
    ├─ Convert ZP: unsigned → signed qint domain
    ├─ [--fuse_lm_head] Absorb final RMSNorm into lm_head weights
    ├─ [--true_quant] Rename keys for TrueQuantRMSNorm compatibility
    └─ Save LiteML-compatible .pth
        ↓
ailabs_liteml RetrainerModel (further QAT / pruning / ONNX export)
```

### Module Roles

**`eval_utils/`** — Used by `ptq.py`:
- `main.py` → `ptq_model()`: orchestrates rotation application + quantization
- `rotation_utils.py`: functions to rotate embeddings, Q/K/V/O projections, MLP weights; `QKRotationWrapper` for post-RoPE K/Q quantization
- `gptq_utils.py`: `GPTQ` class with Hessian computation and `gptq_fwrd()` / `rtn_fwrd()`
- `modeling_llama.py`: LLaMA model with quantization hooks

**`train_utils/`** — Used by `optimize_rotation.py`:
- `main.py` → `prepare_model()`: sets up model for rotation training
- `quant_linear.py`: `QuantizeLinear` with learnable R1/R2 rotations
- `optimizer.py`: `SGDG` — SGD with Stiefel manifold constraint (QR retraction)
- `fsdp_trainer.py`: multi-GPU FSDP training loop
- `apply_r3_r4.py`: applies R4 Hadamard to `down_proj`

**`utils/`** — Shared:
- `quant_utils.py`: `ActQuantizer`, `WeightQuantizer`, `ActQuantWrapper`, `find_qlayers()`
- `hadamard_utils.py`: `random_hadamard_matrix()`, `apply_exact_had_to_linear()`, `HadamardTransform`
- `fuse_norm_utils.py`: `fuse_ln_linear()` / `fuse_layer_norms()` — absorbs RMSNorm scale into adjacent linear weights
- `data_utils.py`: WikiText2 loading + tokenization
- `process_args.py`: all CLI argument dataclasses (`ModelArguments`, `TrainingArguments`, `parser_gen()`)

### Key Concepts

- **R1**: Single hidden_size × hidden_size rotation applied to embeddings and lm_head
- **R2**: Per-layer head_dim × head_dim rotation applied to attention projections
- **Group quantization**: `--w_groupsize 128` means one scale per 128 input elements; `-1` = per-channel
- **V/O-proj special casing**: grouped by `head_dim` rather than a fixed group size
- **Down-proj groupsize**: computed dynamically via `llama_down_proj_groupsize()` based on model hidden size
- **Zero-point domain**: SpinQuant stores ZP in unsigned range [0, 2^b]; LiteML expects signed qint range — `convert_spinquant_zp_for_liteml()` handles the shift
