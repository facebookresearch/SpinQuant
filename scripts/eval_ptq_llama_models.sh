#!/bin/bash
# coding=utf-8
# PTQ evaluation commands for Llama and Qwen model families.
# Run a specific model by passing its name as argument, or run all if no argument given.
# Usage:
#   bash scripts/eval_ptq_llama_models.sh                      # run all models
#   bash scripts/eval_ptq_llama_models.sh Llama-3.1-8B         # run a specific model

COMMON_ARGS_LLAMA="
    --do_train False
    --do_eval True
    --per_device_eval_batch_size 4
    --model_max_length 1024
    --fp16 True
    --bf16 False
    --save_safetensors False
    --w_bits 4
    --a_bits 8
    --k_bits 8
    --v_bits 8
    --rotate
    --w_clip
    --cache_dir /AI_Labs/Models
"

# Qwen: bf16, no k/v quantization (KV stays float)
COMMON_ARGS_QWEN="
    --do_train False
    --do_eval True
    --per_device_eval_batch_size 4
    --model_max_length 1024
    --fp16 False
    --bf16 True
    --save_safetensors False
    --w_bits 4
    --a_bits 8
    --rotate
    --w_clip
    --cache_dir /AI_Labs/Models
"

run_llama() {
    local input_model=$1
    local save_path=$2
    local kv_groupsize=${3:-128}  # default kv groupsize is 128
    local wa_groupsize=${4:-128}  # default w/a groupsize is 128
    echo "=========================================="
    echo "Running PTQ for: ${input_model} (kv_groupsize=${kv_groupsize}, wa_groupsize=${wa_groupsize})"
    echo "=========================================="
    python ptq.py \
        --input_model "${input_model}" \
        --save_qmodel_path "${save_path}" \
        --k_groupsize "${kv_groupsize}" \
        --v_groupsize "${kv_groupsize}" \
        --w_groupsize "${wa_groupsize}" \
        --a_groupsize "${wa_groupsize}" \
        ${COMMON_ARGS_LLAMA}
}

run_qwen() {
    local input_model=$1
    local save_path=$2
    local wa_groupsize=${3:-128}  # default w/a groupsize is 128
    echo "=========================================="
    echo "Running PTQ for: ${input_model} (wa_groupsize=${wa_groupsize})"
    echo "=========================================="
    python ptq.py \
        --input_model "${input_model}" \
        --save_qmodel_path "${save_path}" \
        --w_groupsize "${wa_groupsize}" \
        --a_groupsize "${wa_groupsize}" \
        ${COMMON_ARGS_QWEN}
}

run_all() {
    # Llama models
    run_llama "meta-llama/Llama-2-7b-hf"     "saved_models/llama-2-7b/llama-2-7b-spinquant_gptq_group128.pth"
    run_llama "meta-llama/Meta-Llama-3-8B"    "saved_models/llama-3-8b/llama-3-8b-spinquant_gptq_group128.pth"
    run_llama "meta-llama/Llama-3.1-8B"       "saved_models/llama-3-1-8b/llama-3-1-8b-spinquant_gptq_group128.pth"
    run_llama "meta-llama/Llama-3.2-1B"       "saved_models/llama-3-2-1b/llama-3-2-1b-spinquant_gptq_kv64_group128.pth"  64  128
    run_llama "meta-llama/Llama-3.2-3B"       "saved_models/llama-3-2-3b/llama3-2-3b-spinquant_gptq_group128.pth"
    # Qwen models
    run_qwen  "Qwen/Qwen2.5-1.5B-Instruct"   "saved_models/qwen2/qwen2.5-1.5b-spinquant_gptq_g128_kv_float.pth"
    run_qwen  "Qwen/Qwen2.5-3B"              "saved_models/qwen2/qwen2.5-3b-spinquant_gptq_g128_kv_float.pth"
    # DeepSeek models
    run_qwen  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  "saved_models/deepseek-r1-distill-qwen-1.5b/deepseek-r1-distill-qwen-1.5b-spinquant_gptq_g128_kv_float.pth"
}

case "$1" in
    "Llama-2-7b-hf")
        run_llama "meta-llama/Llama-2-7b-hf"  "saved_models/llama-2-7b/llama-2-7b-spinquant_gptq_group128.pth" ;;
    "Meta-Llama-3-8B")
        run_llama "meta-llama/Meta-Llama-3-8B" "saved_models/llama-3-8b/llama-3-8b-spinquant_gptq_group128.pth" ;;
    "Llama-3.1-8B")
        run_llama "meta-llama/Llama-3.1-8B"    "saved_models/llama-3-1-8b/llama-3-1-8b-spinquant_gptq_group128.pth" ;;
    "Llama-3.2-1B")
        run_llama "meta-llama/Llama-3.2-1B"    "saved_models/llama-3-2-1b/llama-3-2-1b-spinquant_gptq_kv64_group128.pth"  64  128 ;;
    "Llama-3.2-3B")
        run_llama "meta-llama/Llama-3.2-3B"    "saved_models/llama-3-2-3b/llama3-2-3b-spinquant_gptq_group128.pth" ;;
    "Qwen2.5-1.5B-Instruct")
        run_qwen  "Qwen/Qwen2.5-1.5B-Instruct" "saved_models/qwen2/qwen2.5-1.5b-spinquant_gptq_g128_kv_float.pth" ;;
    "Qwen2.5-3B")
        run_qwen  "Qwen/Qwen2.5-3B"              "saved_models/qwen2/qwen2.5-3b-spinquant_gptq_g128_kv_float.pth" ;;
    "DeepSeek-R1-Distill-Qwen-1.5B")
        run_qwen  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  "saved_models/deepseek-r1-distill-qwen-1.5b/deepseek-r1-distill-qwen-1.5b-spinquant_gptq_g128_kv_float.pth" ;;
    "")
        run_all ;;
    *)
        echo "Unknown model: $1"
        echo "Available models: Llama-2-7b-hf, Meta-Llama-3-8B, Llama-3.1-8B, Llama-3.2-1B, Llama-3.2-3B, Qwen2.5-1.5B-Instruct, Qwen2.5-3B, DeepSeek-R1-Distill-Qwen-1.5B"
        exit 1 ;;
esac

