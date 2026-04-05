#!/bin/bash
# coding=utf-8
# LiteML state dict export commands for Llama model family.
# Uncomment the model(s) you want to export and run this script.

# Llama-2-7b
#python liteml_state_dict.py \
#    --spinquant_path saved_models/llama-2-7b/llama-2-7b-spinquant_gptq_group128.pth \
#    --liteml_path saved_models/llama-2-7b/liteml_llama-2-7b-spinquant_gptq_group128_fused_lm_head.pth \
#    --fuse_lm_head \
#    --group_size 128
python liteml_state_dict.py \
    --spinquant_path saved_models/llama-2-7b/llama-2-7b-spinquant_gptq_group128.pth \
    --liteml_path saved_models/llama-2-7b/liteml_llama-2-7b-spinquant_gptq_group128.pth \
    --group_size 128


# Meta-Llama-3-8B
#python liteml_state_dict.py \
#    --spinquant_path saved_models/llama-3-8b/llama-3-8b-spinquant_gptq_group128.pth \
#    --liteml_path saved_models/llama-3-8b/liteml_llama-3-8b-spinquant_gptq_group128_fused_lm_head.pth \
#    --fuse_lm_head \
#    --group_size 128

# Llama-3.1-8B
#python liteml_state_dict.py \
#    --spinquant_path saved_models/llama-3-1-8b/llama-3-1-8b-spinquant_gptq_group128.pth \
#    --liteml_path saved_models/llama-3-1-8b/liteml_llama-3-1-8b-spinquant_gptq_group128_fused_lm_head.pth \
#    --fuse_lm_head \
#    --group_size 128

# Llama-3.2-1B (kv_groupsize=64, wa_groupsize=128)
#python liteml_state_dict.py \
#    --spinquant_path saved_models/llama-3-2-1b/llama-3-2-1b-spinquant_gptq_kv64_group128.pth \
#    --liteml_path saved_models/llama-3-2-1b/liteml_llama-3-2-1b-spinquant_gptq_kv64_group128_fused_lm_head.pth \
#    --fuse_lm_head \
#    --group_size 128

# Llama-3.2-3B
#python liteml_state_dict.py \
#    --spinquant_path saved_models/llama-3-2-3b/llama3-2-3b-spinquant_gptq_group128.pth \
#    --liteml_path saved_models/llama-3-2-3b/liteml_llama3-2-3b-spinquant_gptq_group128_fused_lm_head.pth \
#    --fuse_lm_head \
#    --group_size 128

# Qwen2.5-1.5B-Instruct
#python liteml_state_dict.py \
#    --spinquant_path saved_models/qwen2/qwen2.5-1.5b-spinquant_gptq_g128_kv_float.pth \
#    --liteml_path saved_models/qwen2/liteml_qwen2.5-1.5b-spinquant_gptq_g128_kv_float.pth \
#    --group_size 128

# Qwen2.5-3B
#python liteml_state_dict.py \
#    --spinquant_path saved_models/qwen2/qwen2.5-3b-spinquant_gptq_g128_kv_float.pth \
#    --liteml_path saved_models/qwen2/liteml_qwen2.5-3b-spinquant_gptq_g128_kv_float.pth \
#    --group_size 128

# DeepSeek-R1-Distill-Qwen-1.5B
#python liteml_state_dict.py \
#    --spinquant_path saved_models/deepseek-r1-distill-qwen-1.5b-r1-distill-qwen-1.5b/deepseek-r1-distill-qwen-1.5b-r1-distill-qwen-1.5b-spinquant_gptq_g128_kv_float.pth \
#    --liteml_path saved_models/deepseek-r1-distill-qwen-1.5b-r1-distill-qwen-1.5b/liteml_deepseek-r1-distill-qwen-1.5b-spinquant_gptq_g128_kv_float.pth \
#    --group_size 128

