#!/bin/sh
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

torchrun --nnodes=1 --nproc_per_node=8 ptq.py \
--input_model_filename "/fsx_0/user/cszhao/downloads/checkpoint/llama/llama3-70b/" \
--do_train False \
--do_eval True \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--gradient_checkpointing False \
--w_bits 4 \
--a_bits 4 \
--k_bits 4 \
--v_bits 4 \
--w_clip \
--rotate \
--w_rtn
# --rotate_mode "random"
# --checkpoint_local_path "/fsx_0/user/cszhao/dump/checkpoint/spin/Q2/7b-w4a4kv4/R.bin" \