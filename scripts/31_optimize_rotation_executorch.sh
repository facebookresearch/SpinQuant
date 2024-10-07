# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.
torchrun --nnodes=1 --nproc_per_node=8 optimize_rotation.py \
--input_model $1  \
--output_rotation_path "your_path" \
--output_dir "your_output_path/" \
--logging_dir "your_log_path/" \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--per_device_train_batch_size 1 \
--logging_steps 1 \
--learning_rate 1.5 \
--weight_decay 0. \
--lr_scheduler_type "cosine" \
--gradient_checkpointing True \
--save_safetensors False \
--max_steps 100 \
--w_bits 16 \
--a_bits 8 \
--w_clip \
--a_asym \
--w_groupsize 32
