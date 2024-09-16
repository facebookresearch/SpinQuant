# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

torchrun --nnodes=1 --nproc_per_node=1 ptq.py \
--input_model $1 \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 4 \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--save_safetensors False \
--w_bits 4 \
--a_bits 8 \
--w_clip \
--w_groupsize 256 \
--a_asym \
--rotate \
--optimized_rotation_path "your_path/R.bin" \
--save_qmodel_path "./your_output_model_path/executorch_model.bin" \
--export_to_et
