#!/bin/sh
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

torchrun --nnodes=1 --nproc_per_node=1 optimize_rotation.py \
--input_model_filename $1  \
--output_model_filename "/fsx/zechunliu/8_spinquant_oss/output_models/spin/Q2/7b-w$2a$3kv$4" \
--output_dir "/fsx/zechunliu/8_spinquant_oss/test/output/" \
--logging_dir "/fsx/zechunliu/8_spinquant_oss/test/log/" \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--ddp_find_unused_parameters False \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 8 \
--gradient_accumulation_steps 1 \
--save_steps 1000 \
--eval_steps 1000 \
--logging_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--report_to "tensorboard" \
--save_total_limit 1 \
--learning_rate 1.5 \
--weight_decay 0. \
--lr_scheduler_type "cosine" \
--gradient_checkpointing True \
--save_safetensors False \
--max_steps 10 \
--w_bits $2 \
--a_bits $3 \
--k_bits $4 \
--v_bits $4 \
--w_clip \
--rotate
