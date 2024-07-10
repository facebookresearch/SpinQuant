#!/bin/sh
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

torchrun --nnodes=1 --nproc_per_node=8 optimize_rotation.py \
--input_model_filename "/fsx_0/user/cszhao/downloads/checkpoint/llama/llama3-70b/" \
--output_model_filename "/fsx_0/user/cszhao/dump/checkpoint/spin/R2/70B-w4a4kv4" \
--output_dir "/fsx_0/user/cszhao/test/" \
--do_train True \
--do_eval False \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--ddp_find_unused_parameters False \
--logging_dir "/fsx_0/user/cszhao/test/log/" \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--save_steps 1000 \
--eval_steps 1000 \
--logging_steps 1 \
--evaluation_strategy "no" \
--save_strategy "no" \
--report_to "tensorboard" \
--save_total_limit 1 \
--learning_rate 1.5 \
--weight_decay 0. \
--lr_scheduler_type "cosine" \
--gradient_checkpointing True \
--save_safetensors False \
--max_steps 100 \
--w_bits 4 \
--a_bits 4 \
--k_bits 4 \
--v_bits 4 \
--w_clip \
--rotate \
--fsdp "full_shard auto_wrap" \
--fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'