# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple

import argparse
import transformers

@dataclass
class ModelArguments:
    local_dir: str = field(
        default=None, metadata={"help": "Local Path of storing inputs and outputs "}
    )
    input_model_filename: Optional[str] = field(
        default="test-input", metadata={"help": "Input model relative path"}
    )
    output_model_filename: Optional[str] = field(
        default="test-output", metadata={"help": "Output model relative path"}
    )


@dataclass
class DataArguments:
    train_data_local_path: Optional[str] = field(
        default=None, metadata={"help": "Train data local path"}
    )
    eval_data_local_path: Optional[str] = field(
        default=None, metadata={"help": "Eval data local path"}
    )
    checkpoint_local_path: Optional[str] = field(
        default=None, metadata={"help": "Rotation checkpoint_local_path"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(default="/tmp/output/")
    model_max_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)"
        },
    )


def parser_gen():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=0, help="Random Seed for HuggingFace and PyTorch"
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="wikitext2",
        help="Dataset for Evaluation (default: wikitext2)",
        choices=["wikitext2", "ptb", "c4"],
    )
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument(
        "--bsz", type=int, default=32, help="Batch-size for PPL evaluation (default:32)"
    )

    # Rotation Arguments
    parser.add_argument(
        "--rotate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="""Rotate the moodel. This will include online rotation for down-projection and
                        out-projection. Note that this does not apply rotation to the K/Q and they will be rotated
                        if we want to quantize the Keys""",
    )
    parser.add_argument(
        "--rotate_mode", type=str, default="hadamard", choices=["hadamard", "random"]
    )
    parser.add_argument(
        "--rotation_seed",
        type=int,
        default=-1,
        help="Random Seed for generating random matrix!!",
    )
    parser.add_argument(
        "--fp32_had",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply Hadamard rotation in FP32 (default: False)",
    )

    # Activation Quantization Arguments
    parser.add_argument(
        "--a_bits",
        type=int,
        default=16,
        help="""Number of bits for inputs of the Linear layers. This will be
                        for all the linear layers in the model (including down-projection and out-projection)""",
    )
    parser.add_argument(
        "--a_groupsize",
        type=int,
        default=-1,
        help="Groupsize for activation quantization. Note that this should be the same as w_groupsize",
    )
    parser.add_argument(
        "--a_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric Activation quantization (default: False)",
    )
    parser.add_argument(
        "--a_clip_ratio",
        type=float,
        default=1.0,
        help="Clip ratio for activation quantization. new_max = max * clip_ratio",
    )

    # Weight Quantization Arguments
    parser.add_argument(
        "--w_bits",
        type=int,
        default=16,
        help="Number of bits for weights of the Linear layers",
    )
    parser.add_argument(
        "--w_groupsize",
        type=int,
        default=-1,
        help="Groupsize for weight quantization. Note that this should be the same as a_groupsize",
    )
    parser.add_argument(
        "--w_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric weight quantization (default: False)",
    )
    parser.add_argument(
        "--w_rtn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Quantize the weights using RtN. If the w_bits < 16 and this flag is not set, we use GPTQ",
    )
    parser.add_argument(
        "--w_clip",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="""Clipping the weight quantization!
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization""",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="Number of calibration data samples for GPTQ.",
    )
    parser.add_argument(
        "--cal_dataset",
        type=str,
        default="wikitext2",
        help="calibration data samples for GPTQ.",
        choices=["wikitext2", "ptb", "c4"],
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--act_order",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="act-order in GPTQ",
    )

    # General Quantization Arguments
    parser.add_argument(
        "--int8_down_proj",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use INT8 for Down Projection! If this set, both weights and activations of this layer will be in INT8",
    )

    # KV-Cache Quantization Arguments
    parser.add_argument(
        "--v_bits",
        type=int,
        default=16,
        help="""Number of bits for V-cache quantization.
                        Note that quantizing the V-cache does not need any other rotation""",
    )
    parser.add_argument("--v_groupsize", type=int, default=-1)
    parser.add_argument(
        "--v_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric V-cache quantization",
    )
    parser.add_argument(
        "--v_clip_ratio",
        type=float,
        default=1.0,
        help="Clip ratio for v-cache quantization. new_max = max * clip_ratio",
    )

    parser.add_argument(
        "--k_bits",
        type=int,
        default=16,
        help="""Number of bits for K-cache quantization.
                        Note that quantizing the K-cache needs another rotation for the keys/queries""",
    )
    parser.add_argument("--k_groupsize", type=int, default=-1)
    parser.add_argument(
        "--k_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric K-cache quantization",
    )
    parser.add_argument(
        "--k_pre_rope",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pre-RoPE quantization for K-cache (not Supported yet!)",
    )
    parser.add_argument(
        "--k_clip_ratio",
        type=float,
        default=1.0,
        help="Clip ratio for k-cache quantization. new_max = max * clip_ratio",
    )

    # Save/Load Quantized Model Arguments
    parser.add_argument(
        "--load_qmodel_path",
        type=str,
        default=None,
        help="Load the quantized model from the specified path!",
    )
    parser.add_argument(
        "--save_qmodel_path",
        type=str,
        default=None,
        help="Save the quantized model to the specified path!",
    )

    # WandB Arguments
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)

    # Experiments Arguments
    parser.add_argument(
        "--save_name",
        type=str,
        default=None,
        help="The path to save experiment data, "
        "including quantized models, dumped layer inputs, etc. The data will be saved in experiments/[model]/save_name. Default: [datetime].",
    )
    parser.add_argument(
        "--capture_layer_io",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Capture the input and output of the specified decoder layer and dump into a file",
    )
    parser.add_argument(
        "--layer_idx", type=int, default=10, help="Which decoder layer to capture"
    )

    args, unknown = parser.parse_known_args()

    # quant_type = f'w{args.w_bits}a{args.a_bits}_{args.rotate_mode}'
    if args.save_name is None:
        args.save_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    assert (
        args.a_groupsize == args.w_groupsize
    ), "a_groupsize should be the same as w_groupsize!"
    assert args.k_pre_rope == False, "Pre-RoPE quantization is not supported yet!"

    return args, unknown


def process_args_ptq() -> Tuple[ModelArguments, DataArguments, TrainingArguments, None]:
    ptq_args = None

    ptq_args, unknown_args = parser_gen()

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args=unknown_args
    )

    return model_args, data_args, training_args, ptq_args
