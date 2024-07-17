# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import logging
import os
import random
from typing import Optional

import numpy as np
import torch
from fast_hadamard_transform import hadamard_transform
from torch.distributed.fsdp import (
    FullStateDictConfig,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as PT_FSDP,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

# These flags disable using TensorFloat-32 tensor cores (to avoid numerical issues)
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False
DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def pt_fsdp_state_dict(model: torch.nn.Module):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with PT_FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        return model.state_dict()


class HadamardTransform(torch.autograd.Function):
    """The unnormalized Hadamard transform (i.e. without dividing by sqrt(2))"""

    @staticmethod
    def forward(ctx, u):
        return hadamard_transform(u)

    @staticmethod
    def backward(ctx, grad):
        return hadamard_transform(grad)


def llama_down_proj_groupsize(model, groupsize):
    assert groupsize > 1, "groupsize should be greater than 1!"

    if model.config.intermediate_size % groupsize == 0:
        logging.info(f"(Act.) Groupsiz = Down_proj Groupsize: {groupsize}")
        return groupsize

    group_num = int(model.config.hidden_size / groupsize)
    assert (
        groupsize * group_num == model.config.hidden_size
    ), "Invalid groupsize for llama!"

    down_proj_groupsize = model.config.intermediate_size // group_num
    assert (
        down_proj_groupsize * group_num == model.config.intermediate_size
    ), "Invalid groupsize for down_proj!"
    logging.info(
        f"(Act.) Groupsize: {groupsize}, Down_proj Groupsize: {down_proj_groupsize}"
    )
    return down_proj_groupsize


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


# Dump the log both to console and a log file.
def config_logging(log_file, level=logging.INFO):
    class LogFormatter(logging.Formatter):
        def format(self, record):
            if record.levelno == logging.INFO:
                self._style._fmt = "%(message)s"
            else:
                self._style._fmt = "%(levelname)s: %(message)s"
            return super().format(record)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(LogFormatter())

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(LogFormatter())

    logging.basicConfig(level=level, handlers=[console_handler, file_handler])


def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect

    caller_name = ""
    try:
        caller_name = f" (from {inspect.stack()[1].function})"
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(
            torch.cuda.memory_reserved(device=i)
            for i in range(torch.cuda.device_count())
        )

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )


# Define a utility method for setting the logging parameters of a logger
def get_logger(logger_name: Optional[str]) -> logging.Logger:
    # Get the logger with the specified name
    logger = logging.getLogger(logger_name)

    # Set the logging level of the logger to INFO
    logger.setLevel(logging.INFO)

    # Define a formatter for the log messages
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a console handler for outputting log messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    return logger


def get_local_rank() -> int:
    if os.environ.get("LOCAL_RANK"):
        return int(os.environ["LOCAL_RANK"])
    else:
        logging.warning(
            "LOCAL_RANK from os.environ is None, fall back to get rank from torch distributed"
        )
        return torch.distributed.get_rank()


def get_global_rank() -> int:
    """
    Get rank using torch.distributed if available. Otherwise, the RANK env var instead if initialized.
    Returns 0 if neither condition is met.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()

    environ_rank = os.environ.get("RANK", "")
    if environ_rank.isdecimal():
        return int(os.environ["RANK"])

    return 0
