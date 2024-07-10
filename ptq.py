import datetime
import logging
import os
from logging import Logger
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import transformers

from eval_utils.main import ptq_model
from eval_utils.modeling_llama import LlamaForCausalLM
from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq


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


log: Logger = get_logger("spinquant")


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




def train() -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    model_args, data_args, training_args, ptq_args = process_args_ptq()
    local_rank = get_local_rank()

    log.info("the rank is {}".format(local_rank))
    torch.distributed.barrier()

    if data_args.checkpoint_local_path is not None:
        ptq_args.checkpoint_local_path = data_args.checkpoint_local_path
    else:
        ptq_args.checkpoint_local_path = None
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model_filename
    )

    model = ptq_model(ptq_args, model, model_args)
    model.seqlen = training_args.model_max_length
    if local_rank == 0:
        log.info("Model PTQ completed {}".format(model))
        log.info("Start to load tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model_filename,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
    )
    log.info("Complete tokenizer loading...")
    model.config.use_cache = False

    testloader = data_utils.get_loaders(
        ptq_args.eval_dataset,
        seed=ptq_args.seed,
        seqlen=2048,
        tokenizer=tokenizer,
        eval_mode=True,
    )

    dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, ptq_args)
    log.info("wiki2 ppl is: {}".format(dataset_ppl))
    dist.barrier()


if __name__ == "__main__":
    train()
