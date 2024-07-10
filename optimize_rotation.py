import datetime
import logging
import os
from logging import Logger
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import transformers
from torch import nn
from transformers import Trainer, default_data_collator
import datasets
from train_utils.fsdp_trainer import FSDPTrainer
from train_utils.main import prepare_model
from train_utils.modeling_llama_quant import LlamaForCausalLM as LlamaForCausalLMQuant
from train_utils.optimizer import SGDG
from utils.data_utils import CustomJsonDataset
from utils.hadamard_utils import hadamard_matrix
from utils.process_args import process_args_ptq
from utils.utils import pt_fsdp_state_dict


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


class RotateModule(nn.Module):
    def __init__(self, R_init):
        super(RotateModule, self).__init__()
        self.weight = nn.Parameter(R_init.to(torch.float32).to(torch.device("cuda")))

    def forward(self, x, transpose=False):
        if transpose:
            return x @ self.weight
        else:
            return self.weight @ x


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
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = LlamaForCausalLMQuant.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model_filename,
        torch_dtype=dtype,
    )

    model = prepare_model(ptq_args, model)
    if training_args.do_train:
        for param in model.parameters():
            param.requires_grad = False
        R1 = torch.empty(model.config.hidden_size, model.config.hidden_size)
        nn.init.orthogonal_(R1)
        model.R1 = RotateModule(R1)
        for i in range(model.config.num_hidden_layers):
            # Each head dim = 128 for Llama model
            R2 = hadamard_matrix(
                model.config.hidden_size // model.config.num_attention_heads, "cuda"
            )
            model.model.layers[i].self_attn.R2 = RotateModule(R2)
    if local_rank == 0:
        log.info("Model init completed for training {}".format(model))
        log.info("Start to load tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model_filename,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
    )
    log.info("Complete tokenizer loading...")
    model.config.use_cache = False
    calibration_datasets = datasets.load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1"
    )
    train_data = CustomJsonDataset(
        calibration_datasets["train"],
        tokenizer,
        block_size=min(training_args.model_max_length, 2048),
    )

    if training_args.do_train:
        trainable_parameters = [model.R1.weight] + [
            model.model.layers[i].self_attn.R2.weight
            for i in range(model.config.num_hidden_layers)
        ]
        optimizer = SGDG(
            trainable_parameters, lr=training_args.learning_rate, stiefel=True
        )
        MyTrainer = Trainer
        # Use FSDP for 70B rotation training
        if training_args.fsdp != "" and training_args.fsdp != []:
            MyTrainer = FSDPTrainer

        trainer = MyTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=None,
            data_collator=default_data_collator,
            optimizers=(optimizer, None),
        )
        torch.distributed.barrier()

        trainer.train()
        if training_args.fsdp != "" and training_args.fsdp != []:
            cpu_state = pt_fsdp_state_dict(trainer.model)
        else:
            cpu_state = trainer.model.state_dict()

        R_dict = {
            key.replace(".weight", ""): value
            for key, value in cpu_state.items()
            if "R1.weight" in key or "self_attn.R2" in key
        }
        if local_rank == 0:
            os.makedirs(model_args.output_model_filename, exist_ok=True)
            path = os.path.join(model_args.output_model_filename, "R.bin")
            torch.save(
                R_dict,
                path,
            )
    dist.barrier()


if __name__ == "__main__":
    train()
