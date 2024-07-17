# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib.metadata
import inspect
import os
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

# isort: on
import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data import Dataset, IterableDataset, RandomSampler
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor

# Integrations must be imported before ML frameworks:
# isort: off
from transformers.integrations.deepspeed import is_deepspeed_available
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES
from transformers.pytorch_utils import (
    is_torch_greater_or_equal_than_2_3,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    ExportableState,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    LabelSmoother,
)
from transformers.trainer_utils import (
    enable_full_determinism,
    EvalPrediction,
    has_length,
    set_seed,
    TrainerMemoryTracker,
)
from transformers.training_args import ParallelMode, TrainingArguments
from transformers.utils import (
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_peft_available,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_xla_available,
    logging,
    XLA_FSDPV2_MIN_VERSION,
)
from transformers import Trainer
from transformers.data.data_collator import (
    DataCollator,
    DataCollatorWithPadding,
    default_data_collator,
)
from transformers.utils.quantization_config import QuantizationMethod


DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from .utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    pass

if is_datasets_available():
    import datasets

if is_torch_xla_available():
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(
        XLA_FSDPV2_MIN_VERSION
    )
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
        import torch_xla.runtime as xr
else:
    IS_XLA_FSDPV2_POST_2_2 = False


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if is_peft_available():
    from peft import PeftModel


if is_accelerate_available():
    from accelerate import __version__ as accelerate_version

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        pass

if is_accelerate_available("0.28.0"):
    pass


logger = logging.get_logger(__name__)


def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False


class FSDPTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[
            Union[Dataset, IterableDataset, "datasets.Dataset"]
        ] = None,
        eval_dataset: Optional[
            Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]
        ] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
    ):
        if args is None:
            output_dir = "tmp_trainer"
            logger.info(
                f"No `TrainingArguments` passed, using `output_dir={output_dir}`."
            )
            args = TrainingArguments(output_dir=output_dir)
        if args.batch_eval_metrics and compute_metrics is not None:
            if (
                "compute_result"
                not in inspect.signature(compute_metrics).parameters.keys()
            ):
                raise ValueError(
                    "When using `batch_eval_metrics`, your `compute_metrics` function must take a `compute_result`"
                    " boolean argument which will be triggered after the last batch of the eval set to signal that the"
                    " summary statistics should be returned by the function."
                )
        self.args = args
        # Seed must be set before instantiating the model when using model
        (
            enable_full_determinism(self.args.seed)
            if self.args.full_determinism
            else set_seed(self.args.seed)
        )
        self.hp_name = None
        self.deepspeed = None
        self.is_in_train = False

        self.create_accelerator_and_postprocess()

        # memory metrics - must set up as early as possible
        self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        self._memory_tracker.start()

        # set the correct log level depending on the node
        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)

        # force device and distributed setup init explicitly
        args._setup_devices

        if model is None:
            if model_init is not None:
                self.model_init = model_init
                model = self.call_model_init()
            else:
                raise RuntimeError(
                    "`Trainer` requires either a `model` or `model_init` argument"
                )
        else:
            if model_init is not None:
                warnings.warn(
                    "`Trainer` requires either a `model` or `model_init` argument, but not both. `model_init` will"
                    " overwrite your model when calling the `train` method. This will become a fatal error in the next"
                    " release.",
                    FutureWarning,
                )
            self.model_init = model_init

        if model.__class__.__name__ in MODEL_MAPPING_NAMES:
            raise ValueError(
                f"The model you have picked ({model.__class__.__name__}) cannot be used as is for training: it only "
                "computes hidden states and does not accept any labels. You should choose a model with a head "
                "suitable for your task like any of the `AutoModelForXxx` listed at "
                "https://huggingface.co/docs/transformers/model_doc/auto"
            )

        if getattr(model, "is_parallelizable", False) and getattr(
            model, "model_parallel", False
        ):
            self.is_model_parallel = True
        else:
            self.is_model_parallel = False

        if getattr(model, "hf_device_map", None) is not None:
            devices = [
                device
                for device in set(model.hf_device_map.values())
                if device not in ["cpu", "disk"]
            ]
            if len(devices) > 1:
                self.is_model_parallel = True
            elif len(devices) == 1:
                self.is_model_parallel = self.args.device != torch.device(devices[0])
            else:
                self.is_model_parallel = False

            # warn users
            if self.is_model_parallel:
                logger.info(
                    "You have loaded a model on multiple GPUs. `is_model_parallel` attribute will be force-set"
                    " to `True` to avoid any unexpected behavior such as device placement mismatching."
                )

        _is_quantized_and_base_model = getattr(
            model, "is_quantized", False
        ) and not getattr(model, "_hf_peft_config_loaded", False)
        _quantization_method_supports_training = (
            getattr(model, "hf_quantizer", None) is not None
            and model.hf_quantizer.is_trainable
        )

        # Filter out quantized + compiled models
        if _is_quantized_and_base_model and hasattr(model, "_orig_mod"):
            raise ValueError(
                "You cannot fine-tune quantized model with `torch.compile()` make sure to pass a non-compiled model when fine-tuning a quantized model with PEFT"
            )

        # At this stage the model is already loaded
        if _is_quantized_and_base_model and not _is_peft_model(model):
            raise ValueError(
                "You cannot perform fine-tuning on purely quantized models. Please attach trainable adapters on top of"
                " the quantized model to correctly perform fine-tuning. Please see: https://huggingface.co/docs/transformers/peft"
                " for more details"
            )
        elif (
            _is_quantized_and_base_model and not _quantization_method_supports_training
        ):
            raise ValueError(
                f"The model you are trying to fine-tune is quantized with {model.hf_quantizer.quantization_config.quant_method}"
                " but that quantization method do not support training. Please open an issue on GitHub: https://github.com/huggingface/transformers"
                f" to request the support for training support for {model.hf_quantizer.quantization_config.quant_method}"
            )

        self.is_fsdp_xla_enabled = args.fsdp_config["xla"]
        if len(args.fsdp) > 0:
            if self.is_deepspeed_enabled:
                raise ValueError(
                    "Using --fsdp xxx together with --deepspeed is not possible, deactivate one of those flags."
                )
            if (
                not args.fsdp_config["xla"]
                and args.parallel_mode != ParallelMode.DISTRIBUTED
            ):
                raise ValueError("Using fsdp only works in distributed training.")

        # one place to sort out whether to place the model on device or not
        # postpone switching model to cuda when:
        # 1. MP - since we are trying to fit a much bigger than 1 gpu model
        # 2. fp16-enabled DeepSpeed loads the model in half the size and it doesn't need .to() anyway,
        #    and we only use deepspeed for training at the moment
        # 3. full bf16 or fp16 eval - since the model needs to be cast to the right dtype first
        # 4. FSDP - same as MP
        self.place_model_on_device = args.place_model_on_device
        if (
            self.is_model_parallel
            or self.is_deepspeed_enabled
            or ((args.fp16_full_eval or args.bf16_full_eval) and not args.do_train)
            or self.is_fsdp_xla_enabled
            or self.is_fsdp_enabled
        ):
            self.place_model_on_device = False

        default_collator = (
            DataCollatorWithPadding(tokenizer)
            if tokenizer is not None
            and isinstance(
                tokenizer, (PreTrainedTokenizerBase, SequenceFeatureExtractor)
            )
            else default_data_collator
        )
        self.data_collator = (
            data_collator if data_collator is not None else default_collator
        )
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        # Bnb Quantized models doesn't support `.to` operation.
        if (
            self.place_model_on_device
            and not getattr(model, "quantization_method", None)
            == QuantizationMethod.BITS_AND_BYTES
        ):
            self._move_model_to_device(model, args.device)

        # Force n_gpu to 1 to avoid DataParallel as MP will manage the GPUs
        if self.is_model_parallel:
            self.args._n_gpu = 1

        # later use `self.model is self.model_wrapped` to check if it's wrapped or not
        self.model_wrapped = model
        self.model = model

        self.neftune_noise_alpha = args.neftune_noise_alpha

        self.compute_metrics = compute_metrics
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self.optimizer, self.lr_scheduler = optimizers

        if model_init is not None and (
            self.optimizer is not None or self.lr_scheduler is not None
        ):
            raise RuntimeError(
                "Passing a `model_init` is incompatible with providing the `optimizers` argument. "
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )
        if is_torch_xla_available() and self.optimizer is not None:
            for param in self.model.parameters():
                model_device = param.device
                break
            for param_group in self.optimizer.param_groups:
                if len(param_group["params"]) > 0:
                    optimizer_device = param_group["params"][0].device
                    break
            if model_device != optimizer_device:
                raise ValueError(
                    "The model and the optimizer parameters are not on the same device, which probably means you"
                    " created an optimizer around your model **before** putting on the device and passing it to the"
                    " `Trainer`. Make sure the lines `import torch_xla.core.xla_model as xm` and"
                    " `model.to(xm.xla_device())` is performed before the optimizer creation in your script."
                )

        # Overwrite FSDP Optimizer setting
        # if (
        #     self.is_deepspeed_enabled
        #     or self.is_fsdp_xla_enabled
        #     or self.is_fsdp_enabled
        # ) and (self.optimizer is not None or self.lr_scheduler is not None):
        #     raise RuntimeError(
        #         "Passing `optimizers` is not allowed if Deepspeed or PyTorch FSDP is enabled. "
        #         "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
        #     )
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(
            self.args.report_to
        )
        callbacks = (
            default_callbacks if callbacks is None else default_callbacks + callbacks
        )
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.add_callback(
            PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK
        )

        # Will be set to True by `self._setup_loggers()` on first call to `self.log()`.
        self._loggers_initialized = False

        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        if not callable(self.data_collator) and callable(
            getattr(self.data_collator, "collate_batch", None)
        ):
            raise ValueError(
                "The `data_collator` should be a simple callable (function, class with `__call__`)."
            )

        if args.max_steps > 0 and args.num_train_epochs > 0:
            logger.warning(
                "max_steps is given, it will override any value given in num_train_epochs"
            )

        if (
            train_dataset is not None
            and not has_length(train_dataset)
            and args.max_steps <= 0
        ):
            raise ValueError(
                "The train_dataset does not implement __len__, max_steps has to be specified. "
                "The number of steps needs to be known in advance for the learning rate scheduler."
            )

        if (
            train_dataset is not None
            and isinstance(train_dataset, torch.utils.data.IterableDataset)
            and args.group_by_length
        ):
            raise ValueError(
                "the `--group_by_length` option is only available for `Dataset`, not `IterableDataset"
            )

        self._signature_columns = None

        # Mixed precision setup
        self.use_apex = False
        self.use_cpu_amp = False

        # Mixed precision setup for SageMaker Model Parallel
        if is_sagemaker_mp_enabled():
            # BF16 + model parallelism in SageMaker: currently not supported, raise an error
            if args.bf16:
                raise ValueError(
                    "SageMaker Model Parallelism does not support BF16 yet. Please use FP16 instead "
                )

            if IS_SAGEMAKER_MP_POST_1_10:
                # When there's mismatch between SMP config and trainer argument, use SMP config as truth
                if args.fp16 != smp.state.cfg.fp16:
                    logger.warning(
                        f"FP16 provided in SM_HP_MP_PARAMETERS is {smp.state.cfg.fp16}, "
                        f"but FP16 provided in trainer argument is {args.fp16}, "
                        f"setting to {smp.state.cfg.fp16}"
                    )
                    args.fp16 = smp.state.cfg.fp16
            else:
                # smp < 1.10 does not support fp16 in trainer.
                if hasattr(smp.state.cfg, "fp16"):
                    logger.warning(
                        f"FP16 provided in SM_HP_MP_PARAMETERS is {smp.state.cfg.fp16}, "
                        "but SageMaker Model Parallelism < 1.10 does not support FP16 in trainer."
                    )
        if (args.fp16 or args.bf16) and args.half_precision_backend == "auto":
            if args.device == torch.device("cpu"):
                if args.fp16:
                    if not is_torch_greater_or_equal_than_2_3:
                        raise ValueError(
                            "Tried to use `fp16` but it is not supported on cpu"
                        )
                else:
                    args.half_precision_backend = "cpu_amp"
            logger.info(f"Using {args.half_precision_backend} half precision backend")

        if (args.fp16 or args.bf16) and not (
            self.is_deepspeed_enabled or is_sagemaker_mp_enabled()
        ):
            # deepspeed and SageMaker Model Parallel manage their own half precision
            if args.half_precision_backend == "cpu_amp":
                self.use_cpu_amp = True
                self.amp_dtype = torch.bfloat16
            elif args.half_precision_backend == "apex":
                if not is_apex_available():
                    raise ImportError(
                        "Using FP16 with APEX but APEX is not installed, please refer to"
                        " https://www.github.com/nvidia/apex."
                    )
                self.use_apex = True

        # Label smoothing
        if self.args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(
                epsilon=self.args.label_smoothing_factor
            )
        else:
            self.label_smoother = None

        self.control = TrainerControl()

        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb
                for cb in self.callback_handler.callbacks + [self.control]
                if isinstance(cb, ExportableState)
            ],
        )
        # Internal variable to count flos in each process, will be accumulated in `self.state.total_flos` then
        # returned to 0 every time flos need to be logged
        self.current_flos = 0
        self.hp_search_backend = None
        default_label_names = find_labels(self.model.__class__)
        self.label_names = (
            default_label_names
            if self.args.label_names is None
            else self.args.label_names
        )
        self.can_return_loss = can_return_loss(self.model.__class__)
        self.control = self.callback_handler.on_init_end(
            self.args, self.state, self.control
        )

        # Internal variables to help with automatic batch size reduction
        self._train_batch_size = args.train_batch_size
        self._created_lr_scheduler = False

        # very last
        self._memory_tracker.stop_and_update_metrics()

        # torch.compile
        if args.torch_compile and not is_torch_compile_available():
            raise RuntimeError("Using torch.compile requires PyTorch 2.0 or higher.")

        self.is_fsdp_xla_v2_enabled = args.fsdp_config.get("xla_fsdp_v2", False)
        if self.is_fsdp_xla_v2_enabled:
            if not IS_XLA_FSDPV2_POST_2_2:
                raise ValueError("FSDPv2 requires `torch_xla` 2.2 or higher.")
            # Prepare the SPMD mesh that is going to be used by the data loader and the FSDPv2 wrapper.
            # Tensor axis is just a placeholder where it will not be used in FSDPv2.
            num_devices = xr.global_runtime_device_count()
            xs.set_global_mesh(
                xs.Mesh(
                    np.array(range(num_devices)),
                    (num_devices, 1),
                    axis_names=("fsdp", "tensor"),
                )
            )
        # Do not wrap rotation matrix
        self.accelerator.state.fsdp_plugin.ignored_modules = [model.R1] + [
            layer.self_attn.R2 for layer in model.model.layers
        ]
        # use_orig_params because part of the model is freezed
        self.accelerator.state.fsdp_plugin.use_orig_params = True

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """

        # Overwrite optimizer creation because optimizer is already created
        optimizer = self.optimizer
        self.create_scheduler(
            num_training_steps=num_training_steps,
            optimizer=optimizer,
        )
