import copy
import math
import os
import sys
import time
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch_npu
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from optimum.ascend.training_args import NPUTrainingArguments
from optimum.utils import logging
from transformers import Trainer, __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations import (
    hp_params,
    get_reporting_integration_callbacks,
)
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_10, is_torch_less_than_1_11
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import IterableDatasetShard, get_parameter_names
from transformers.trainer_utils import (
    EvalPrediction,
    HPSearchBackend,
    TrainOutput,
    TrainerMemoryTracker,
    ShardedDDPOption,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    set_seed,
    speed_metrics,
)
from transformers.training_args import TrainingArguments
from transformers.utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    can_return_loss,
    find_labels,
    is_apex_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
)

_is_native_cpu_amp_available = is_torch_greater_or_equal_than_1_10

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if TYPE_CHECKING:
    import optuna

if is_apex_available():
    from apex import amp

logger = logging.get_logger(__name__)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class NPUTrainer(Trainer):
    """
    NPUTrainer is built on top of the tranformers' Trainer to enable
    deployment on Ascend's NPU.
    """

    def __init__(
            self,
            model: Union[PreTrainedModel, torch.nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
    ):
        if args is None:
            output_dir = "tmp_trainer"
            logger.info(f"No `NPUTrainingArguments` passed, using `output_dir={output_dir}`.")
            args = NPUTrainingArguments(output_dir=output_dir)

        self.args = args
        # Seed must be set before instantiating the model when using model
        enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
        self.hp_name = None
        self.deepspeed = None
        self.is_in_train = False

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
                raise RuntimeError("`Trainer` requires either a `model` or `model_init` argument")
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
                "https://huggingface.co/docs/transformers/model_doc/auto."
            )

        if hasattr(model, "is_parallelizable") and model.is_parallelizable and model.model_parallel:
            self.is_model_parallel = True
        else:
            self.is_model_parallel = False

        # Setup Sharded DDP training
        self.sharded_ddp = None
        if len(args.sharded_ddp) > 0:
            if args.deepspeed:
                raise ValueError(
                    "Using --sharded_ddp xxx together with --deepspeed is not possible, deactivate one of those flags."
                )
            if len(args.fsdp) > 0:
                raise ValueError(
                    "Using --sharded_ddp xxx together with --fsdp is not possible, deactivate one of those flags."
                )

            if args.local_rank == -1:
                raise ValueError("Using sharded DDP only works in distributed training.")
            elif not is_fairscale_available():
                raise ImportError("Sharded DDP training requires fairscale: `pip install fairscale`.")
            elif ShardedDDPOption.SIMPLE not in args.sharded_ddp and FullyShardedDDP is None:
                raise ImportError(
                    "Sharded DDP in a mode other than simple training requires fairscale version >= 0.3, found "
                    f"{fairscale.__version__}. Upgrade your fairscale library: `pip install --upgrade fairscale`."
                )
            elif ShardedDDPOption.SIMPLE in args.sharded_ddp:
                self.sharded_ddp = ShardedDDPOption.SIMPLE
            elif ShardedDDPOption.ZERO_DP_2 in args.sharded_ddp:
                self.sharded_ddp = ShardedDDPOption.ZERO_DP_2
            elif ShardedDDPOption.ZERO_DP_3 in args.sharded_ddp:
                self.sharded_ddp = ShardedDDPOption.ZERO_DP_3

        self.fsdp = None
        if len(args.fsdp) > 0:
            if args.deepspeed:
                raise ValueError(
                    "Using --fsdp xxx together with --deepspeed is not possible, deactivate one of those flags."
                )
            if args.local_rank == -1:
                raise ValueError("Using fsdp only works in distributed training.")

            # dep_version_check("torch>=1.12.0")
            # Would have to update setup.py with torch>=1.12.0
            # which isn't ideally given that it will force people not using FSDP to also use torch>=1.12.0
            # below is the current alternative.
            if version.parse(version.parse(torch.__version__).base_version) < version.parse("1.12.0"):
                raise ValueError("FSDP requires PyTorch >= 1.12.0")

            from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy

            if FSDPOption.FULL_SHARD in args.fsdp:
                self.fsdp = ShardingStrategy.FULL_SHARD
            elif FSDPOption.SHARD_GRAD_OP in args.fsdp:
                self.fsdp = ShardingStrategy.SHARD_GRAD_OP
            elif FSDPOption.NO_SHARD in args.fsdp:
                self.fsdp = ShardingStrategy.NO_SHARD

        # one place to sort out whether to place the model on device or not
        # postpone switching model to cuda when:
        # 1. MP - since we are trying to fit a much bigger than 1 gpu model
        # 2. fp16-enabled DeepSpeed loads the model in half the size and it doesn't need .to() anyway,
        #    and we only use deepspeed for training at the moment
        # 3. full bf16 or fp16 eval - since the model needs to be cast to the right dtype first
        # 4. Sharded DDP - same as MP
        # 5. FSDP - same as MP
        self.place_model_on_device = args.place_model_on_device
        if (
            self.is_model_parallel
            or args.deepspeed
            or ((args.fp16_full_eval or args.bf16_full_eval) and not args.do_train)
            or (self.sharded_ddp in [ShardedDDPOption.ZERO_DP_2, ShardedDDPOption.ZERO_DP_3])
            or (self.fsdp is not None)
        ):
            self.place_model_on_device = False

        default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        if self.place_model_on_device:
            self._move_model_to_device(model, args.device)

        # Force n_gpu to 1 to avoid DataParallel as MP will manage the GPUs
        if self.is_model_parallel:
            self.args._n_gpu = 1

        # later use `self.model is self.model_wrapped` to check if it's wrapped or not
        self.model_wrapped = model
        self.model = model

        self.compute_metrics = compute_metrics
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self.optimizer, self.lr_scheduler = optimizers
        if model_init is not None and (self.optimizer is not None or self.lr_scheduler is not None):
            raise RuntimeError(
                "Passing a `model_init` is incompatible with providing the `optimizers` argument. "
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )
        if is_torch_tpu_available() and self.optimizer is not None:
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
        if ((self.sharded_ddp is not None) or args.deepspeed or (self.fsdp is not None)) and (
            self.optimizer is not None or self.lr_scheduler is not None
        ):
            raise RuntimeError(
                "Passing `optimizers` is not allowed if Fairscale, Deepspeed or PyTorch FSDP is enabled."
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)

        # Will be set to True by `self._setup_loggers()` on first call to `self.log()`.
        self._loggers_initialized = False

        # Create clone of distant repo and output directory if needed
        if self.args.push_to_hub:
            self.init_git_repo(at_init=True)
            # In case of pull, we need to make sure every process has the latest.
            if is_torch_tpu_available():
                xm.rendezvous("init git repo")
            elif args.local_rank != -1:
                dist.barrier()

        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        if not callable(self.data_collator) and callable(getattr(self.data_collator, "collate_batch", None)):
            raise ValueError("The `data_collator` should be a simple callable (function, class with `__call__`).")

        if args.max_steps > 0:
            logger.info("max_steps is given, it will override any value given in num_train_epochs")

        if train_dataset is not None and not has_length(train_dataset) and args.max_steps <= 0:
            raise ValueError("train_dataset does not implement __len__, max_steps has to be specified")

        if (
            train_dataset is not None
            and isinstance(train_dataset, torch.utils.data.IterableDataset)
            and args.group_by_length
        ):
            raise ValueError("the `--group_by_length` option is only available for `Dataset`, not `IterableDataset")

        self._signature_columns = None

        # Mixed precision setup
        self.use_apex = False
        self.use_cuda_amp = False
        self.use_cpu_amp = False

        # Mixed precision setup for SageMaker Model Parallel
        if is_sagemaker_mp_enabled():
            # BF16 + model parallelism in SageMaker: currently not supported, raise an error
            if args.bf16:
                raise ValueError("SageMaker Model Parallelism does not support BF16 yet. Please use FP16 instead ")

            if IS_SAGEMAKER_MP_POST_1_10:
                # When there's mismatch between SMP config and trainer argument, use SMP config as truth
                if args.fp16 != smp.state.cfg.fp16:
                    logger.warning(
                        f"FP16 provided in SM_HP_MP_PARAMETERS is {smp.state.cfg.fp16},"
                        f"but FP16 provided in trainer argument is {args.fp16},"
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

        # [mark: for npu]
        if args.fp16 or args.bf16:
            if args.half_precision_backend == "auto":
                if is_apex_available():
                    args.half_precision_backend = "apex"
                elif args.device == torch.device("cpu"):
                    if args.fp16:
                        raise ValueError("Tried to use `fp16` but it is not supported on cpu")
                    elif _is_native_cpu_amp_available:
                        args.half_precision_backend = "cpu_amp"
                    else:
                        raise ValueError("Tried to use cpu amp but native cpu amp is not available")
                else:
                    args.half_precision_backend = "cuda_amp"

            logger.info(f"Using {args.half_precision_backend} half precision backend")

        self.do_grad_scaling = False
        if (args.fp16 or args.bf16) and not (args.deepspeed or is_sagemaker_mp_enabled()):
            # deepspeed and SageMaker Model Parallel manage their own half precision
            if args.half_precision_backend == "cuda_amp":
                self.use_cuda_amp = True
                self.amp_dtype = torch.float16 if args.fp16 else torch.bfloat16
                self.do_grad_scaling = True
                if self.sharded_ddp is not None:
                    self.scaler = ShardedGradScaler()
                elif self.fsdp is not None:
                    if self.amp_dtype == torch.float16:
                        from torch.distributed.fsdp.sharded_grad_scaler import (
                            ShardedGradScaler as FSDPShardedGradScaler,
                        )

                        self.scaler = FSDPShardedGradScaler()
                    else:
                        self.do_grad_scaling = False
                        self.use_cuda_amp = False
                        self.amp_dtype = None

                elif is_torch_tpu_available():
                    from torch_xla.amp import GradScaler

                    self.scaler = GradScaler()
                else:
                    self.scaler = torch.cuda.amp.GradScaler()
            elif args.half_precision_backend == "cpu_amp":
                self.use_cpu_amp = True
                self.amp_dtype = torch.bfloat16
            else:
                if not is_apex_available():
                    raise ImportError(
                        "Using FP16 with APEX but APEX is not installed, please refer to"
                        " https://www.github.com/nvidia/apex."
                    )
                self.use_apex = True

        # FP16 + model parallelism in SageMaker: gradient clipping does not work for now so we raise a helpful error.
        if (
            is_sagemaker_mp_enabled()
            and self.use_cuda_amp
            and args.max_grad_norm is not None
            and args.max_grad_norm > 0
        ):
            raise ValueError(
                "SageMaker Model Parallelism in mixed precision mode does not support gradient clipping yet. Pass "
                "along 'max_grad_norm': 0 in your hyperparameters."
            )

        # Label smoothing
        if self.args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
        else:
            self.label_smoother = None

        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
        )

        self.control = TrainerControl()
        # Internal variable to count flos in each process, will be accumulated in `self.state.total_flos` then
        # returned to 0 every time flos need to be logged
        self.current_flos = 0
        self.hp_search_backend = None
        self.use_tune_checkpoints = False
        default_label_names = find_labels(self.model.__class__)
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names
        self.can_return_loss = can_return_loss(self.model.__class__)
        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

        # Internal variables to keep track of the original batch size
        self._train_batch_size = args.train_batch_size

        # very last
        self._memory_tracker.stop_and_update_metrics()

        # torchdynamo
        if args.torchdynamo is not None and not is_torchdynamo_available():
            raise RuntimeError("Using torchdynamo requires a nighly install of PyTorch.")
        
        # =============================
        # Mixed precision setup for NPU
        # =============================
        # Set the correct log level depending on the node
        # Already done in super().init() but we have to do it again
        # because we use optimum.utils.logging here and not
        # transformers.utils.logging
        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)

    def _wrap_model(self, model, training=True, dataloader=None):
        if self.args.torchdynamo is not None:
            import torch._dynamo as dynamo

            model = dynamo.optimize(self.args.torchdynamo)(model)
        if self.args.use_ipex:
            dtype = torch.bfloat16 if self.use_cpu_amp else torch.float32
            model = self.ipex_optimize_model(model, training, dtype=dtype)

        if is_sagemaker_mp_enabled():
            # Wrapping the base model twice in a DistributedModel will raise an error.
            if isinstance(self.model_wrapped, smp.model.DistributedModel):
                return self.model_wrapped
            return smp.DistributedModel(model, backward_passes_per_step=self.args.gradient_accumulation_steps)

        # already initialized its own DDP and AMP
        if self.deepspeed:
            return self.deepspeed

        # train/eval could be run multiple-times - if already wrapped, don't re-wrap it again
        if unwrap_model(model) is not model:
            return model

        # Mixed precision training with apex (torch < 1.6)
        if self.use_apex and training:
            if self.args.loss_scale is None:
                model, self.optimizer = amp.initialize(model, self.optimizer,
                                                       opt_level=self.args.fp16_opt_level,
                                                       combine_grad=self.args.use_combine_grad)
            else:
                model, self.optimizer = amp.initialize(model, self.optimizer,
                                                       opt_level=self.args.fp16_opt_level,
                                                       loss_scale=self.args.loss_scale,
                                                       combine_grad=self.args.use_combine_grad)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = nn.DataParallel(model)

        if self.args.jit_mode_eval:
            model = self.torch_jit_model_eval(model, dataloader, training)

        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        if not training:
            return model

        # Distributed training (should be after apex fp16 initialization)
        if self.sharded_ddp is not None:
            # Sharded DDP!
            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                model = ShardedDDP(model, self.optimizer)
            else:
                mixed_precision = self.args.fp16 or self.args.bf16
                cpu_offload = ShardedDDPOption.OFFLOAD in self.args.sharded_ddp
                zero_3 = self.sharded_ddp == ShardedDDPOption.ZERO_DP_3
                # XXX: Breaking the self.model convention but I see no way around it for now.
                if ShardedDDPOption.AUTO_WRAP in self.args.sharded_ddp:
                    model = auto_wrap(model)
                self.model = model = FullyShardedDDP(
                    model,
                    mixed_precision=mixed_precision,
                    reshard_after_forward=zero_3,
                    cpu_offload=cpu_offload,
                ).to(self.args.device)
        # Distributed training using PyTorch FSDP
        elif self.fsdp is not None:
            # PyTorch FSDP!
            from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
            from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy

            if FSDPOption.OFFLOAD in self.args.fsdp:
                cpu_offload = CPUOffload(offload_params=True)
            else:
                cpu_offload = CPUOffload(offload_params=False)

            auto_wrap_policy = None
            if FSDPOption.AUTO_WRAP in self.args.fsdp:
                if self.args.fsdp_min_num_params > 0:
                    auto_wrap_policy = functools.partial(
                        size_based_auto_wrap_policy, min_num_params=self.args.fsdp_min_num_params
                    )
                elif self.args.fsdp_transformer_layer_cls_to_wrap is not None:
                    transformer_cls_to_wrap = get_module_class_from_name(
                        model, self.args.fsdp_transformer_layer_cls_to_wrap
                    )
                    if transformer_cls_to_wrap is None:
                        raise Exception("Could not find the transformer layer class to wrap in the model.")
                    auto_wrap_policy = functools.partial(
                        transformer_auto_wrap_policy,
                        # Transformer layer class to wrap
                        transformer_layer_cls={transformer_cls_to_wrap},
                    )
            mixed_precision_policy = None
            dtype = None
            if self.args.fp16:
                dtype = torch.float16
            elif self.args.bf16:
                dtype = torch.bfloat16
            if dtype is not None:
                mixed_precision_policy = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
            if type(model) != FSDP:
                # XXX: Breaking the self.model convention but I see no way around it for now.
                self.model = model = FSDP(
                    model,
                    sharding_strategy=self.fsdp,
                    cpu_offload=cpu_offload,
                    auto_wrap_policy=auto_wrap_policy,
                    mixed_precision=mixed_precision_policy,
                )
                if FSDPOption.OFFLOAD not in self.args.fsdp:
                    model.to(self.args.device)
        elif is_sagemaker_dp_enabled():
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[int(os.getenv("SMDATAPARALLEL_LOCAL_RANK"))]
            )
        elif self.args.local_rank != -1:
            kwargs = {}
            if self.args.ddp_find_unused_parameters is not None:
                kwargs["find_unused_parameters"] = self.args.ddp_find_unused_parameters
            elif isinstance(model, PreTrainedModel):
                # find_unused_parameters breaks checkpointing as per
                # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
                kwargs["find_unused_parameters"] = not model.is_gradient_checkpointing
            else:
                kwargs["find_unused_parameters"] = True

            if self.args.ddp_bucket_cap_mb is not None:
                kwargs["bucket_cap_mb"] = self.args.ddp_bucket_cap_mb
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank] if self.args._n_gpu != 0 else None,
                output_device=self.args.local_rank if self.args._n_gpu != 0 else None,
                **kwargs,
            )

        return model

    def train(
            self,
            resume_from_checkpoint: Optional[Union[str, bool]] = None,
            trial: Union["optuna.Trial", Dict[str, Any]] = None,
            ignore_keys_for_eval: Optional[List[str]] = None,
            **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # # the following is a workaround:
        # if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
        #     self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)
        self._train_batch_size = self.args.train_batch_size

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint)

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        inner_training_loop = find_executable_batch_size(
            self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
        )
        return inner_training_loop(
            args=args,
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
        )

    def _inner_training_loop(
            self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                f"args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if (
                        ((step + 1) % args.gradient_accumulation_steps != 0)
                        and args.local_rank != -1
                        and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step = self.training_step(model, inputs)
                else:
                    tr_loss_step = self.training_step(model, inputs)

                if args.logging_nan_inf_filter and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step)):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_optimizer_grad_norm_fused"):  # [for npu]
                            self.optimizer.clip_optimizer_grad_norm_fused(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if args.local_rank != -1:
                dist.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)
