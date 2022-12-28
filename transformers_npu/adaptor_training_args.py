import os
import sys
import datetime

# INPLACE.1: transformers.training_args.OptimizerNames
import transformers
from transformers.utils import ExplicitEnum


class OptimizerNamesNPU(ExplicitEnum):
    """
    Stores the acceptable string identifiers for optimizers.
    """

    ADAMW_HF = "adamw_hf"
    ADAMW_TORCH = "adamw_torch"
    ADAMW_TORCH_XLA = "adamw_torch_xla"
    ADAMW_APEX_FUSED = "adamw_apex_fused"
    ADAMW_APEX_FUSED_NPU = "adamw_apex_fused_npu"  # [for npu]
    ADAFACTOR = "adafactor"


transformers.training_args.OptimizerNames = OptimizerNamesNPU

# INPLACE.2: transformers.training_args.TrainingArguments
from dataclasses import (
    dataclass,
    field,
)
from typing import Optional
from transformers.utils import (
    cached_property,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_available,
    is_torch_tpu_available,
    torch_required,
    logging,
)
from transformers.trainer_utils import (
    EvaluationStrategy,
    HubStrategy,
    IntervalStrategy,
    ShardedDDPOption,
    SchedulerType,
)
from transformers.training_args import (
    OptimizerNames,
    TrainingArguments,
)
from .adaptor_utils import is_torch_npu_available

if is_torch_available():
    import torch
    import torch.distributed as dist

if is_torch_npu_available():
    import torch_npu

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

    smp.init()

logger = transformers.training_args.logger
trainer_log_levels = transformers.training_args.trainer_log_levels
default_logdir = transformers.training_args.default_logdir


@dataclass
class TrainingArgumentsNPU(TrainingArguments):
    # [for npu]
    loss_scale: float = field(
        default=1024.0,
        metadata={
            "help": "loss scale for amp."
        },
    )
    use_combine_grad: bool = field(
        default=False,
        metadata={
            "help": "whether use combine_grad option for amp."
        },
    )
    use_combine_ddp: bool = field(
        default=False,
        metadata={
            "help": "whether use combine_grad_ option for amp."
        },
    )
    half_precision_operators: str = field(
        default=" ",
        metadata={"help": "The half_precision_operators."},
    )
    float_precision_operators: str = field(
        default=" ",
        metadata={"help": "The float_precision_operators."},
    )
    use_dynamic_scale: bool = field(
        default=True,
        metadata={"help": "whether use dynamic_scale."},
    )
    distributed_process_group_timeout: Optional[int] = field(
        default=1800,
        metadata={
            "help": "Timeout(seconds) for operations executed against the process group, the value of the flag "
                    "`timeout` passed to `init_process_group`."
        }
    )
    optim: OptimizerNamesNPU = field(  # [for npu]
        default="adamw_hf",
        metadata={"help": "The optimizer to use."},
    )

    def __post_init__(self):
        # Handle --use_env option in torch.distributed.launch (local_rank not passed as an arg then).
        # This needs to happen before any call to self.device or self.n_gpu.
        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if env_local_rank != -1 and env_local_rank != self.local_rank:
            self.local_rank = env_local_rank

        # convert to int
        self.log_level = trainer_log_levels[self.log_level]
        self.log_level_replica = trainer_log_levels[self.log_level_replica]

        # expand paths, if not os.makedirs("~/bar") will make directory
        # in the current directory instead of the actual home
        #  see https://github.com/huggingface/transformers/issues/10628
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)
        if self.logging_dir is None and self.output_dir is not None:
            self.logging_dir = os.path.join(self.output_dir, default_logdir())
        if self.logging_dir is not None:
            self.logging_dir = os.path.expanduser(self.logging_dir)

        if self.disable_tqdm is None:
            self.disable_tqdm = logger.getEffectiveLevel() > logging.WARN

        if isinstance(self.evaluation_strategy, EvaluationStrategy):
            warnings.warn(
                "using `EvaluationStrategy` for `evaluation_strategy` is deprecated and will be removed in version 5 of 🤗 Transformers. Use `IntervalStrategy` instead",
                FutureWarning,
            )
            # Go back to the underlying string or we won't be able to instantiate `IntervalStrategy` on it.
            self.evaluation_strategy = self.evaluation_strategy.value

        self.evaluation_strategy = IntervalStrategy(self.evaluation_strategy)
        self.logging_strategy = IntervalStrategy(self.logging_strategy)
        self.save_strategy = IntervalStrategy(self.save_strategy)
        self.hub_strategy = HubStrategy(self.hub_strategy)

        self.lr_scheduler_type = SchedulerType(self.lr_scheduler_type)
        if self.do_eval is False and self.evaluation_strategy != IntervalStrategy.NO:
            self.do_eval = True

        # eval_steps has to be defined and non-zero, fallbacks to logging_steps if the latter is non-zero
        if self.evaluation_strategy == IntervalStrategy.STEPS and (self.eval_steps is None or self.eval_steps == 0):
            if self.logging_steps > 0:
                logger.info(f"using `logging_steps` to initialize `eval_steps` to {self.logging_steps}")
                self.eval_steps = self.logging_steps
            else:
                raise ValueError(
                    f"evaluation strategy {self.evaluation_strategy} requires either non-zero --eval_steps or --logging_steps"
                )

        # logging_steps must be non-zero for logging_strategy that is other than 'no'
        if self.logging_strategy == IntervalStrategy.STEPS and self.logging_steps == 0:
            raise ValueError(f"logging strategy {self.logging_strategy} requires non-zero --logging_steps")

        # Sanity checks for load_best_model_at_end: we require save and eval strategies to be compatible.
        if self.load_best_model_at_end:
            if self.evaluation_strategy != self.save_strategy:
                raise ValueError(
                    "--load_best_model_at_end requires the save and eval strategy to match, but found\n- Evaluation "
                    f"strategy: {self.evaluation_strategy}\n- Save strategy: {self.save_strategy}"
                )
            if self.evaluation_strategy == IntervalStrategy.STEPS and self.save_steps % self.eval_steps != 0:
                raise ValueError(
                    "--load_best_model_at_end requires the saving steps to be a round multiple of the evaluation "
                    f"steps, but found {self.save_steps}, which is not a round multiple of {self.eval_steps}."
                )

        if self.load_best_model_at_end and self.metric_for_best_model is None:
            self.metric_for_best_model = "loss"
        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = self.metric_for_best_model not in ["loss", "eval_loss"]
        if self.run_name is None:
            self.run_name = self.output_dir

        if self.fp16_backend and self.fp16_backend != "auto":
            warnings.warn(
                "`fp16_backend` is deprecated and will be removed in version 5 of 🤗 Transformers. Use `half_precision_backend` instead",
                FutureWarning,
            )
            self.half_precision_backend = self.fp16_backend

        if (self.bf16 or self.bf16_full_eval) and not is_torch_bf16_available():
            raise ValueError("Your setup doesn't support bf16. You need Ampere GPU, torch>=1.10, cuda>=11.0")

        if self.fp16 and self.bf16:
            raise ValueError("At most one of fp16 and bf16 can be True, but not both")
        if self.bf16:
            if self.half_precision_backend == "apex":
                raise ValueError(
                    " `--half_precision_backend apex`: bf16 is not supported by apex. Use `--half_precision_backend amp` instead"
                )
            if not (self.sharded_ddp == "" or not self.sharded_ddp):
                raise ValueError("sharded_ddp is not supported with bf16")

        self.optim = OptimizerNames(self.optim)
        if self.adafactor:
            warnings.warn(
                "`--adafactor` is deprecated and will be removed in version 5 of 🤗 Transformers. Use `--optim adafactor` instead",
                FutureWarning,
            )
            self.optim = OptimizerNames.ADAFACTOR

        if (
            is_torch_available()
            and (self.device.type != "npu" and self.device.type != "cuda")  # [for npu]
            and not (self.device.type == "xla" and "GPU_NUM_DEVICES" in os.environ)
            and (self.fp16 or self.fp16_full_eval or self.bf16 or self.bf16_full_eval)
        ):
            raise ValueError(
                "Mixed precision training with AMP or APEX (`--fp16` or `--bf16`) and half precision evaluation (`--fp16_full_eval` or `--bf16_full_eval`) can only be used on CUDA or NPU devices."
            )  # [for npu]

        if is_torch_available() and self.tf32 is not None:
            if self.tf32:
                if is_torch_tf32_available():
                    torch.backends.cuda.matmul.allow_tf32 = True
                else:
                    raise ValueError("--tf32 requires Ampere or a newer GPU arch, cuda>=11 and torch>=1.7")
            else:
                if is_torch_tf32_available():
                    torch.backends.cuda.matmul.allow_tf32 = False
                # no need to assert on else

        if self.report_to is None:
            logger.info(
                "The default value for the training argument `--report_to` will change in v5 (from all installed "
                "integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as "
                "now. You should start updating your code and make this info disappear :-)."
            )
            self.report_to = "all"
        if self.report_to == "all" or self.report_to == ["all"]:
            # Import at runtime to avoid a circular import.
            from transformers.integrations import get_available_reporting_integrations

            self.report_to = get_available_reporting_integrations()
        elif self.report_to == "none" or self.report_to == ["none"]:
            self.report_to = []
        elif not isinstance(self.report_to, list):
            self.report_to = [self.report_to]

        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            raise ValueError("warmup_ratio must lie in range [0,1]")
        elif self.warmup_ratio > 0 and self.warmup_steps > 0:
            logger.info(
                "Both warmup_ratio and warmup_steps given, warmup_steps will override any effect of warmup_ratio during training"
            )

        if isinstance(self.sharded_ddp, bool):
            self.sharded_ddp = "simple" if self.sharded_ddp else ""
        if isinstance(self.sharded_ddp, str):
            self.sharded_ddp = [ShardedDDPOption(s) for s in self.sharded_ddp.split()]
        if self.sharded_ddp == [ShardedDDPOption.OFFLOAD]:
            raise ValueError(
                "`--sharded_ddp offload` can't work on its own. It needs to be added to `--sharded_ddp zero_dp_2` or "
                '`--sharded_ddp zero_dp_3`. For example, `--sharded_ddp "zero_dp_2 offload"`.'
            )
        elif len(self.sharded_ddp) > 1 and ShardedDDPOption.SIMPLE in self.sharded_ddp:
            raise ValueError("`--sharded_ddp simple` is not compatible with any other option.")
        elif ShardedDDPOption.ZERO_DP_2 in self.sharded_ddp and ShardedDDPOption.ZERO_DP_3 in self.sharded_ddp:
            raise ValueError("`--sharded_ddp zero_dp_2` is not compatible with `--sharded_ddp zero_dp_3`.")

        if self.tpu_metrics_debug:
            warnings.warn(
                "using `--tpu_metrics_debug` is deprecated and will be removed in version 5 of 🤗 Transformers. Use `--debug tpu_metrics_debug` instead",
                FutureWarning,
            )
            self.debug += " tpu_metrics_debug"
            self.tpu_metrics_debug = False
        if isinstance(self.debug, str):
            self.debug = [DebugOption(s) for s in self.debug.split()]

        if self.deepspeed:
            # - must be run very last in arg parsing, since it will use a lot of these settings.
            # - must be run before the model is created.
            from transformers.deepspeed import HfTrainerDeepSpeedConfig

            # will be used later by the Trainer
            # note: leave self.deepspeed unmodified in case a user relies on it not to be modified)
            self.hf_deepspeed_config = HfTrainerDeepSpeedConfig(self.deepspeed)
            self.hf_deepspeed_config.trainer_config_process(self)

        if self.push_to_hub_token is not None:
            warnings.warn(
                "`--push_to_hub_token` is deprecated and will be removed in version 5 of 🤗 Transformers. Use "
                "`--hub_token` instead.",
                FutureWarning,
            )
            self.hub_token = self.push_to_hub_token

        if self.push_to_hub_model_id is not None:
            self.hub_model_id = get_full_repo_name(
                self.push_to_hub_model_id, organization=self.push_to_hub_organization, token=self.hub_token
            )
            if self.push_to_hub_organization is not None:
                warnings.warn(
                    "`--push_to_hub_model_id` and `--push_to_hub_organization` are deprecated and will be removed in "
                    "version 5 of 🤗 Transformers. Use `--hub_model_id` instead and pass the full repo name to this "
                    f"argument (in this case {self.hub_model_id}).",
                    FutureWarning,
                )
            else:
                warnings.warn(
                    "`--push_to_hub_model_id` is deprecated and will be removed in version 5 of 🤗 Transformers. Use "
                    "`--hub_model_id` instead and pass the full repo name to this argument (in this case "
                    f"{self.hub_model_id}).",
                    FutureWarning,
                )
        elif self.push_to_hub_organization is not None:
            self.hub_model_id = f"{self.push_to_hub_organization}/{Path(self.output_dir).name}"
            warnings.warn(
                "`--push_to_hub_organization` is deprecated and will be removed in version 5 of 🤗 Transformers. Use "
                "`--hub_model_id` instead and pass the full repo name to this argument (in this case "
                f"{self.hub_model_id}).",
                FutureWarning,
            )

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if torch.distributed.is_initialized() and self.local_rank == -1:
            logger.warning(
                "torch.distributed process group is initialized, but local_rank == -1. "
                "In order to use Torch DDP, launch your script with `python -m torch.distributed.launch"
            )
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
            if self.local_rank != -1 and not torch.distributed.is_initialized():
                # Initializes distributed backend for cpu
                if self.xpu_backend not in ("mpi", "ccl"):
                    raise ValueError(
                        "CPU distributed training backend is not properly set. "
                        "Please set '--xpu_backend' to either 'mpi' or 'ccl'."
                    )
                torch.distributed.init_process_group(backend=self.xpu_backend)
        elif is_torch_tpu_available():
            device = xm.xla_device()
            self._n_gpu = 0
        elif is_sagemaker_mp_enabled():
            local_rank = smp.local_rank()
            device = torch.device("cuda", local_rank)
            self._n_gpu = 1
        elif is_sagemaker_dp_enabled():
            dist.init_process_group(backend="smddp")
            self.local_rank = int(os.getenv("SMDATAPARALLEL_LOCAL_RANK"))
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1
        elif self.deepspeed:
            # deepspeed inits torch.distributed internally
            from transformers.deepspeed import is_deepspeed_available  # [for npu]

            if not is_deepspeed_available():
                raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
            import deepspeed

            deepspeed.init_distributed()

            # workaround for setups like notebooks where the launcher can't be used,
            # but deepspeed requires a dist env.
            # env LOCAL_RANK could be set manually by the user, or via init_distributed if mpi4py is installed
            self.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            if is_torch_npu_available():  # [for npu]
                device = torch.device("npu:0")  # [for npu]
            else:  # [for npu]
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # [for npu]
            # Sometimes the line in the postinit has not been run before we end up here, so just checking we're
            # not at the default value.
            self._n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(
                    backend="hccl" if is_torch_npu_available() else "nccl",
                    timeout=datetime.timedelta(seconds=self.distributed_process_group_timeout),
                )  # [for npu]
            device = torch.device("npu", self.local_rank) if is_torch_npu_available() else \
                torch.device("cuda", self.local_rank)  # [for npu]
            self._n_gpu = 1

        if device.type == "npu":  # [for npu]
            torch.npu.set_device(device)  # [for npu]
        elif device.type == "cuda":  # [for npu]
            torch.cuda.set_device(device)  # [for npu]

        return device


for k, v in sys.modules.items():
    if "transformers" in k and getattr(v, "TrainingArguments", None):
        setattr(v, "TrainingArguments", TrainingArgumentsNPU)
