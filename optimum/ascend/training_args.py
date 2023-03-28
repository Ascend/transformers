import datetime
from dataclasses import asdict, dataclass, field
from typing import Optional

from optimum.utils import logging
from transformers.file_utils import cached_property, is_torch_available, torch_required
from transformers.training_args import TrainingArguments, get_int_from_env

if is_torch_available():
    import torch
    import torch_npu

logger = logging.get_logger(__name__)

# List of arguments that are not supported by optimum-ascend
UNSUPPORTED_ARGUMENTS = [
    "bf16",  # bf16 for CUDA devices
    "bf16_full_eval",  # bf16 for CUDA devices
    "deepspeed",
    "fp16",
    "fp16_backend",
    "fp16_full_eval",
    "fp16_opt_level",
    "half_precision_backend",  # not supported, Ascend Mixed Precision should be used and specified in NPU configuration
    "mp_parameters",
    "sharded_ddp",
    "tf32",
    "tpu_metrics_debug",
    "tpu_num_cores",
]


@dataclass
class NPUTrainingArguments(TrainingArguments):
    """
    NPUTrainingArguments is built on top of the tranformers' TrainingArguments
    to enable deployment on Ascend's NPU.
    """

    use_ascend: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use Ascend's NPU for training the model"}
    )

    device_id: int = field (default=0, metadata={"help": "Specify which card to use during single card training"})

    npu_fp16: bool = field(
        default=False,
        metadata={
            "help": "Whether to use fp16 (mixed) precision instead of 32-bit when training on the NPU"
        },
    )

    npu_fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": (
                "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. "
                "See details at https://nvidia.github.io/apex/amp.html"
            )
        },
    )

    use_npu_adam: bool = field(
        default=False,
        metadata={
            "help": "Wheter use Ascend's custiom AdamW implementation."
        },
    )

    npu_fp16_ops: str = field(
        default=" ",
        metadata={
            "help": "the half precision operators for NPU mixed precision training"
        },
    )

    npu_fp32_ops: str = field(
        default=" ",
        metadata={
            "help": "the float precision operators for NPU mixed precsion training"
        },
    )

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

    use_dynamic_scale: bool = field(
        default=True,
        metadata={
            "help": "whether use dynamic scale."
        },
    )

    distributed_process_group_timeout: Optional[int] = field(
        default=1800,
        metadata={
            "help": "Timeout(seconds) for operations executed against the process group, the value of the flag "
                    "`timeout` passed to `init_process_group`."
        }
    )

    # # Override the default value of epsilon to be consistent with Habana FusedAdamW
    # adam_epsilon: float = field(default=1e-6, metadata={"help": "Epsilon for AdamW optimizer."})

    def __post_init__(self):
        # Raise errors for arguments that are not supported by optimum-ascend
        if self.bf16 or self.bf16_full_eval:
            raise ValueError(
                "--bf16 and --bf16_full_eval are not supported by optimum-ascend."
            )
        if self.tpu_num_cores or self.tpu_metrics_debug:
            raise ValueError("TPUs are not supported by optimum-ascend.")
        if self.deepspeed:
            raise ValueError("--deepspeed is not supported by optimum-ascend.")
        if self.mp_parameters:
            raise ValueError("--mp_parameters is not supported by optimum-ascend.")
        if self.sharded_ddp:
            raise ValueError("--sharded_ddp is not supported by optimum-ascend.")
        if self.tf32:
            raise ValueError("--tf32 is not supported by optimum-ascend.")

        super().__post_init__()

    def __str__(self):
        self_as_dict = asdict(self)

        # Remove deprecated arguments. That code should be removed once
        # those deprecated arguments are removed from TrainingArguments. (TODO: transformers v5)
        del self_as_dict["per_gpu_train_batch_size"]
        del self_as_dict["per_gpu_eval_batch_size"]
        # Remove arguments that are unsupported by optimum-habana
        for key in UNSUPPORTED_ARGUMENTS:
            del self_as_dict[key]

        self_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in self_as_dict.items()}

        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if torch.distributed.is_available() and torch.distributed.is_initialized() and self.local_rank == -1:
            logger.warning(
                "torch.distributed process group is initialized, but local_rank == -1. "
                "In order to use Torch DDP, launch your script with `python -m torch.distributed.launch`"
            )
        device = torch.device("cpu")
        if self.use_ascend:
            logger.info("Ascend is enabled.")
            if self.local_rank == -1:
                # if n_gpu > 1 we'll use nn.DataParallel.
                # If you only want use a specific subset of GPUs use `NPU_VISIBLE_DEVICES=0`
                # Explicitly set NPU to the first (index 0)  NPU device, otherwise `set_device` will
                # trigger an error that a device index is missing. Index 0 takes into account the
                # NPUs available in the environment, so `NPU_VISIBLE_DEVICES=1,2` with `npu:0`
                # will use the first NPU in the env, i.e. NPU#1

                device = torch.device("npu:{}".format(self.device_id) if torch.npu.is_available() else "cpu")
                self._n_gpu = 1  # TODO: NPU_VISIBLE_DEVICES is not equal to CUDA_VISIBLE_DEVICES
            else:
                # Here, we'll use torch.distributed.
                # Initializes the distributed backend which will take care of synchronizing nodes/NPUs
                if not torch.distributed.is_initialized():
                    torch.distributed.init_process_group(backend="hccl", timeout=datetime.timedelta(
                        self.distributed_process_group_timeout))
                device = torch.device("npu", self.local_rank)
                self._n_gpu = 1
                logger.info("Enabled distributed run.")
        else:
            device = torch.device("cpu")
            self._n_gpu = 0
            self.local_rank = get_int_from_env(
                ["LOCAL_RANK", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK"],
                self.local_rank,
            )
            if self.local_rank != -1 and not torch.distributed.is_initialized():
                # Initializes distributed backend for cpu
                if self.xpu_backend not in ("mpi", "ccl", "gloo"):
                    raise ValueError(
                        "CPU distributed training backend is not properly set. "
                        "Please set '--xpu_backend' to either 'mpi' or 'ccl' or 'gloo'."
                    )
                if self.xpu_backend == "ccl":
                    requires_backends(self, "oneccl_bind_pt")
                    if ccl_version >= "1.12":
                        import oneccl_bindings_for_pytorch  # noqa: F401
                    else:
                        import torch_ccl  # noqa: F401
                    if int(os.environ.get("CCL_WORKER_COUNT", 0)) < 1:
                        raise ValueError(
                            "CPU distributed training backend is ccl. but CCL_WORKER_COUNT is not correctly set. "
                            "Please use like 'export CCL_WORKER_COUNT = 1' to set."
                        )

                # Try to get launch configuration from environment variables set by MPI launcher - works for Intel MPI, OpenMPI and MVAPICH
                rank = get_int_from_env(["RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK"], 0)
                size = get_int_from_env(["WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE"], 1)
                local_size = get_int_from_env(
                    ["MPI_LOCALNRANKS", "OMPI_COMM_WORLD_LOCAL_SIZE", "MV2_COMM_WORLD_LOCAL_SIZE"], 1
                )
                os.environ["RANK"] = str(rank)
                os.environ["WORLD_SIZE"] = str(size)
                os.environ["LOCAL_RANK"] = str(self.local_rank)
                if not os.environ.get("MASTER_PORT", None):
                    os.environ["MASTER_PORT"] = "29500"
                if not os.environ.get("MASTER_ADDR", None):
                    if local_size != size or self.xpu_backend != "mpi":
                        raise ValueError(
                            "Looks like distributed multinode run but MASTER_ADDR env not set, "
                            "please try exporting rank 0's hostname as MASTER_ADDR"
                        )
                if (
                        torch.get_num_threads() == 1
                        and get_int_from_env(["OMP_NUM_THREADS", "MKL_NUM_THREADS"], 0) == 0
                        and is_psutil_available()
                ):
                    import psutil

                    num_cpu_threads_per_process = int(psutil.cpu_count(logical=False) / local_size)
                    if num_cpu_threads_per_process == 0:
                        num_cpu_threads_per_process = 1
                    torch.set_num_threads(num_cpu_threads_per_process)
                    logger.info(
                        f"num_cpu_threads_per_process unset, we set it at {num_cpu_threads_per_process} to improve oob"
                        " performance."
                    )
                torch.distributed.init_process_group(
                    backend=self.xpu_backend, rank=rank, world_size=size, timeout=self.ddp_timeout_delta
                )

        if device.type == "npu":
            torch.npu.set_device(device)

        return device
