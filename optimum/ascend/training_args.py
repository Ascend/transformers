# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Optional, Union
from dataclasses import dataclass, field

import torch

import transformers
from transformers.utils import (
    ExplicitEnum,
    is_accelerate_available,
    is_sagemaker_mp_enabled,
    logging,
    requires_backends,
)
from transformers.file_utils import cached_property
from transformers.training_args import ParallelMode, TrainingArguments
from .npu_utils import is_torch_npu_available


logger = logging.get_logger(__name__)

if is_torch_npu_available():
    import torch_npu  # noqa: F401

if is_accelerate_available():
    from accelerate.state import AcceleratorState, PartialState
    from accelerate.utils import DistributedType


class NPUOptimizerNames(ExplicitEnum):
    """
    NPUOptimizerNames is built on top of the transformers' OptimizerNames to
    store the acceptable string identifiers for optimizers.
    """

    ADAMW_HF = "adamw_hf"
    ADAMW_TORCH = "adamw_torch"
    ADAMW_TORCH_FUSED = "adamw_torch_fused"
    ADAMW_APEX_FUSED_NPU = "adamw_apex_fused_npu"
    ADAFACTOR = "adafactor"
    SGD = "sgd"
    ADAGRAD = "adagrad"


transformers.training_args.OptimizerNames = NPUOptimizerNames


@dataclass
class NPUTrainingArguments(TrainingArguments):
    """
    NPUTrainingArguments is built on top of the tranformers' TrainingArguments
    to enable deployment on Ascend's NPU.
    """

    optim: Union[NPUOptimizerNames, str] = field(
        default="adamw_hf",
        metadata={"help": "The optimzer to use."}
    )

    device_id: int = field(default=0, metadata={"help": "Specify which card to use during single card training"})

    use_combine_grad: bool = field(
        default=False,
        metadata={
            "help": "Whether to use combine_grad option for amp."
        },
    )

    loss_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "Whether to use static loss scale for amp. If `loss_scale` is None, "
                    "it means to use dynamic loss scale."
        },
    )

    def __post_init__(self):
        dummy_fp16 = False
        if self.fp16:
            # Avoiding super().__post_init__() raises ValueError
            dummy_fp16 = True
            self.fp16 = False

        super().__post_init__()

        if dummy_fp16:
            self.fp16 = True
        dummy_fp16 = False

        # if training args is specified, it will override the one specified in the accelerate config
        if self.half_precision_backend != "apex" and len(self.sharded_ddp) == 0:
            mixed_precision_dtype = os.environ.get("ACCELERATE_MIXED_PRECISION", "no")
            if self.fp16:
                mixed_precision_dtype = "fp16"
            elif self.bf16:
                mixed_precision_dtype = "bf16"
            os.environ["ACCELERATE_MIXED_PRECISION"] = mixed_precision_dtype

    @cached_property
    def _setup_devices(self) -> "torch.device":
        requires_backends(self, ["torch"])
        logger.info("PyTorch-Optimum-Ascend: setting up devices")
        if not is_sagemaker_mp_enabled():
            if not is_accelerate_available(min_version="0.20.1"):
                raise ImportError(
                    "Using the `Trainer` with `PyTorch` requires `accelerate>=0.20.1`: Please run `pip install transformers[torch]` or `pip install accelerate -U`"
                )
            AcceleratorState._reset_state(reset_partial_state=True)
        self.distributed_state = None
        if self.no_cuda:
            self.distributed_state = PartialState(cpu=True, backend=self.ddp_backend)
            self._n_gpu = 0
        elif self.deepspeed: # deepspeed is temporarily not supported by Ascend NPU
            pass
        else:
            self.distributed_state = PartialState(backend=self.ddp_backend)
            self._n_gpu = 1
        if not is_sagemaker_mp_enabled():
            device = self.distributed_state.device
            self.local_rank = self.distributed_state.local_process_index
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and self.parallel_mode != ParallelMode.DISTRIBUTED
        ):
            logger.warning(
                "torch.distributed process group is initialized, but parallel_mode != ParallelMode.DISTRIBUTED. "
                "In order to use Torch DDP, launch your script with `python -m torch.distributed.launch`"
            )
        if self.distributed_state.distributed_type == DistributedType.NO:
            if self.no_cuda:
                device = torch.device("cpu")
                self._n_gpu = 0
            else:
                device = torch.device("npu:{}".format(self.device_id) if torch.npu.is_available() else "cpu")
                self._n_gpu = 1
                if device.type == "npu":
                    torch.npu.set_device(device)
                    logger.info("Single Ascend NPU is enabled.")
        return device
