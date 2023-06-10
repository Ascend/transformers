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

from typing import Optional, Union
from dataclasses import dataclass, field

import torch
import torch_npu

import transformers
from transformers.utils import ExplicitEnum, logging
from transformers.file_utils import cached_property
from transformers.training_args import TrainingArguments
from .npu_utils import is_torch_npu_available


logger = logging.get_logger(__name__)


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

    @cached_property
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch-Optimum-Ascend: setting up devices")
        if torch.distributed.is_available() and torch.distributed.is_initialized() and self.local_rank == -1:
            logger.warning(
                "torch.distributed process group is initialized, but local_rank == -1. "
                "In order to use Torch DDP, launch your script with `python -m torch.distributed.launch`"
            )
        if is_torch_npu_available():
            if self.local_rank == -1:
                device = torch.device("npu:{}".format(self.device_id) if torch.npu.is_available() else "cpu")
                self._n_gpu = 1
            else:
                # Here, we'll use torch.distributed.
                # Initializes the distributed backend which will take care of synchronizing nodes/NPUs
                if not torch.distributed.is_initialized():
                    torch.distributed.init_process_group(backend="hccl")
                device = torch.device("npu", self.local_rank)
                self._n_gpu = 1
        else:
            logger.warning("Ascend NPU is not available, will use CPU instead.")
            self.no_cuda = True
            super()._setup_devices()

        if device.type == "npu":
            torch.npu.set_device(device)
            logger.info("Ascend NPU is enabled.")

        return device
