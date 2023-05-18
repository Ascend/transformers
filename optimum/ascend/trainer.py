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

from torch import nn

from .training_args import NPUTrainingArguments
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.utils import is_apex_available
from transformers import Trainer


if is_apex_available():
    from apex import amp


class NPUTrainer(Trainer):
    """
    NPUTrainer is built on top of the tranformers' Trainer to enable
    deployment on Ascend's NPU.
    """

    def _wrap_model(self, model, training=True, dataloader=None):
        if self.args.torchdynamo is not None:
            import torch._dynamo as dynamo

            model = dynamo.optimize(self.args.torchdynamo)(model)
        if self.args.use_ipex:
            dtype = torch.bfloat16 if self.use_cpu_amp else torch.float32
            model = self.ipex_optimize_model(model, training, dtype=dtype)

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

        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        if not training:
            return model

        # Distributed training using PyTorch FSDP
        if self.fsdp is not None:
            # PyTorch FSDP!
            pass # PyTorch FSDP is not supported by Ascend NPU, will come soon
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