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

import collections.abc
import numpy as np
import torch
from transformers.utils import is_apex_available, ModelOutput


if is_apex_available():
    import apex

    container_abcs = collections.abc


    def applier_npu(value, fn):
        if isinstance(value, torch.Tensor):
            return fn(value)
        elif isinstance(value, str):
            return value
        elif isinstance(value, np.ndarray):
            return value
        elif hasattr(value, "to"):  # Allow handling of custom batch classes
            return fn(value)
        elif isinstance(value, container_abcs.Mapping):
            if isinstance(value, ModelOutput):
                for k in value:
                    value[k] = applier_npu(value[k], fn)
                return value
            else:
                return {applier_npu(k, fn): applier_npu(v, fn) for k, v in value.items()}
        elif isinstance(value, container_abcs.Iterable):
            return type(value)(applier_npu(v, fn) for v in value)
        else:
            # Do I want this to fire off even if someone chooses to pass something ordinary like
            # an int or float?  May be more annoying than it's worth.
            # print("Warning:  unrecognized type in applier.  If your input data is a custom class, "
            #     "provide it with a .to(dtype) method which converts its floating-point Tensors to dtype. "
            #     "Amp will check for your custom to() and invoke it to cast the batch's "
            #     "floating-point Tensors to the appropriate type. "
            #     "Also, if your data is a custom class, it is your responsibility to ensure that "
            #     "any Tensors you want to be cuda are already cuda."
            return value


def patch_bugfix_apex():
    if is_apex_available():
        apex.amp._initialize.applier = applier_npu
