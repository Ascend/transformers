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

import importlib.util
from functools import lru_cache

@lru_cache(maxsize=128)
def is_torch_npu_available(check_device=True):
    """Checks if `torch_npu` is installed and potentially if a Ascend NPU is in the environment"""
    _torch_available = importlib.util.find_spec("torch") is not None
    if not _torch_available:
        return False
    if importlib.util.find_spec("torch_npu") is not None:
        if check_device:
            # We need to check if `torch.npu.device_count` > 0, will raise a RuntimeError if not
            try:
                import torch
                import torch_npu
                return torch.npu.device_count() > 0
            except RuntimeError:
                return False
        return True
    return False

