# Copyright 2023 Huawei Technologies Co., Ltd. All rights reserved.
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

import sys
import transformers
import optimum.ascend

all_monkey_paths = [
    ["TrainingArguments", optimum.ascend.NPUTrainingArguments],
    ["Trainer", optimum.ascend.NPUTrainer]
]

def apply_monkey_patches(monkey_patches):
    for k, v in sys.modules.items():
        if "transformers" in k:
            for dest, patch in monkey_patches:
                if getattr(v, dest, None):
                    setattr(v, dest, patch)

# Apply monkey patches
apply_monkey_patches(all_monkey_paths)
