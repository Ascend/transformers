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

import sys

from optimum.ascend.testing_utils import require_torch_npu
from transformers.testing_utils import (
    TestCasePlus,
    execute_subprocess_async,
    get_torch_dist_unique_port,
)

class TestTrainerDistributedNPU(TestCasePlus):
    @require_torch_npu
    def test_trainer(self):
        distributed_args = f"""
            -m torch.distributed.launch
            --nproc_per_node=2
            --master_port={get_torch_dist_unique_port()}
            {self.test_file_dir}/test_trainer_distributed.py
        """.split()
        output_dir = self.get_auto_remove_tmp_dir()
        args = f"--output_dir {output_dir}".split()
        cmd = [sys.executable] + distributed_args + args
        execute_subprocess_async(cmd, env=self.get_env())
        # successful return here == success - any error would have caused an error in the sub-call
