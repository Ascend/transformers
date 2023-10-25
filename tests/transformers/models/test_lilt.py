# coding=utf-8
# Copyright 2023-present Huawei Technologies Co., Ltd. All rights reserved.
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch Lilt model. """

import unittest

from transformers.testing_utils import require_torch, torch_device
from transformers.utils import is_torch_available

if is_torch_available():
    import torch
    from transformers import LiltModel

MODEL_NAME_OR_PATH = "SCUT-DLVCLab/lilt-roberta-en-base"

@require_torch
class LiltModelIntegrationTest(unittest.TestCase):
    def test_inference_no_head(self):
        model = LiltModel.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)

        input_ids = torch.tensor([[1, 2]], device=torch_device)
        bbox = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]]], device=torch_device)


        with torch.no_grad():
            outputs = model(input_ids=input_ids, bbox=bbox)

        expected_shape = torch.Size([1, 2, 768])
        expected_slice = torch.tensor(
            [[-0.0653, 0.0950, -0.0061], [-0.0545, 0.0926, -0.0324]],
            device=torch_device,
        )

        self.assertTrue(outputs.last_hidden_state.shape, expected_shape)
        self.assertTrue(torch.allclose(outputs.last_hidden_state[0, :, :3], expected_slice, atol=1e-3))

if __name__=='__main__':
    unittest.main(argv=['', 'LiltModelIntegrationTest.test_inference_no_head'])
