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
""" Testing suite for the PyTorch Xmod model. """

import unittest

from transformers.utils import is_torch_available
from transformers.testing_utils import torch_device, require_torch

if is_torch_available():
    import torch
    from transformers import XmodModel

MODEL_NAME_OR_PATH = "facebook/xmod-base"

@require_torch
class XmodModelIntegrationTest(unittest.TestCase):
    def test_xmod_base(self):
        model = XmodModel.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)

        model.set_default_language("en_XX")
        input_ids = torch.tensor([[0, 581, 10269, 83, 99942, 136, 60742, 23, 70, 80583, 18276, 2]]).to(torch_device)

        expected_output_shape = torch.Size((1, 12, 768))
        expected_output_values_last_dim = torch.tensor(
            [[-0.2394, -0.0036, 0.1252, -0.0087, 0.1325, 0.0580, -0.2049, -0.1978, -0.1223, 0.0648, -0.2599, -0.3724]]
        ).to(torch_device)
        output = model(input_ids)["last_hidden_state"].detach()
        self.assertEqual(output.shape, expected_output_shape)

        self.assertTrue(torch.allclose(output[:, :, -1], expected_output_values_last_dim, atol=1e-3))

        model.set_default_language("de_DE")
        input_ids = torch.tensor([[0, 1310, 49083, 443, 269, 71, 5486, 165, 60429, 660, 23, 2315, 58761, 18391, 5, 2]]).to(torch_device)

        expected_output_shape = torch.Size((1, 16, 768))

        expected_output_values_last_dim = torch.tensor(
            [[0.0162, 0.0075, -0.1882, 0.2335, -0.0952, -0.3994, -0.0317, -0.1174, 0.0177, 0.4280, -0.0240, -0.2138,
              0.0785, -0.1045, -0.2811, -0.3220]]
        ).to(torch_device)

        output = model(input_ids)["last_hidden_state"].detach()
        self.assertEqual(output.shape, expected_output_shape)

        self.assertTrue(torch.allclose(output[:, :, -1], expected_output_values_last_dim, atol=1e-3))

if __name__=='__main__':
    unittest.main(argv=['', 'XmodModelIntegrationTest.test_xmod_base'])
