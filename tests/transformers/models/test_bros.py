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
""" Testing suite for the PyTorch Bros model. """

import unittest

from transformers.testing_utils import require_torch, torch_device
from transformers.utils import is_torch_available

if is_torch_available():
    import torch
    from transformers import (
        BrosModel,
    )

MODEL_NAME_OR_PATH = "jinho8345/bros-base-uncased"

def prepare_bros_batch_inputs():
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

    bbox = torch.tensor(
        [
            [
                [0.0000, 0.0000, 0.0000, 0.0000],
                [0.5223, 0.5590, 0.5787, 0.5720],
                [0.5853, 0.5590, 0.6864, 0.5720],
                [0.5853, 0.5590, 0.6864, 0.5720],
                [0.1234, 0.5700, 0.2192, 0.5840],
                [0.2231, 0.5680, 0.2782, 0.5780],
                [0.2874, 0.5670, 0.3333, 0.5780],
                [0.3425, 0.5640, 0.4344, 0.5750],
                [0.0866, 0.7770, 0.1181, 0.7870],
                [0.1168, 0.7770, 0.1522, 0.7850],
                [0.1535, 0.7750, 0.1864, 0.7850],
                [0.1890, 0.7750, 0.2572, 0.7850],
                [1.0000, 1.0000, 1.0000, 1.0000],
            ],
            [
                [0.0000, 0.0000, 0.0000, 0.0000],
                [0.4396, 0.6720, 0.4659, 0.6850],
                [0.4698, 0.6720, 0.4843, 0.6850],
                [0.1575, 0.6870, 0.2021, 0.6980],
                [0.2047, 0.6870, 0.2730, 0.7000],
                [0.1299, 0.7010, 0.1430, 0.7140],
                [0.1299, 0.7010, 0.1430, 0.7140],
                [0.1562, 0.7010, 0.2441, 0.7120],
                [0.1562, 0.7010, 0.2441, 0.7120],
                [0.2454, 0.7010, 0.3150, 0.7120],
                [0.3176, 0.7010, 0.3320, 0.7110],
                [0.3333, 0.7000, 0.4029, 0.7140],
                [1.0000, 1.0000, 1.0000, 1.0000],
            ],
        ]
    )
    input_ids = torch.tensor(
        [
            [101, 1055, 8910, 1012, 5719, 3296, 5366, 3378, 2146, 2846, 10807, 13494, 102],
            [101, 2112, 1997, 3671, 6364, 1019, 1012, 5057, 1011, 4646, 2030, 2974, 102],
        ]
    )

    return input_ids, bbox, attention_mask

@require_torch
class BrosModelIntegrationTest(unittest.TestCase):

    def test_inference_no_head(self):
        model = BrosModel.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)

        input_ids, bbox, attention_mask = prepare_bros_batch_inputs()

        with torch.no_grad():
            outputs = model(
                input_ids.to(torch_device),
                bbox.to(torch_device),
                attention_mask=attention_mask.to(torch_device),
                return_dict=True,
            )

        expected_shape = torch.Size((2, 13, 768))
        self.assertEqual(outputs.last_hidden_state.shape, expected_shape)

        expected_slice = torch.tensor(
            [[-0.3074, 0.1363, 0.3143], [0.0925, -0.1155, 0.1050], [0.0221, 0.0003, 0.1285]]
        ).to(torch_device)
        torch.set_printoptions(sci_mode=False)

        self.assertTrue(torch.allclose(outputs.last_hidden_state[0, :3, :3], expected_slice, atol=1e-4))

if __name__=='__main__':
    unittest.main(argv=['', 'BrosModelIntegrationTest.test_inference_no_head'])
