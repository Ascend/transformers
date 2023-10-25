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
""" Testing suite for the PyTorch QdqBert model. """

import unittest
import pytorch_quantization.nn as quant_nn

from transformers.testing_utils import require_torch, require_pytorch_quantization, torch_device
from transformers.utils import is_torch_available
from pytorch_quantization.tensor_quant import QuantDescriptor

if is_torch_available():
    import torch
    from transformers import QDQBertModel

MODEL_NAME_OR_PATH = "bert-base-uncased"

@require_torch
@require_pytorch_quantization
class QDQBertModelIntegrationTest(unittest.TestCase):

    def test_inference_no_head_absolute_embedding(self):

        input_desc = QuantDescriptor(num_bits=8, calib_method="max")

        weight_desc = QuantDescriptor(num_bits=8, axis=((0,)))
        quant_nn.QuantLinear.set_default_quant_desc_input(input_desc)
        quant_nn.QuantLinear.set_default_quant_desc_weight(weight_desc)

        model = QDQBertModel.from_pretrained(MODEL_NAME_OR_PATH)
        input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = torch.Size((1, 11, 768))
        self.assertEqual(output.shape, expected_shape)
        expected_slice = torch.tensor(
            [[[0.4571, -0.0735, 0.8594], [0.2774, -0.0278, 0.8794], [0.3548, -0.0473, 0.7593]]]
        )
        self.assertTrue(torch.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))

if __name__=='__main__':
    unittest.main(argv=['', 'QDQBertModelIntegrationTest.test_inference_no_head_absolute_embedding'])
