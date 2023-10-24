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
""" Testing suite for the PyTorch CPMAnt model. """

import unittest

from transformers.testing_utils import is_torch_available, require_torch

if is_torch_available():
    import torch

    from transformers import (
        CpmAntForCausalLM,
        CpmAntTokenizer,
    )

MODEL_NAME_OR_PATH = "openbmb/cpm-ant-10b"

@require_torch
class CpmAntForCausalLMlIntegrationTest(unittest.TestCase):
    def test_inference_casual(self):
        texts = "今天天气真好！"
        model = CpmAntForCausalLM.from_pretrained(MODEL_NAME_OR_PATH)
        tokenizer = CpmAntTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
        inputs = tokenizer(texts, return_tensors="pt")
        hidden_states = model(**inputs).logits

        expected_slice = torch.tensor(
            [[[-6.4267, -6.4083, -6.3958], [-5.8802, -5.9447, -5.7811], [-5.3896, -5.4820, -5.4295]]],
        )
        self.assertTrue(torch.allclose(hidden_states[:, :3, :3], expected_slice, atol=1e-2))

if __name__=='__main__':
    unittest.main(argv=['', 'CpmAntForCausalLMlIntegrationTest.test_inference_casual'])
