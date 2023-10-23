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
""" Testing suite for the PyTorch Gpt-bigcode model. """

import unittest

from transformers.testing_utils import require_torch, torch_device
from transformers.utils import is_torch_available

if is_torch_available():
    import torch
    from transformers import GPT2TokenizerFast, GPTBigCodeForCausalLM

MODEL_NAME_OR_PATH = "bigcode/gpt_bigcode-santacoder"

@require_torch
class GPTBigCodeModelLanguageGenerationTest(unittest.TestCase):

    def test_generate_batched(self):
        tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME_OR_PATH)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = GPTBigCodeForCausalLM.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)

        inputs = tokenizer(["def print_hello_world():", "def say_hello():"], return_tensors="pt", padding=True).to(
            torch_device
        )
        outputs = model.generate(**inputs)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        expected_output = [
            'def print_hello_world():\n    print("Hello World!")\n\n\ndef print_hello_',
            'def say_hello():\n    print("Hello, World!")\n\n\nsay_hello()',
        ]
        self.assertListEqual(outputs, expected_output)

if __name__=='__main__':
    unittest.main(argv=['', 'GPTBigCodeModelLanguageGenerationTest.test_generate_batched'])
