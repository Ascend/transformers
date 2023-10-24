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
""" Testing suite for the PyTorch Codegen model. """

import unittest

from transformers.utils import is_torch_available, cached_property
from transformers.testing_utils import torch_device, require_torch

if is_torch_available():
    import torch
    from transformers import CodeGenForCausalLM, AutoTokenizer

MODEL_NAME_OR_PATH = "Salesforce/codegen-350M-mono"

@require_torch
class CodeGenModelLanguageGenerationTest(unittest.TestCase):
    @cached_property
    def cached_tokenizer(self):
        return AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)

    @cached_property
    def cached_model(self):
        return CodeGenForCausalLM.from_pretrained(MODEL_NAME_OR_PATH)

    def test_lm_generate_codegen(self):
        tokenizer = self.cached_tokenizer
        for checkpointing in [True, False]:
            model = self.cached_model

            if checkpointing:
                model.gradient_checkpointing_enable()
            else:
                model.gradient_checkpointing_disable()
            model.to(torch_device)

            inputs = tokenizer("def hello_world():", return_tensors="pt").to(torch_device)
            expected_output = 'def hello_world():\n    print("Hello World")\n\nhello_world()\n\n'

            output_ids = model.generate(**inputs, do_sample=False)
            output_str = tokenizer.batch_decode(output_ids)[0]

            self.assertEqual(output_str, expected_output)

if __name__=='__main__':
    unittest.main(argv=['', 'CodeGenModelLanguageGenerationTest.test_lm_generate_codegen'])
