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
""" Testing suite for the PyTorch Mistral model. """

import unittest

from transformers.utils import is_torch_available
from transformers.testing_utils import require_torch

if is_torch_available():
    import torch
    from transformers import MistralForCausalLM


MODEL_NAME_OR_PATH = "mistralai/Mistral-7B-v0.1"

@require_torch
class MistralIntegrationTest(unittest.TestCase):

    def test_model_7b_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = MistralForCausalLM.from_pretrained(MODEL_NAME_OR_PATH, device_map="auto")
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
        out = model(input_ids).logits.cpu()

        EXPECTED_MEAN = torch.tensor([[-2.5548, -2.5737, -3.0600, -2.5906, -2.8478, -2.8118, -2.9325, -2.7694]])
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)

        EXPECTED_SLICE = torch.tensor([-5.8781, -5.8616, -0.1052, -4.7200, -5.8781, -5.8774, -5.8773, -5.8777, -5.8781, -5.8780, -5.8781, -5.8779, -1.0787,  1.7583, -5.8779, -5.8780, -5.8783, -5.8778, -5.8776, -5.8781, -5.8784, -5.8778, -5.8778, -5.8777, -5.8779, -5.8778, -5.8776, -5.8780, -5.8779, -5.8781])

        print(out[0, 0, :30])
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, atol=1e-4, rtol=1e-4)


if __name__=='__main__':
    unittest.main(argv=['', 'MistralIntegrationTest.test_model_7b_logits'])
