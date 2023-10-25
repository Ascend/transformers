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
""" Testing suite for the PyTorch Visual_Bert model. """

import unittest

from transformers.utils import is_torch_available
from transformers.testing_utils import torch_device, require_torch

if is_torch_available():
    import torch
    from transformers import VisualBertForMultipleChoice

MODEL_NAME_OR_PATH = "uclanlp/visualbert-vcr"

@require_torch
class VisualBertModelIntegrationTest(unittest.TestCase):
    def test_inference_vcr(self):
        model = VisualBertForMultipleChoice.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)

        input_ids = torch.tensor([[[1, 2, 3, 4, 5, 6] for i in range(4)]], dtype=torch.long).to(torch_device)
        attention_mask = torch.ones_like(input_ids)
        token_type_ids = torch.ones_like(input_ids)

        visual_embeds = torch.ones(size=(1, 4, 10, 512), dtype=torch.float32) * 0.5
        visual_embeds = visual_embeds.to(torch_device)
        visual_token_type_ids = torch.ones(size=(1, 4, 10), dtype=torch.long).to(torch_device)
        visual_attention_mask = torch.ones_like(visual_token_type_ids).to(torch_device)

        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                visual_embeds=visual_embeds,
                visual_attention_mask=visual_attention_mask,
                visual_token_type_ids=visual_token_type_ids,
            )

        expected_shape = torch.Size((1, 4))
        self.assertEqual(output.logits.shape, expected_shape)

        expected_slice = torch.tensor([[-7.7697, -7.7697, -7.7697, -7.7697]]).to(torch_device)

        self.assertTrue(torch.allclose(output.logits, expected_slice, atol=1e-4))

if __name__=='__main__':
    unittest.main(argv=['', 'VisualBertModelIntegrationTest.test_inference_vcr'])
