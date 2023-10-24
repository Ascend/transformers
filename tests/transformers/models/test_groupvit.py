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
""" Testing suite for the PyTorch GroupVit model. """

import unittest

from transformers.testing_utils import is_torch_available, is_vision_available
from transformers.testing_utils import torch_device, require_torch, require_vision

if is_torch_available():
    import torch

    from transformers import (
        GroupViTModel, CLIPProcessor
    )

if is_vision_available():
    from PIL import Image

MODEL_NAME_OR_PATH = "nvidia/groupvit-gcc-yfcc"

def prepare_img():
    image = Image.open("000000039769.png")
    return image

@require_vision
@require_torch
class GroupViTModelIntegrationTest(unittest.TestCase):
    def test_inference(self):
        model = GroupViTModel.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)
        processor = CLIPProcessor.from_pretrained(MODEL_NAME_OR_PATH)

        image = prepare_img()
        inputs = processor(
            text=["a photo of a cat", "a photo of a dog"], images=image, padding=True, return_tensors="pt"
        )

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        self.assertEqual(
            outputs.logits_per_image.shape,
            torch.Size((inputs.pixel_values.shape[0], inputs.input_ids.shape[0])),
        )
        self.assertEqual(
            outputs.logits_per_text.shape,
            torch.Size((inputs.input_ids.shape[0], inputs.pixel_values.shape[0])),
        )

        expected_logits = torch.tensor([[13.3523, 6.3629]])

        self.assertTrue(torch.allclose(outputs.logits_per_image, expected_logits, atol=1e-3))

if __name__=='__main__':
    unittest.main(argv=['', 'GroupViTModelIntegrationTest.test_inference'])
