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
""" Testing suite for the PyTorch Owlvit model. """

import unittest

from transformers.testing_utils import require_torch, require_vision, torch_device
from transformers.utils import is_torch_available, is_vision_available, cached_property

if is_torch_available():
    import torch
    from transformers import OwlViTModel

if is_vision_available():
    from PIL import Image
    from transformers import OwlViTProcessor

MODEL_NAME_OR_PATH = "google/owlvit-base-patch32"

def prepare_img():
    image = Image.open("000000039769.png")
    return image

@require_vision
@require_torch
class OwlViTModelIntegrationTest(unittest.TestCase):
    def test_inference(self):
        model = OwlViTModel.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)
        processor = OwlViTProcessor.from_pretrained(MODEL_NAME_OR_PATH)

        image = prepare_img()
        inputs = processor(
            text=[["a photo of a cat", "a photo of a dog"]],
            images=image,
            max_length=16,
            padding="max_length",
            return_tensors="pt",
        ).to(torch_device)


        with torch.no_grad():
            outputs = model(**inputs)

        self.assertEqual(
            outputs.logits_per_image.shape,
            torch.Size((inputs.pixel_values.shape[0], inputs.input_ids.shape[0])),
        )
        self.assertEqual(
            outputs.logits_per_text.shape,
            torch.Size((inputs.input_ids.shape[0], inputs.pixel_values.shape[0])),
        )
        expected_logits = torch.tensor([[3.4613, 0.9403]], device=torch_device)
        self.assertTrue(torch.allclose(outputs.logits_per_image, expected_logits, atol=1e-3))

if __name__=='__main__':
    unittest.main(argv=['', 'OwlViTModelIntegrationTest.test_inference'])
