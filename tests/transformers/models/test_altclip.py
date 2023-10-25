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
""" Testing suite for the PyTorch AltCLIP model. """

import unittest

from transformers.utils import is_torch_available, is_vision_available
from transformers.testing_utils import torch_device, require_torch, require_vision

if is_torch_available():
    import torch
    from transformers import AltCLIPModel, AltCLIPProcessor

if is_vision_available():
    from PIL import Image

MODEL_NAME_OR_PATH = "BAAI/AltCLIP"


def prepare_img():
    image = Image.open("000000039769.png")
    return image

@require_vision
@require_torch
class AltCLIPModelIntegrationTest(unittest.TestCase):
    def test_inference(self):
        model = AltCLIPModel.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)
        processor = AltCLIPProcessor.from_pretrained(MODEL_NAME_OR_PATH)

        image = prepare_img()
        inputs = processor(text=["一张猫的照片", "一张狗的照片"], images=image, padding=True, return_tensors="pt").to(torch_device)

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

        probs = outputs.logits_per_image.softmax(dim=1)
        expected_probs = torch.tensor([[9.9942e-01, 5.7805e-04]], device=torch_device)

        self.assertTrue(torch.allclose(probs, expected_probs, atol=5e-3))

if __name__=='__main__':
    unittest.main(argv=['', 'AltCLIPModelIntegrationTest.test_inference'])
