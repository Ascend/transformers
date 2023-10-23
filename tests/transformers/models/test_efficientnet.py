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
""" Testing suite for the PyTorch Efficientnet model. """

import unittest

from transformers.testing_utils import require_torch, require_vision, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available

if is_torch_available():
    import torch
    from transformers import EfficientNetForImageClassification


if is_vision_available():
    from PIL import Image
    from transformers import AutoImageProcessor

MODEL_NAME_OR_PATH = "google/efficientnet-b7"

def prepare_img():
    image = Image.open("000000039769.png")
    return image

@require_torch
@require_vision
class EfficientNetModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return AutoImageProcessor.from_pretrained(MODEL_NAME_OR_PATH) if is_vision_available() else None


    def test_inference_image_classification_head(self):

        model = EfficientNetForImageClassification.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor([-0.2962, 0.4487, 0.4499]).to(torch_device)
        self.assertTrue(torch.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4))

if __name__=='__main__':
    unittest.main(argv=['', 'EfficientNetModelIntegrationTest.test_inference_image_classification_head'])
