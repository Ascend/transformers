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
""" Testing suite for the PyTorch Conditional-Detr model. """

import unittest
import math

from transformers.testing_utils import require_torch, torch_device, require_vision, require_timm
from transformers.utils import is_torch_available, is_vision_available, cached_property

if is_torch_available():
    import torch
    from transformers import ConditionalDetrModel

if is_vision_available():
    from PIL import Image
    from transformers import ConditionalDetrImageProcessor

MODEL_NAME_OR_PATH = "microsoft/conditional-detr-resnet-50"

def prepare_img():
    image = Image.open("000000039769.png")
    return image

@require_timm
@require_vision
class ConditionalDetrModelIntegrationTests(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return (
            ConditionalDetrImageProcessor.from_pretrained(MODEL_NAME_OR_PATH)
            if is_vision_available()
            else None
        )

    def test_inference_no_head(self):
        model = ConditionalDetrModel.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        encoding = image_processor(images=image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**encoding)

        expected_shape = torch.Size((1, 300, 256))
        self.assertEqual(outputs.last_hidden_state.shape, expected_shape)
        expected_slice = torch.tensor(
            [[0.4222, 0.7471, 0.8760], [0.6395, -0.2729, 0.7127], [-0.3090, 0.7642, 0.9529]]
        ).to(torch_device)
        self.assertTrue(torch.allclose(outputs.last_hidden_state[0, :3, :3], expected_slice, atol=1e-4))

if __name__=='__main__':
    unittest.main(argv=['', 'ConditionalDetrModelIntegrationTests.test_inference_no_head'])
