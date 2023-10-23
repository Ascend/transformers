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
""" Testing suite for the PyTorch Dinov2 model. """

import unittest

from transformers import is_torch_available, is_vision_available
from transformers.file_utils import cached_property
from transformers.testing_utils import require_torch, require_vision, torch_device

if is_torch_available():
    import torch
    from transformers import Dinov2Model

if is_vision_available():
    from PIL import Image
    from transformers import AutoImageProcessor

MODEL_NAME_OR_PATH = "facebook/dinov2-base"

def prepare_img():
    image = Image.open("000000039769.png")
    return image

@require_torch
@require_vision
class Dinov2ModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return AutoImageProcessor.from_pretrained(MODEL_NAME_OR_PATH) if is_vision_available() else None

    def test_inference_no_head(self):
        model = Dinov2Model.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the last hidden states
        expected_shape = torch.Size((1, 257, 768))
        self.assertEqual(outputs.last_hidden_state.shape, expected_shape)

        expected_slice = torch.tensor(
            [[-2.1747, -0.4729, 1.0936], [-3.2780, -0.8269, -0.9210], [-2.9129, 1.1284, -0.7306]],
            device=torch_device,
        )
        self.assertTrue(torch.allclose(outputs.last_hidden_state[0, :3, :3], expected_slice, atol=1e-4))

if __name__=='__main__':
    unittest.main(argv=['', 'Dinov2ModelIntegrationTest.test_inference_no_head'])
