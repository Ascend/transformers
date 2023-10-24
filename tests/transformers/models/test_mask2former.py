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
""" Testing suite for the PyTorch Mask2Former model. """

import unittest

from transformers.testing_utils import require_torch, require_vision, torch_device
from transformers.utils import is_torch_available, is_vision_available, cached_property

if is_torch_available():
    import torch
    from transformers import Mask2FormerModel

if is_vision_available():
    from PIL import Image
    from transformers import Mask2FormerImageProcessor

MODEL_NAME_OR_PATH = "facebook/mask2former-swin-small-coco-instance"

def prepare_img():
    image = Image.open("000000039769.png")
    return image

TOLERANCE = 1e-4

@require_vision
class Mask2FormerModelIntegrationTest(unittest.TestCase):

    @cached_property
    def default_image_processor(self):
        return Mask2FormerImageProcessor.from_pretrained(MODEL_NAME_OR_PATH) if is_vision_available() else None

    def test_inference_no_head(self):
        model = Mask2FormerModel.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)
        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(image, return_tensors="pt").to(torch_device)
        inputs_shape = inputs["pixel_values"].shape
        # check size is divisible by 32
        self.assertTrue((inputs_shape[-1] % 32) == 0 and (inputs_shape[-2] % 32) == 0)
        # check size
        self.assertEqual(inputs_shape, (1, 3, 384, 384))

        with torch.no_grad():
            outputs = model(**inputs)

        expected_slice_hidden_state = torch.tensor(
            [[-0.2790, -1.0717, -1.1668], [-0.5128, -0.3128, -0.4987], [-0.5832, 0.1971, -0.0197]]
        ).to(torch_device)
        self.assertTrue(
            torch.allclose(
                outputs.encoder_last_hidden_state[0, 0, :3, :3], expected_slice_hidden_state, atol=TOLERANCE
            )
        )

        expected_slice_hidden_state = torch.tensor(
            [[0.8973, 1.1847, 1.1776], [1.1934, 1.5040, 1.5128], [1.1153, 1.4486, 1.4951]]
        ).to(torch_device)
        self.assertTrue(
            torch.allclose(
                outputs.pixel_decoder_last_hidden_state[0, 0, :3, :3], expected_slice_hidden_state, atol=TOLERANCE
            )
        )

        expected_slice_hidden_state = torch.tensor(
            [[2.1152, 1.7000, -0.8603], [1.5808, 1.8004, -0.9353], [1.6043, 1.7495, -0.5999]]
        ).to(torch_device)
        self.assertTrue(
            torch.allclose(
                outputs.transformer_decoder_last_hidden_state[0, :3, :3], expected_slice_hidden_state, atol=TOLERANCE
            )
        )

if __name__=='__main__':
    unittest.main(argv=['', 'Mask2FormerModelIntegrationTest.test_inference_no_head'])
