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
""" Testing suite for the PyTorch Vision_Encoder_Decoder model. """

import unittest

from transformers.utils import is_torch_available, is_vision_available
from transformers.testing_utils import torch_device, require_torch, require_vision

if is_torch_available():
    import torch
    from transformers import VisionTextDualEncoderModel

if is_vision_available():
    import PIL
    from PIL import Image
    from transformers import VisionTextDualEncoderProcessor

MODEL_NAME_OR_PATH = "clip-italian/clip-italian"

@require_vision
@require_torch
class VisionTextDualEncoderIntegrationTest(unittest.TestCase):

    def test_inference(self):
        model = VisionTextDualEncoderModel.from_pretrained(MODEL_NAME_OR_PATH, logit_scale_init_value=1.0).to(torch_device)
        processor = VisionTextDualEncoderProcessor.from_pretrained(MODEL_NAME_OR_PATH)

        image = Image.open("000000039769.png")
        inputs = processor(
            text=["una foto di un gatto", "una foto di un cane"], images=image, padding=True, return_tensors="pt"
        ).to(torch_device)

        outputs = model(**inputs)

        self.assertEqual(outputs.logits_per_image.shape, (inputs.pixel_values.shape[0], inputs.input_ids.shape[0]))
        self.assertEqual(
            outputs.logits_per_text.shape,
            (inputs.input_ids.shape[0], inputs.pixel_values.shape[0]),
        )

        expected_logits = torch.tensor([[1.2284727, 0.3104122]]).to(torch_device)

        self.assertTrue(torch.allclose(outputs.logits_per_image, expected_logits, atol=1e-3))

if __name__=='__main__':
    unittest.main(argv=['', 'VisionTextDualEncoderIntegrationTest.test_inference'])
