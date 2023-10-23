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
""" Testing suite for the PyTorch Flava model. """

import unittest
import random

from transformers.testing_utils import require_torch, require_vision, torch_device
from transformers.utils import is_torch_available, is_vision_available

if is_torch_available():
    import torch
    from transformers import FlavaForPreTraining


if is_vision_available():
    from PIL import Image
    from transformers import FlavaProcessor

MODEL_NAME_OR_PATH = "facebook/flava-full"

def prepare_img():
    image = Image.open("000000039769.png")
    return image

@require_vision
@require_torch
class FlavaForPreTrainingIntegrationTest(unittest.TestCase):

    def test_inference(self):
        model = FlavaForPreTraining.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)
        processor = FlavaProcessor.from_pretrained(MODEL_NAME_OR_PATH)
        torch.manual_seed(1)
        random.seed(1)

        image = prepare_img()
        inputs = processor(
            text=["a photo of a cat", "a photo of a dog"],
            images=[image, image],
            padding="max_length",
            max_length=77,
            return_tensors="pt",
            return_codebook_pixels=True,
            return_image_mask=True,
        )
        inputs["input_ids_masked"] = inputs["input_ids"].clone()
        inputs["input_ids_masked"][0, 4:6] = 103
        inputs["mlm_labels"] = inputs["input_ids"].clone()
        inputs["mlm_labels"][:, :] = -100
        inputs["mlm_labels"][0, 4:6] = inputs["input_ids"][0, 4:6]
        inputs = inputs.to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        self.assertEqual(
            outputs.contrastive_logits_per_image.shape,
            torch.Size((inputs.pixel_values.shape[0], inputs.input_ids.shape[0])),
        )
        self.assertEqual(
            outputs.contrastive_logits_per_text.shape,
            torch.Size((inputs.input_ids.shape[0], inputs.pixel_values.shape[0])),
        )

        expected_logits = torch.tensor([[16.1291, 8.4033], [16.1291, 8.4033]], device=torch_device)
        self.assertTrue(torch.allclose(outputs.contrastive_logits_per_image, expected_logits, atol=1e-3))
        self.assertAlmostEqual(outputs.loss_info.mmm_text.item(), 1.75533199, places=4)
        self.assertAlmostEqual(outputs.loss_info.mmm_image.item(), 7.0290069, places=4)
        self.assertAlmostEqual(outputs.loss.item(), 11.0626, places=4)

if __name__=='__main__':
    unittest.main(argv=['', 'FlavaForPreTrainingIntegrationTest.test_inference'])
