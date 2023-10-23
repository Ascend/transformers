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
""" Testing suite for the PyTorch Clipseg model. """

import unittest

from transformers import CLIPSegProcessor
from transformers.testing_utils import (
    require_torch,
    require_vision,
    torch_device,
)
from transformers.utils import is_torch_available, is_vision_available

if is_torch_available():
    import torch
    from transformers import CLIPSegForImageSegmentation

if is_vision_available():
    from PIL import Image

MODEL_NAME_OR_PATH = "CIDAS/clipseg-rd64-refined"

def prepare_img():
    image = Image.open("000000039769.png")
    return image

@require_vision
@require_torch
class CLIPSegModelIntegrationTest(unittest.TestCase):

    def test_inference_image_segmentation(self):
        processor = CLIPSegProcessor.from_pretrained(MODEL_NAME_OR_PATH)
        model = CLIPSegForImageSegmentation.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)

        image = prepare_img()
        texts = ["a cat", "a remote", "a blanket"]
        inputs = processor(text=texts, images=[image] * len(texts), padding=True, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the predicted masks
        self.assertEqual(
            outputs.logits.shape,
            torch.Size((3, 352, 352)),
        )
        expected_masks_slice = torch.tensor(
            [[-7.4613, -7.4785, -7.3628], [-7.3268, -7.0899, -7.1333], [-6.9838, -6.7900, -6.8913]]
        ).to(torch_device)
        self.assertTrue(torch.allclose(outputs.logits[0, :3, :3], expected_masks_slice, atol=1e-3))

        # verify conditional and pooled output
        expected_conditional = torch.tensor([0.5601, -0.0314, 0.1980]).to(torch_device)
        expected_pooled_output = torch.tensor([0.5036, -0.2681, -0.2644]).to(torch_device)
        self.assertTrue(torch.allclose(outputs.conditional_embeddings[0, :3], expected_conditional, atol=1e-3))
        self.assertTrue(torch.allclose(outputs.pooled_output[0, :3], expected_pooled_output, atol=1e-3))

if __name__=='__main__':
    unittest.main(argv=['', 'CLIPSegModelIntegrationTest.test_inference_image_segmentation'])
