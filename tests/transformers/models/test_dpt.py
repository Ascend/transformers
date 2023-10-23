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
""" Testing suite for the PyTorch Dpt model. """

import unittest

from transformers import is_torch_available, is_vision_available
from transformers.testing_utils import require_torch, require_vision, torch_device

if is_torch_available():
    import torch
    from transformers import DPTForSemanticSegmentation

if is_vision_available():
    from PIL import Image
    from transformers import DPTImageProcessor

MODEL_NAME_OR_PATH = "Intel/dpt-large-ade"

def prepare_img():
    image = Image.open("000000039769.png")
    return image

@require_torch
@require_vision
class DPTModelIntegrationTest(unittest.TestCase):
    def test_inference_semantic_segmentation(self):
        image_processor = DPTImageProcessor.from_pretrained(MODEL_NAME_OR_PATH)
        model = DPTForSemanticSegmentation.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)

        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 150, 480, 480))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [[4.0480, 4.2420, 4.4360], [4.3124, 4.5693, 4.8261], [4.5768, 4.8965, 5.2163]]
        ).to(torch_device)

        self.assertTrue(torch.allclose(outputs.logits[0, 0, :3, :3], expected_slice, atol=1e-4))

if __name__=='__main__':
    unittest.main(argv=['', 'DPTModelIntegrationTest.test_inference_semantic_segmentation'])
