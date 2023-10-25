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
""" Testing suite for the PyTorch BridgeTower model. """

import unittest

from transformers import is_torch_available, is_vision_available
from transformers.testing_utils import torch_device, require_torch, require_vision
from transformers.utils import cached_property

if is_torch_available():
    import torch

    from transformers import (
        BridgeTowerForImageAndTextRetrieval, BridgeTowerProcessor
    )

if is_vision_available():
    from PIL import Image

MODEL_NAME_OR_PATH = "BridgeTower/bridgetower-base-itm-mlm"

def prepare_img():
    image = Image.open("000000039769.png")
    return image

@require_torch
@require_vision
class BridgeTowerModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_processor(self):
        return (
            BridgeTowerProcessor.from_pretrained(MODEL_NAME_OR_PATH)
            if is_vision_available()
            else None
        )
    def test_image_and_text_retrieval(self):
        model = BridgeTowerForImageAndTextRetrieval.from_pretrained(MODEL_NAME_OR_PATH).to(
            torch_device
        )
        model.eval()
        processor = self.default_processor
        image = prepare_img()
        text = "a bunch of cats laying on a tower."
        inputs = processor(image, text, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        expected_shape = torch.Size([1, 2])
        self.assertEqual(outputs.logits.shape, expected_shape)
        self.assertTrue(outputs.logits[0, 1].item() > outputs.logits[0, 0].item())

        inputs["labels"] = torch.ones(1, dtype=torch.long, device=torch_device)
        inputs = inputs.to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs)
        self.assertAlmostEqual(outputs.loss.item(), 0.5108, places=4)

if __name__=='__main__':
    unittest.main(argv=['', 'BridgeTowerModelIntegrationTest.test_image_and_text_retrieval'])
