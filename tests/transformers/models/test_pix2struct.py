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
""" Testing suite for the PyTorch Pix2Struct model. """

import unittest

from transformers.testing_utils import require_torch, require_vision, torch_device
from transformers.utils import is_torch_available, is_vision_available

if is_torch_available():
    import torch
    from transformers import Pix2StructForConditionalGeneration

if is_vision_available():
    from PIL import Image
    from transformers import Pix2StructProcessor

MODEL_NAME_OR_PATH = "google/pix2struct-textcaps-base"

def prepare_img():
    im = Image.open("australia.jpg")
    return im

@require_vision
@require_torch
class Pix2StructIntegrationTest(unittest.TestCase):
    def test_batched_inference_image_captioning(self):
        model = Pix2StructForConditionalGeneration.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)
        processor = Pix2StructProcessor.from_pretrained(MODEL_NAME_OR_PATH)
        image_1 = prepare_img()

        image_2 = Image.open("temple-bar-dublin-world-famous-irish-pub.jpg")

        inputs = processor(images=[image_1, image_2], return_tensors="pt").to(torch_device)

        predictions = model.generate(**inputs)

        self.assertEqual(
            processor.decode(predictions[0], skip_special_tokens=True), "A stop sign is on a street corner."
        )

        self.assertEqual(
            processor.decode(predictions[1], skip_special_tokens=True),
            "A row of books including The Temple Bar and Guiness.",
        )

if __name__=='__main__':
    unittest.main(argv=['', 'Pix2StructIntegrationTest.test_batched_inference_image_captioning'])
