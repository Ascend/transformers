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
""" Testing suite for the PyTorch Blip model. """

import unittest

from transformers.utils import is_torch_available, is_vision_available
from transformers.testing_utils import require_torch, torch_device

if is_torch_available():
    import torch
    from transformers import (
        BlipForConditionalGeneration,
    )
    from transformers.models.blip.modeling_blip import BLIP_PRETRAINED_MODEL_ARCHIVE_LIST

if is_vision_available():
    from PIL import Image
    from transformers import BlipProcessor

MODEL_NAME_OR_PATH = "Salesforce/blip-image-captioning-base"

def prepare_img():
    url = "demo.jpg"
    im = Image.open(url)
    return im

@require_torch
class BlipModelIntegrationTest(unittest.TestCase):
    def test_inference_image_captioning(self):
        model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)
        processor = BlipProcessor.from_pretrained(MODEL_NAME_OR_PATH)
        image = prepare_img()

        inputs = processor(images=image, return_tensors="pt").to(torch_device)

        predictions = model.generate(**inputs)

        self.assertEqual(predictions[0].tolist(), [30522, 1037, 2450, 3564, 2006, 1996, 3509, 2007, 2014, 3899, 102])

        context = ["a picture of"]
        inputs = processor(images=image, text=context, return_tensors="pt").to(torch_device)

        predictions = model.generate(**inputs)

        self.assertEqual(
            predictions[0].tolist(),
            [30522, 1037, 3861, 1997, 1037, 2450, 1998, 2014, 3899, 2006, 1996, 3509, 102],
        )

if __name__=='__main__':
    unittest.main(argv=['', 'BlipModelIntegrationTest.test_inference_image_captioning'])
