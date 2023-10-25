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
""" Testing suite for the PyTorch BLIP-2 model. """

import unittest
import requests

from transformers.utils import is_torch_available, is_vision_available
from transformers.testing_utils import torch_device, require_torch, require_vision

if is_torch_available():
    import torch
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

if is_vision_available():
    from PIL import Image

MODEL_NAME_OR_PATH = "Salesforce/blip2-flan-t5-xl"

def prepare_img():
    url = "https://huggingface.co/hf-internal-testing/blip-test-image/resolve/main/demo.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    return image

@require_vision
@require_torch
class Blip2ModelIntegrationTest(unittest.TestCase):
    def test_inference_t5(self):
        processor = Blip2Processor.from_pretrained(MODEL_NAME_OR_PATH)
        model = Blip2ForConditionalGeneration.from_pretrained(
            MODEL_NAME_OR_PATH, torch_dtype=torch.float16
        ).to(torch_device)

        image = prepare_img()
        inputs = processor(images=image, return_tensors="pt").to(torch_device, dtype=torch.float16)

        predictions = model.generate(**inputs)
        generated_text = processor.batch_decode(predictions, skip_special_tokens=True)[0].strip()

        self.assertEqual(predictions[0].tolist(), [0, 2335, 1556, 28, 1782, 30, 8, 2608, 1])
        self.assertEqual("woman playing with dog on the beach", generated_text)

        prompt = "Question: which city is this? Answer:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(torch_device, dtype=torch.float16)

        predictions = model.generate(**inputs)
        generated_text = processor.batch_decode(predictions, skip_special_tokens=True)[0].strip()

        self.assertEqual(
            predictions[0].tolist(),
            [0, 3, 7, 152, 67, 839, 1],
        )
        self.assertEqual(generated_text, "san diego")

if __name__=='__main__':
    unittest.main(argv=['', 'Blip2ModelIntegrationTest.test_inference_t5'])
