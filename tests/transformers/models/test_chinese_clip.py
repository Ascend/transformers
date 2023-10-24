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
""" Testing suite for the PyTorch Chinese_clip model. """

import unittest
import requests

from transformers.testing_utils import is_torch_available, is_vision_available, torch_device

if is_torch_available():
    import torch

    from transformers import (
        ChineseCLIPModel, ChineseCLIPProcessor
    )

if is_vision_available():
    from PIL import Image

MODEL_NAME_OR_PATH = "OFA-Sys/chinese-clip-vit-base-patch16"

def prepare_img():
    url = "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im

class ChineseCLIPModelIntegrationTest(unittest.TestCase):
    def test_inference(self):
        model = ChineseCLIPModel.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)
        processor = ChineseCLIPProcessor.from_pretrained(MODEL_NAME_OR_PATH)

        image = prepare_img()
        inputs = processor(text=["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"], images=image, padding=True, return_tensors="pt").to(
            torch_device
        )

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        self.assertEqual(
            outputs.logits_per_image.shape,
            torch.Size((inputs.pixel_values.shape[0], inputs.input_ids.shape[0])),
        )
        self.assertEqual(
            outputs.logits_per_text.shape,
            torch.Size((inputs.input_ids.shape[0], inputs.pixel_values.shape[0])),
        )

        probs = outputs.logits_per_image.softmax(dim=1)
        expected_probs = torch.tensor([[1.2686e-03, 5.4499e-02, 6.7968e-04, 9.4355e-01]], device=torch_device)

        self.assertTrue(torch.allclose(probs, expected_probs, atol=5e-3))

if __name__=='__main__':
    unittest.main(argv=['', 'ChineseCLIPModelIntegrationTest.test_inference'])
