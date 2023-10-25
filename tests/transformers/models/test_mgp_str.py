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
""" Testing suite for the PyTorch MGP-STR model. """

import unittest

from transformers.testing_utils import require_torch, require_vision, torch_device
from transformers.utils import is_torch_available, is_vision_available

if is_torch_available():
    import torch
    from transformers import MgpstrForSceneTextRecognition

if is_vision_available():
    from PIL import Image
    from transformers import MgpstrProcessor

MODEL_NAME_OR_PATH = "alibaba-damo/mgp-str-base"

def prepare_img():
    image = Image.open("000000039769.png")
    return image

@require_vision
@require_torch
class MgpstrModelIntegrationTest(unittest.TestCase):

    def test_inference(self):
        model = MgpstrForSceneTextRecognition.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)
        processor = MgpstrProcessor.from_pretrained(MODEL_NAME_OR_PATH)

        image = prepare_img()
        inputs = processor(images=image, return_tensors="pt").pixel_values.to(torch_device)


        with torch.no_grad():
            outputs = model(inputs)


        self.assertEqual(outputs.logits[0].shape, torch.Size((1, 27, 38)))

        out_strs = processor.batch_decode(outputs.logits)
        expected_text = "ticket"

        self.assertEqual(out_strs["generated_text"][0], expected_text)

        expected_slice = torch.tensor(
            [[[-39.5397, -44.4024, -36.1844], [-61.4709, -63.8639, -58.3454], [-74.0225, -68.5494, -71.2164]]],
            device=torch_device,
        )

        self.assertTrue(torch.allclose(outputs.logits[0][:, 1:4, 1:4], expected_slice, atol=1e-4))

if __name__=='__main__':
    unittest.main(argv=['', 'MgpstrModelIntegrationTest.test_inference'])
