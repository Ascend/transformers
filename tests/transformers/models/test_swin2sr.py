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
""" Testing suite for the PyTorch Swin2Sr model. """

import unittest

from transformers.utils import is_torch_available, is_vision_available
from transformers.testing_utils import torch_device, require_vision, require_torch

if is_torch_available():
    import torch
    from transformers import Swin2SRImageProcessor, Swin2SRForImageSuperResolution

if is_vision_available():
    from PIL import Image

MODEL_NAME_OR_PATH = "caidas/swin2SR-classical-sr-x2-64"

def prepare_img():
    image = Image.open("000000039769.png")
    return image

@require_vision
@require_torch
class Swin2SRModelIntegrationTest(unittest.TestCase):
    def test_inference_image_super_resolution_head(self):
        processor = Swin2SRImageProcessor()
        model = Swin2SRForImageSuperResolution.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)

        image = prepare_img()
        inputs = processor(images=image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        expected_shape = torch.Size([1, 3, 976, 1296])
        self.assertEqual(outputs.reconstruction.shape, expected_shape)
        expected_slice = torch.tensor(
            [[0.5458, 0.5546, 0.5638], [0.5526, 0.5565, 0.5651], [0.5396, 0.5426, 0.5621]]
        ).to(torch_device)
        self.assertTrue(torch.allclose(outputs.reconstruction[0, 0, :3, :3], expected_slice, atol=1e-4))

if __name__=='__main__':
    unittest.main(argv=['', 'Swin2SRModelIntegrationTest.test_inference_image_super_resolution_head'])
