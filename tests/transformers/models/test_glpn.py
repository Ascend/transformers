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
""" Testing suite for the PyTorch Glpn model. """

import unittest

from transformers.testing_utils import require_torch, require_vision, torch_device
from transformers.utils import is_torch_available, is_vision_available

if is_torch_available():
    import torch
    from transformers import GLPNForDepthEstimation

if is_vision_available():
    from PIL import Image
    from transformers import GLPNImageProcessor

MODEL_NAME_OR_PATH = "vinvino02/glpn-kitti"

def prepare_img():
    image = Image.open("000000039769.png")
    return image

@require_torch
@require_vision
class GLPNModelIntegrationTest(unittest.TestCase):

    def test_inference_depth_estimation(self):
        image_processor = GLPNImageProcessor.from_pretrained(MODEL_NAME_OR_PATH)
        model = GLPNForDepthEstimation.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)

        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        expected_shape = torch.Size([1, 480, 640])
        self.assertEqual(outputs.predicted_depth.shape, expected_shape)

        expected_slice = torch.tensor(
            [[3.4291, 2.7865, 2.5151], [3.2841, 2.7021, 2.3502], [3.1147, 2.4625, 2.2481]]
        ).to(torch_device)

        self.assertTrue(torch.allclose(outputs.predicted_depth[0, :3, :3], expected_slice, atol=1e-4))

if __name__=='__main__':
    unittest.main(argv=['', 'GLPNModelIntegrationTest.test_inference_depth_estimation'])
