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
""" Testing suite for the PyTorch Detr model. """

import unittest

from transformers.utils import is_torch_available, is_vision_available
from transformers.testing_utils import torch_device, require_torch, require_vision
from transformers.file_utils import cached_property

if is_torch_available():
    import torch
    from transformers import DetrModel

if is_vision_available():
    from PIL import Image
    from transformers import DetrImageProcessor

MODEL_NAME_OR_PATH = "facebook/detr-resnet-50"

def prepare_img():
    image = Image.open("000000039769.png")
    return image

@require_vision
@require_torch
class DetrModelIntegrationTests(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return (
            DetrImageProcessor.from_pretrained(MODEL_NAME_OR_PATH, revision="no_timm")
            if is_vision_available()
            else None
        )

    def test_inference_no_head(self):
        model = DetrModel.from_pretrained(MODEL_NAME_OR_PATH, revision="no_timm").to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        encoding = image_processor(images=image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**encoding)

        expected_shape = torch.Size((1, 100, 256))
        assert outputs.last_hidden_state.shape == expected_shape
        expected_slice = torch.tensor(
            [[0.0616, -0.5146, -0.4032], [-0.7629, -0.4934, -1.7153], [-0.4768, -0.6403, -0.7826]]
        ).to(torch_device)
        self.assertTrue(torch.allclose(outputs.last_hidden_state[0, :3, :3], expected_slice, atol=1e-4))

if __name__=='__main__':
    unittest.main(argv=['', 'DetrModelIntegrationTests.test_inference_no_head'])
