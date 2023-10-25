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
""" Testing suite for the PyTorch UperNet model. """

import unittest

from huggingface_hub import hf_hub_download
from transformers.utils import is_torch_available, is_vision_available
from transformers.testing_utils import torch_device, require_torch, require_vision

if is_torch_available():
    import torch
    from transformers import UperNetForSemanticSegmentation

if is_vision_available():
    from PIL import Image
    from transformers import AutoImageProcessor

MODEL_NAME_OR_PATH = "openmmlab/upernet-swin-tiny"

def prepare_img():
    filepath = hf_hub_download(
        repo_id="hf-internal-testing/fixtures_ade20k", repo_type="dataset", filename="ADE_val_00000001.jpg"
    )
    image = Image.open(filepath).convert("RGB")
    return image

@require_torch
@require_vision
class UperNetModelIntegrationTest(unittest.TestCase):
    def test_inference_swin_backbone(self):
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME_OR_PATH)
        model = UperNetForSemanticSegmentation.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)

        image = prepare_img()
        inputs = processor(images=image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        expected_shape = torch.Size((1, model.config.num_labels, 512, 512))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [[-7.5958, -7.5958, -7.4302], [-7.5958, -7.5958, -7.4302], [-7.4797, -7.4797, -7.3068]]
        ).to(torch_device)
        self.assertTrue(torch.allclose(outputs.logits[0, 0, :3, :3], expected_slice, atol=1e-4))

if __name__=='__main__':
    unittest.main(argv=['', 'UperNetModelIntegrationTest.test_inference_swin_backbone'])
