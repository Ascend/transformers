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
""" Testing suite for the PyTorch Table Transformer model. """

import unittest

from huggingface_hub import hf_hub_download
from transformers.utils import is_torch_available, is_vision_available
from transformers.testing_utils import torch_device, require_vision, require_torch, require_timm

if is_torch_available():
    import torch
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection

if is_vision_available():
    from PIL import Image

MODEL_NAME_OR_PATH = "microsoft/table-transformer-detection"

@require_timm
@require_vision
class TableTransformerModelIntegrationTests(unittest.TestCase):
    def test_table_detection(self):
        image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME_OR_PATH)
        model = TableTransformerForObjectDetection.from_pretrained(MODEL_NAME_OR_PATH)
        model.to(torch_device)

        file_path = hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename="example_pdf.png")
        image = Image.open(file_path).convert("RGB")
        inputs = image_processor(image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        expected_shape = (1, 15, 3)
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_logits = torch.tensor(
            [[-6.7329, -16.9590, 6.7447], [-8.0038, -22.3071, 6.9288], [-7.2445, -20.9855, 7.3465]],
            device=torch_device,
        )
        self.assertTrue(torch.allclose(outputs.logits[0, :3, :3], expected_logits, atol=1e-4))

        expected_boxes = torch.tensor(
            [[0.4868, 0.1764, 0.6729], [0.6674, 0.4621, 0.3864], [0.4720, 0.1757, 0.6362]], device=torch_device
        )
        self.assertTrue(torch.allclose(outputs.pred_boxes[0, :3, :3], expected_boxes, atol=1e-3))

if __name__=='__main__':
    unittest.main(argv=['', 'TableTransformerModelIntegrationTests.test_table_detection'])
