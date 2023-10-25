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
""" Testing suite for the PyTorch Dit model. """

import unittest

from transformers import is_torch_available, is_vision_available
from transformers.testing_utils import require_torch, require_vision, torch_device

if is_torch_available():
    import torch
    from transformers import AutoModelForImageClassification

if is_vision_available():
    from transformers import AutoImageProcessor

MODEL_NAME_OR_PATH = "microsoft/dit-base-finetuned-rvlcdip"

DATASET_PATH = "nielsr/rvlcdip-demo"

@require_torch
@require_vision
class DiTIntegrationTest(unittest.TestCase):
    def test_for_image_classification(self):
        image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME_OR_PATH)
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME_OR_PATH)
        model.to(torch_device)

        from datasets import load_dataset

        dataset = load_dataset(DATASET_PATH)

        image = dataset["train"][0]["image"].convert("RGB")

        inputs = image_processor(image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        expected_shape = torch.Size((1, 16))
        self.assertEqual(logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [-0.4158, -0.4092, -0.4347],
            device=torch_device,
            dtype=torch.float,
        )
        self.assertTrue(torch.allclose(logits[0, :3], expected_slice, atol=1e-4))

if __name__=='__main__':
    unittest.main(argv=['', 'DiTIntegrationTest.test_for_image_classification'])
