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
""" Testing suite for the PyTorch XCLIP model. """

import unittest

import numpy as np
from huggingface_hub import hf_hub_download

from transformers.utils import is_torch_available, is_vision_available
from transformers.testing_utils import torch_device, require_torch

if is_torch_available():
    import torch
    from transformers import XCLIPModel, XCLIPProcessor

MODEL_NAME_OR_PATH = "microsoft/xclip-base-patch32"

def prepare_video():
    file = hf_hub_download(
        repo_id="hf-internal-testing/spaghetti-video", filename="eating_spaghetti_8_frames.npy", repo_type="dataset"
    )
    video = np.load(file)
    return list(video)

@require_torch
class XCLIPModelIntegrationTest(unittest.TestCase):
    def test_inference(self):
        model = XCLIPModel.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)
        processor = XCLIPProcessor.from_pretrained(MODEL_NAME_OR_PATH)

        video = prepare_video()
        inputs = processor(
            text=["playing sports", "eating spaghetti", "go shopping"], videos=video, return_tensors="pt", padding=True
        ).to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        self.assertEqual(
            outputs.logits_per_video.shape,
            torch.Size((inputs.pixel_values.shape[0], inputs.input_ids.shape[0])),
        )
        self.assertEqual(
            outputs.logits_per_text.shape,
            torch.Size((inputs.input_ids.shape[0], inputs.pixel_values.shape[0])),
        )

        expected_logits = torch.tensor([[14.0181, 20.2771, 14.4776]], device=torch_device)

        self.assertTrue(torch.allclose(outputs.logits_per_video, expected_logits, atol=1e-3))

if __name__=='__main__':
    unittest.main(argv=['', 'XCLIPModelIntegrationTest.test_inference'])
