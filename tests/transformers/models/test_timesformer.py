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
""" Testing suite for the PyTorch Timesformer model. """

import unittest
import numpy as np

from huggingface_hub import hf_hub_download
from transformers.utils import is_torch_available, is_vision_available, cached_property
from transformers.testing_utils import torch_device, require_torch, require_vision

if is_torch_available():
    import torch
    from transformers import TimesformerForVideoClassification, VideoMAEImageProcessor

MODEL_NAME_OR_PATH = "facebook/timesformer-base-finetuned-k400"

def prepare_video():
    file = hf_hub_download(
        repo_id="hf-internal-testing/spaghetti-video", filename="eating_spaghetti.npy", repo_type="dataset"
    )
    video = np.load(file)
    return list(video)

@require_torch
@require_vision
class TimesformerModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):

        return (
            VideoMAEImageProcessor(image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
            if is_vision_available()
            else None
        )

    def test_inference_for_video_classification(self):
        model = TimesformerForVideoClassification.from_pretrained(MODEL_NAME_OR_PATH).to(
            torch_device
        )

        image_processor = self.default_image_processor
        video = prepare_video()
        inputs = image_processor(video[:8], return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        expected_shape = torch.Size((1, 400))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor([-0.3016, -0.7713, -0.4205]).to(torch_device)

        self.assertTrue(torch.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4))

if __name__=='__main__':
    unittest.main(argv=['', 'TimesformerModelIntegrationTest.test_inference_for_video_classification'])
