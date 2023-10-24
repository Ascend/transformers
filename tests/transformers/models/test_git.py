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
""" Testing suite for the PyTorch Git model. """

import unittest

from transformers.testing_utils import is_torch_available, is_vision_available
from transformers.testing_utils import torch_device, require_torch, require_vision

if is_torch_available():
    import torch

    from transformers import (
        GitForCausalLM, GitProcessor
    )

if is_vision_available():
    from PIL import Image

MODEL_NAME_OR_PATH = "microsoft/git-base"

def prepare_img():
    image = Image.open("000000039769.png")
    return image

@require_torch
@require_vision
class GitModelIntegrationTest(unittest.TestCase):
    def test_inference_image_captioning(self):
        processor = GitProcessor.from_pretrained(MODEL_NAME_OR_PATH)
        model = GitForCausalLM.from_pretrained(MODEL_NAME_OR_PATH)
        model.to(torch_device)

        image = prepare_img()
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(torch_device)

        outputs = model.generate(
            pixel_values=pixel_values, max_length=20, output_scores=True, return_dict_in_generate=True
        )
        generated_caption = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]

        expected_shape = torch.Size((1, 9))
        self.assertEqual(outputs.sequences.shape, expected_shape)
        self.assertEquals(generated_caption, "two cats laying on a pink blanket")
        self.assertTrue(outputs.scores[-1].shape, expected_shape)
        expected_slice = torch.tensor([[-0.8805, -0.8803, -0.8799]], device=torch_device)
        self.assertTrue(torch.allclose(outputs.scores[-1][0, :3], expected_slice, atol=1e-4))

if __name__=='__main__':
    unittest.main(argv=['', 'GitModelIntegrationTest.test_inference_image_captioning'])
