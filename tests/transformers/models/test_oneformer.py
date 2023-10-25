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
""" Testing suite for the PyTorch Oneformer model. """

import unittest

from transformers.testing_utils import require_torch, require_vision, torch_device
from transformers.utils import is_torch_available, is_vision_available, cached_property

if is_torch_available():
    import torch
    from transformers import OneFormerModel

if is_vision_available():
    from PIL import Image
    from transformers import OneFormerProcessor

MODEL_NAME_OR_PATH = "shi-labs/oneformer_ade20k_swin_tiny"

def prepare_img():
    image = Image.open("000000039769.png")
    return image

TOLERANCE = 1e-4

@require_vision
class OneFormerModelIntegrationTest(unittest.TestCase):
    @cached_property
    def model_checkpoints(self):
        return MODEL_NAME_OR_PATH

    @cached_property
    def default_processor(self):
        return OneFormerProcessor.from_pretrained(self.model_checkpoints) if is_vision_available() else None

    def test_inference_no_head(self):
        model = OneFormerModel.from_pretrained(self.model_checkpoints).to(torch_device)
        processor = self.default_processor
        image = prepare_img()
        inputs = processor(image, ["semantic"], return_tensors="pt").to(torch_device)
        inputs_shape = inputs["pixel_values"].shape

        self.assertEqual(inputs_shape, (1, 3, 512, 682))

        task_inputs_shape = inputs["task_inputs"].shape

        self.assertEqual(task_inputs_shape, (1, 77))

        with torch.no_grad():
            outputs = model(**inputs)

        expected_slice_hidden_state = torch.tensor(
            [[0.2723, 0.8280, 0.6026], [1.2699, 1.1257, 1.1444], [1.1344, 0.6153, 0.4177]]
        ).to(torch_device)
        self.assertTrue(
            torch.allclose(
                outputs.encoder_hidden_states[-1][0, 0, :3, :3], expected_slice_hidden_state, atol=TOLERANCE
            )
        )

        expected_slice_hidden_state = torch.tensor(
            [[1.0581, 1.2276, 1.2003], [1.1903, 1.2925, 1.2862], [1.158, 1.2559, 1.3216]]
        ).to(torch_device)
        self.assertTrue(
            torch.allclose(
                outputs.pixel_decoder_hidden_states[0][0, 0, :3, :3], expected_slice_hidden_state, atol=TOLERANCE
            )
        )

        expected_slice_hidden_state = torch.tensor(
            [[3.0668, -1.1833, -5.1103], [3.344, -3.362, -5.1101], [2.6017, -4.3613, -4.1444]]
        ).to(torch_device)
        self.assertTrue(
            torch.allclose(
                outputs.transformer_decoder_class_predictions[0, :3, :3], expected_slice_hidden_state, atol=TOLERANCE
            )
        )

if __name__=='__main__':
    unittest.main(argv=['', 'OneFormerModelIntegrationTest.test_inference_no_head'])
