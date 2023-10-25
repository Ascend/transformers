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
""" Testing suite for the PyTorch Vision_Encoder_Decoder model. """

import unittest

from datasets import load_dataset
from transformers.utils import is_torch_available, is_vision_available, cached_property
from transformers.testing_utils import torch_device, require_torch, require_vision

if is_torch_available():
    import torch
    from transformers import VisionEncoderDecoderModel

if is_vision_available():
    import PIL
    from PIL import Image
    from transformers import TrOCRProcessor

MODEL_NAME_OR_PATH = "microsoft/trocr-base-handwritten"

@require_vision
@require_torch
class TrOCRModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_processor(self):
        return TrOCRProcessor.from_pretrained(MODEL_NAME_OR_PATH) if is_vision_available() else None

    def test_inference_handwritten(self):
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)

        dataset = load_dataset("hf-internal-testing/fixtures_ocr", split="test")
        image = Image.open(dataset[0]["file"]).convert("RGB")

        processor = self.default_processor
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(torch_device)

        decoder_input_ids = torch.tensor([[model.config.decoder.decoder_start_token_id]]).to(torch_device)
        outputs = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)
        logits = outputs.logits

        expected_shape = torch.Size((1, 1, model.decoder.config.vocab_size))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [-1.4502, -4.6683, -0.5347, -2.9291, 9.1435, -3.0571, 8.9764, 1.7560, 8.7358, -1.5311]
        ).to(torch_device)

        self.assertTrue(torch.allclose(logits[0, 0, :10], expected_slice, atol=1e-4))

if __name__=='__main__':
    unittest.main(argv=['', 'TrOCRModelIntegrationTest.test_inference_handwritten'])
