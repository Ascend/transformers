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
""" Testing suite for the PyTorch Sew model. """

import unittest

from transformers.utils import is_torch_available
from transformers.testing_utils import torch_device, require_soundfile, require_torch

if is_torch_available():
    import torch
    from transformers import Wav2Vec2FeatureExtractor, SEWModel

MODEL_NAME_OR_PATH = "asapp/sew-tiny-100k"
DATASET_NAME = "hf-internal-testing/librispeech_asr_dummy"

@require_torch
@require_soundfile
class SEWModelIntegrationTest(unittest.TestCase):
    def _load_datasamples(self, num_samples):
        from datasets import load_dataset

        ds = load_dataset(DATASET_NAME, "clean", split="validation")

        speech_samples = ds.sort("id").filter(
            lambda x: x["id"] in [f"1272-141231-000{i}" for i in range(num_samples)]
        )[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    def test_inference_pretrained_batched(self):
        model = SEWModel.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)
        processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME_OR_PATH)

        input_speech = self._load_datasamples(2)

        inputs = processor(input_speech, return_tensors="pt", padding=True)

        input_values = inputs.input_values.to(torch_device)

        with torch.no_grad():
            outputs = model(input_values).last_hidden_state

        expected_outputs_first = torch.tensor(
            [
                [
                    [0.1509, 0.5372, 0.3061, -0.1694],
                    [-0.1700, 0.5764, 0.2753, -0.1299],
                    [0.1281, 0.7949, 0.2342, -0.1624],
                    [-0.1627, 0.6710, 0.2215, -0.1317],
                ],
                [
                    [0.0408, 1.4355, 0.8605, -0.0968],
                    [0.0393, 1.2368, 0.6826, 0.0364],
                    [-0.1269, 1.9215, 1.1677, -0.1297],
                    [-0.1654, 1.6524, 0.6877, -0.0196],
                ],
            ],
            device=torch_device,
        )
        expected_outputs_last = torch.tensor(
            [
                [
                    [1.3379, -0.1450, -0.1500, -0.0515],
                    [0.8364, -0.1680, -0.1248, -0.0689],
                    [1.2791, -0.1507, -0.1523, -0.0564],
                    [0.8208, -0.1690, -0.1199, -0.0751],
                ],
                [
                    [0.6959, -0.0861, -0.1235, -0.0861],
                    [0.4700, -0.1686, -0.1141, -0.1199],
                    [1.0776, -0.1137, -0.0124, -0.0472],
                    [0.5774, -0.1675, -0.0376, -0.0823],
                ],
            ],
            device=torch_device,
        )
        expected_output_sum = 62146.7422

        self.assertTrue(torch.allclose(outputs[:, :4, :4], expected_outputs_first, atol=5e-3))
        self.assertTrue(torch.allclose(outputs[:, -4:, -4:], expected_outputs_last, atol=5e-3))

if __name__=='__main__':
    unittest.main(argv=['', 'SEWModelIntegrationTest.test_inference_pretrained_batched'])
