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
""" Testing suite for the PyTorch Sew_d model. """

import unittest

from transformers.utils import is_torch_available
from transformers.testing_utils import torch_device, require_soundfile, require_torch

if is_torch_available():
    import torch
    from transformers import SEWDModel, Wav2Vec2FeatureExtractor

MODEL_NAME_OR_PATH = "asapp/sew-d-tiny-100k"
DATASET_NAME = "hf-internal-testing/librispeech_asr_dummy"

@require_torch
@require_soundfile
class SEWDModelIntegrationTest(unittest.TestCase):
    def _load_datasamples(self, num_samples):
        from datasets import load_dataset

        ds = load_dataset(DATASET_NAME, "clean", split="validation")

        speech_samples = ds.sort("id").filter(
            lambda x: x["id"] in [f"1272-141231-000{i}" for i in range(num_samples)]
        )[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    def test_inference_pretrained_batched(self):
        model = SEWDModel.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)
        processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME_OR_PATH)

        input_speech = self._load_datasamples(2)

        inputs = processor(input_speech, return_tensors="pt", padding=True)

        input_values = inputs.input_values.to(torch_device)

        with torch.no_grad():
            outputs = model(input_values).last_hidden_state

        expected_outputs_first = torch.tensor(
            [
                [
                    [-0.1619, 0.6995, 0.4062, -0.1014],
                    [-0.1364, 0.5960, 0.0952, -0.0873],
                    [-0.1572, 0.5718, 0.4228, -0.0864],
                    [-0.1325, 0.6823, 0.1387, -0.0871],
                ],
                [
                    [-0.1296, 0.4008, 0.4952, -0.1450],
                    [-0.1152, 0.3693, 0.3037, -0.1290],
                    [-0.1194, 0.6074, 0.3531, -0.1466],
                    [-0.1113, 0.3135, 0.2224, -0.1338],
                ],
            ],
            device=torch_device,
        )
        expected_outputs_last = torch.tensor(
            [
                [
                    [-0.1577, 0.5108, 0.8553, 0.2550],
                    [-0.1530, 0.3580, 0.6143, 0.2672],
                    [-0.1535, 0.4954, 0.8503, 0.1387],
                    [-0.1572, 0.3363, 0.6217, 0.1490],
                ],
                [
                    [-0.1338, 0.5459, 0.9607, -0.1133],
                    [-0.1502, 0.3738, 0.7313, -0.0986],
                    [-0.0953, 0.4708, 1.0821, -0.0944],
                    [-0.1474, 0.3598, 0.7248, -0.0748],
                ],
            ],
            device=torch_device,
        )
        expected_output_sum = 54201.0469

        self.assertTrue(torch.allclose(outputs[:, :4, :4], expected_outputs_first, atol=1e-3))
        self.assertTrue(torch.allclose(outputs[:, -4:, -4:], expected_outputs_last, atol=1e-3))
        self.assertTrue(abs(outputs.sum() - expected_output_sum) < 1)


if __name__=='__main__':
    unittest.main(argv=['', 'SEWDModelIntegrationTest.test_inference_pretrained_batched'])
