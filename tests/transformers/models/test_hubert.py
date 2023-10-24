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
""" Testing suite for the PyTorch Hubert  model. """

import unittest

from transformers.utils import is_torch_available
from transformers.testing_utils import torch_device, require_torch

if is_torch_available():
    import torch
    from transformers import HubertForCTC, Wav2Vec2Processor

MODEL_NAME_OR_PATH = "facebook/hubert-large-ls960-ft"
DATASET_NAME = "hf-internal-testing/librispeech_asr_dummy"

@require_torch
class HubertModelIntegrationTest(unittest.TestCase):
    def _load_datasamples(self, num_samples):
        from datasets import load_dataset

        ds = load_dataset(DATASET_NAME, "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id").filter(
            lambda x: x["id"] in [f"1272-141231-000{i}" for i in range(num_samples)]
        )[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    def test_inference_ctc_batched(self):
        model = HubertForCTC.from_pretrained(MODEL_NAME_OR_PATH, torch_dtype=torch.float16).to(
            torch_device
        )
        processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME_OR_PATH, do_lower_case=True)

        input_speech = self._load_datasamples(2)

        inputs = processor(input_speech, return_tensors="pt", padding=True)

        input_values = inputs.input_values.half().to(torch_device)
        attention_mask = inputs.attention_mask.to(torch_device)

        with torch.no_grad():
            logits = model(input_values, attention_mask=attention_mask).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_trans = processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = [
            "a man said to the universe sir i exist",
            "sweat covered brion's body trickling into the tight loin cloth that was the only garment he wore",
        ]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)

if __name__=='__main__':
    unittest.main(argv=['', 'HubertModelIntegrationTest.test_inference_ctc_batched'])
