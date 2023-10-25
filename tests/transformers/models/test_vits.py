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
""" Testing suite for the PyTorch VITS model. """

import unittest

from transformers.utils import is_torch_available
from transformers.testing_utils import torch_device, require_torch
from transformers.trainer_utils import set_seed

if is_torch_available():
    import torch
    from transformers import VitsModel, VitsTokenizer

MODEL_NAME_OR_PATH = "facebook/mms-tts-eng"

@require_torch
class VitsModelIntegrationTests(unittest.TestCase):
    def test_forward(self):
        model = VitsModel.from_pretrained(MODEL_NAME_OR_PATH)
        model.to(torch_device)

        tokenizer = VitsTokenizer.from_pretrained(MODEL_NAME_OR_PATH)

        set_seed(555)

        input_text = "Mister quilter is the apostle of the middle classes and we are glad to welcome his gospel!"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(torch_device)

        with torch.no_grad():
            outputs = model(input_ids)

        self.assertEqual(outputs.waveform.shape, (1, 87040))

        EXPECTED_LOGITS = torch.tensor(
            [
                -0.0042,  0.0176,  0.0354,  0.0504,  0.0621,  0.0777,  0.0980,  0.1224,
                 0.1475,  0.1679,  0.1817,  0.1832,  0.1713,  0.1542,  0.1384,  0.1256,
                 0.1147,  0.1066,  0.1026,  0.0958,  0.0823,  0.0610,  0.0340,  0.0022,
                -0.0337, -0.0677, -0.0969, -0.1178, -0.1311, -0.1363
            ]
        )
        # fmt: on
        self.assertTrue(torch.allclose(outputs.waveform[0, 10000:10030].cpu(), EXPECTED_LOGITS, atol=1e-4))

if __name__=='__main__':
    unittest.main(argv=['', 'VitsModelIntegrationTests.test_forward'])
