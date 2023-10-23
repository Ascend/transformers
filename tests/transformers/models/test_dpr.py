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
""" Testing suite for the PyTorch Dpr model. """

import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device
from transformers.file_utils import cached_property

if is_torch_available():
    import torch
    from transformers import DPRQuestionEncoder

MODEL_NAME_OR_PATH = "facebook/dpr-question_encoder-single-nq-base"

@require_torch
class DPRModelIntegrationTest(unittest.TestCase):

    def test_inference_no_head(self):
        model = DPRQuestionEncoder.from_pretrained(MODEL_NAME_OR_PATH, return_dict=False)
        model.to(torch_device)

        input_ids = torch.tensor(
            [[101, 7592, 1010, 2003, 2026, 3899, 10140, 1029, 102]], dtype=torch.long, device=torch_device
        )  # [CLS] hello, is my dog cute? [SEP]
        output = model(input_ids)[0]  # embedding shape = (1, 768)
        # compare the actual values for a slice.
        expected_slice = torch.tensor(
            [
                [
                    0.03236253,
                    0.12753335,
                    0.16818509,
                    0.00279786,
                    0.3896933,
                    0.24264945,
                    0.2178971,
                    -0.02335227,
                    -0.08481959,
                    -0.14324117,
                ]
            ],
            dtype=torch.float,
            device=torch_device,
        )
        self.assertTrue(torch.allclose(output[:, :10], expected_slice, atol=1e-4))

if __name__=='__main__':
    unittest.main(argv=['', 'DPRModelIntegrationTest.test_inference_no_head'])
