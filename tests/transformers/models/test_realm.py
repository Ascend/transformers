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
""" Testing suite for the PyTorch Realm model. """

import unittest

from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available

if is_torch_available():
    import torch
    from transformers import RealmScorer

MODEL_NAME_OR_PATH = "google/realm-cc-news-pretrained-scorer"

@require_torch
class RealmModelIntegrationTest(unittest.TestCase):
    def test_inference_scorer(self):
        num_candidates = 2

        model = RealmScorer.from_pretrained(MODEL_NAME_OR_PATH, num_candidates=num_candidates)

        input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
        candidate_input_ids = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
        output = model(input_ids, candidate_input_ids=candidate_input_ids)[0]

        expected_shape = torch.Size((1, 2))
        self.assertEqual(output.shape, expected_shape)

        expected_slice = torch.tensor([[0.7410, 0.7170]])
        self.assertTrue(torch.allclose(output, expected_slice, atol=1e-4))

if __name__=='__main__':
    unittest.main(argv=['', 'RealmModelIntegrationTest.test_inference_scorer'])
