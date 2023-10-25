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
""" Testing suite for the PyTorch Tapas model. """

import unittest
import pandas as pd

from transformers.utils import is_torch_available, cached_property
from transformers.testing_utils import torch_device, require_torch

if is_torch_available():
    import torch
    from transformers import TapasTokenizer, TapasModel

MODEL_NAME_OR_PATH = "google/tapas-base-finetuned-wtq"

def prepare_tapas_single_inputs_for_inference():
    # Here we prepare a single table-question pair to test TAPAS inference on:
    data = {
        "Footballer": ["Lionel Messi", "Cristiano Ronaldo"],
        "Age": ["33", "35"],
    }
    queries = "Which footballer is 33 years old?"
    table = pd.DataFrame.from_dict(data)

    return table, queries

@require_torch
class TapasModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_tokenizer(self):
        return TapasTokenizer.from_pretrained(MODEL_NAME_OR_PATH)

    def test_inference_no_head(self):
        model = TapasModel.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)

        tokenizer = self.default_tokenizer
        table, queries = prepare_tapas_single_inputs_for_inference()
        inputs = tokenizer(table=table, queries=queries, return_tensors="pt")
        inputs = {k: v.to(torch_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        expected_slice = torch.tensor(
            [
                [
                    [-0.141581565, -0.599805772, 0.747186482],
                    [-0.143664181, -0.602008104, 0.749218345],
                    [-0.15169853, -0.603363097, 0.741370678],
                ]
            ],
            device=torch_device,
        )

        self.assertTrue(torch.allclose(outputs.last_hidden_state[:, :3, :3], expected_slice, atol=0.0005))

        expected_slice = torch.tensor([[0.987518311, -0.970520139, -0.994303405]], device=torch_device)

        self.assertTrue(torch.allclose(outputs.pooler_output[:, :3], expected_slice, atol=0.0005))

if __name__=='__main__':
    unittest.main(argv=['', 'TapasModelIntegrationTest.test_inference_no_head'])
