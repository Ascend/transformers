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
""" Testing suite for the PyTorch Informer model. """

import unittest

from transformers.testing_utils import require_torch, torch_device
from transformers.utils import is_torch_available
from transformers import InformerModel

if is_torch_available():
    import torch

MODEL_NAME_OR_PATH = "huggingface/informer-tourism-monthly"

TOLERANCE = 1e-4

def prepare_batch(filename="train-batch.pt"):
    file = "hf-internal-testing/tourism-monthly-batch/train-batch.pt"
    batch = torch.load(file, map_location=torch_device)
    return batch

@require_torch
class InformerModelIntegrationTests(unittest.TestCase):

    def test_inference_no_head(self):
        model = InformerModel.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)
        batch = prepare_batch()

        torch.manual_seed(0)
        with torch.no_grad():
            output = model(
                past_values=batch["past_values"],
                past_time_features=batch["past_time_features"],
                past_observed_mask=batch["past_observed_mask"],
                static_categorical_features=batch["static_categorical_features"],
                future_values=batch["future_values"],
                future_time_features=batch["future_time_features"],
            ).last_hidden_state
        expected_shape = torch.Size((64, model.config.context_length, model.config.d_model))
        self.assertEqual(output.shape, expected_shape)

        expected_slice = torch.tensor(
            [[0.4699, 0.7295, 0.8967], [0.4858, 0.3810, 0.9641], [-0.0233, 0.3608, 1.0303]],
            device=torch_device,
        )
        self.assertTrue(torch.allclose(output[0, :3, :3], expected_slice, atol=TOLERANCE))

if __name__=='__main__':
    unittest.main(argv=['', 'InformerModelIntegrationTests.test_inference_no_head'])
