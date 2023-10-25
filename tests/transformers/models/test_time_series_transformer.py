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
""" Testing suite for the PyTorch Time_series_transformer model. """

import unittest

from huggingface_hub import hf_hub_download
from transformers.utils import is_torch_available, cached_property
from transformers.testing_utils import torch_device, require_torch

if is_torch_available():
    import torch
    from transformers import TimeSeriesTransformerModel

MODEL_NAME_OR_PATH = "huggingface/time-series-transformer-tourism-monthly"

TOLERANCE = 1e-4

def prepare_batch(filename="train-batch.pt"):
    file = hf_hub_download(repo_id="hf-internal-testing/tourism-monthly-batch", filename=filename, repo_type="dataset")
    batch = torch.load(file, map_location=torch_device)
    return batch

@require_torch
class TimeSeriesTransformerModelIntegrationTests(unittest.TestCase):
    def test_inference_no_head(self):
        model = TimeSeriesTransformerModel.from_pretrained(MODEL_NAME_OR_PATH).to(
            torch_device
        )
        batch = prepare_batch()

        with torch.no_grad():
            output = model(
                past_values=batch["past_values"],
                past_time_features=batch["past_time_features"],
                past_observed_mask=batch["past_observed_mask"],
                static_categorical_features=batch["static_categorical_features"],
                static_real_features=batch["static_real_features"],
                future_values=batch["future_values"],
                future_time_features=batch["future_time_features"],
            ).last_hidden_state

        expected_shape = torch.Size((64, model.config.context_length, model.config.d_model))
        self.assertEqual(output.shape, expected_shape)

        expected_slice = torch.tensor(
            [[0.8196, -1.5131, 1.4620], [1.1268, -1.3238, 1.5997], [1.5098, -1.0715, 1.7359]], device=torch_device
        )
        self.assertTrue(torch.allclose(output[0, :3, :3], expected_slice, atol=TOLERANCE))

if __name__=='__main__':
    unittest.main(argv=['', 'TimeSeriesTransformerModelIntegrationTests.test_inference_no_head'])
