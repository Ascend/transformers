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
""" Testing suite for the PyTorch Lxmert model. """

import unittest
import numpy as np

from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available

if is_torch_available():
    import torch
    from transformers import LxmertModel

MODEL_NAME_OR_PATH = "unc-nlp/lxmert-base-uncased"

@require_torch
class LxmertModelIntegrationTest(unittest.TestCase):

    def test_inference_no_head_absolute_embedding(self):
        model = LxmertModel.from_pretrained(MODEL_NAME_OR_PATH)
        input_ids = torch.tensor([[101, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 102]])
        num_visual_features = 10
        _, visual_feats = np.random.seed(0), np.random.rand(1, num_visual_features, model.config.visual_feat_dim)
        _, visual_pos = np.random.seed(0), np.random.rand(1, num_visual_features, 4)
        visual_feats = torch.as_tensor(visual_feats, dtype=torch.float32)
        visual_pos = torch.as_tensor(visual_pos, dtype=torch.float32)
        output = model(input_ids, visual_feats=visual_feats, visual_pos=visual_pos)[0]
        expected_shape = torch.Size([1, 11, 768])
        self.assertEqual(expected_shape, output.shape)
        expected_slice = torch.tensor(
            [[[0.2417, -0.9807, 0.1480], [1.2541, -0.8320, 0.5112], [1.4070, -1.1052, 0.6990]]]
        )

        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=1e-4))

if __name__=='__main__':
    unittest.main(argv=['', 'LxmertModelIntegrationTest.test_inference_no_head_absolute_embedding'])
