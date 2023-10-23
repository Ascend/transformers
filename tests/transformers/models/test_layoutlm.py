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
""" Testing suite for the PyTorch Layoutlm model. """

import unittest

from transformers.testing_utils import require_torch, torch_device
from transformers.utils import is_torch_available
from transformers import LayoutLMForTokenClassification

if is_torch_available():
    import torch

MODEL_NAME_OR_PATH = "microsoft/layoutlm-base-uncased"

def prepare_layoutlm_batch_inputs():

    input_ids = torch.tensor([[101,1019,1014,1016,1037,12849,4747,1004,14246,2278,5439,4524,5002,2930,2193,2930,4341,3208,1005,1055,2171,2848,11300,3531,102],[101,4070,4034,7020,1024,3058,1015,1013,2861,1013,6070,19274,2772,6205,27814,16147,16147,4343,2047,10283,10969,14389,1012,2338,102]],device=torch_device)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],],device=torch_device)
    bbox = torch.tensor([[[0,0,0,0],[423,237,440,251],[427,272,441,287],[419,115,437,129],[961,885,992,912],[256,38,330,58],[256,38,330,58],[336,42,353,57],[360,39,401,56],[360,39,401,56],[411,39,471,59],[479,41,528,59],[533,39,630,60],[67,113,134,131],[141,115,209,132],[68,149,133,166],[141,149,187,164],[195,148,287,165],[195,148,287,165],[195,148,287,165],[295,148,349,165],[441,149,492,166],[497,149,546,164],[64,201,125,218],[1000,1000,1000,1000]],[[0,0,0,0],[662,150,754,166],[665,199,742,211],[519,213,554,228],[519,213,554,228],[134,433,187,454],[130,467,204,480],[130,467,204,480],[130,467,204,480],[130,467,204,480],[130,467,204,480],[314,469,376,482],[504,684,582,706],[941,825,973,900],[941,825,973,900],[941,825,973,900],[941,825,973,900],[610,749,652,765],[130,659,168,672],[176,657,237,672],[238,657,312,672],[443,653,628,672],[443,653,628,672],[716,301,825,317],[1000,1000,1000,1000]]],device=torch_device)
    token_type_ids = torch.tensor([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],device=torch_device)
    labels = torch.tensor([[-100,10,10,10,9,1,-100,7,7,-100,7,7,4,2,5,2,8,8,-100,-100,5,0,3,2,-100],[-100,12,12,12,-100,12,10,-100,-100,-100,-100,10,12,9,-100,-100,-100,10,10,10,9,12,-100,10,-100]],device=torch_device)


    return input_ids, attention_mask, bbox, token_type_ids, labels

@require_torch
class LayoutLMModelIntegrationTest(unittest.TestCase):

    def test_forward_pass_token_classification(self):

        model = LayoutLMForTokenClassification.from_pretrained(MODEL_NAME_OR_PATH, num_labels=13).to(
            torch_device
        )

        input_ids, attention_mask, bbox, token_type_ids, labels = prepare_layoutlm_batch_inputs()

        outputs = model(
            input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels
        )

        logits = outputs.logits
        expected_shape = torch.Size((2, 25, 13))
        self.assertEqual(logits.shape, expected_shape)

if __name__=='__main__':
    unittest.main(argv=['', 'LayoutLMModelIntegrationTest.test_forward_pass_token_classification'])
