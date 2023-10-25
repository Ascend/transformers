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
""" Testing suite for the PyTorch Markuplm model. """

import unittest

from transformers.testing_utils import require_torch, torch_device
from transformers.utils import is_torch_available, cached_property
from transformers import MarkupLMFeatureExtractor, MarkupLMProcessor, MarkupLMTokenizer

if is_torch_available():
    import torch
    from transformers import MarkupLMModel

MODEL_NAME_OR_PATH = "microsoft/markuplm-base"

def prepare_html_string():
    html_string = """
    <!DOCTYPE html>
    <html>
    <head>
    <title>Page Title</title>
    </head>
    <body>

    <h1>This is a Heading</h1>
    <p>This is a paragraph.</p>

    </body>
    </html>
    """

    return html_string

@require_torch
class MarkupLMModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_processor(self):
        feature_extractor = MarkupLMFeatureExtractor()
        tokenizer = MarkupLMTokenizer.from_pretrained(MODEL_NAME_OR_PATH)

        return MarkupLMProcessor(feature_extractor, tokenizer)


    def test_forward_pass_no_head(self):
        model = MarkupLMModel.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)

        processor = self.default_processor

        inputs = processor(prepare_html_string(), return_tensors="pt")
        inputs = inputs.to(torch_device)


        with torch.no_grad():
            outputs = model(**inputs)


        expected_shape = torch.Size([1, 14, 768])
        self.assertEqual(outputs.last_hidden_state.shape, expected_shape)

        expected_slice = torch.tensor(
            [[0.0675, -0.0052, 0.5001], [-0.2281, 0.0802, 0.2192], [-0.0583, -0.3311, 0.1185]]
        ).to(torch_device)

        self.assertTrue(torch.allclose(outputs.last_hidden_state[0, :3, :3], expected_slice, atol=1e-4))

if __name__=='__main__':
    unittest.main(argv=['', 'MarkupLMModelIntegrationTest.test_forward_pass_no_head'])
