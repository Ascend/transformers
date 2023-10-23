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
""" Testing suite for the PyTorch Hertbert model. """

import unittest

from transformers.testing_utils import require_tokenizers
from transformers.utils import is_torch_available

if is_torch_available():
    import torch
    from transformers import HerbertTokenizer, HerbertTokenizerFast

MODEL_NAME_OR_PATH = "allegro/herbert-base-cased"

@require_tokenizers
class HerbertTokenizationTest(unittest.TestCase):
    tokenizer_class = HerbertTokenizer
    rust_tokenizer_class = HerbertTokenizerFast
    test_rust_tokenizer = True

    def test_sequence_builders(self):
        tokenizer = self.tokenizer_class.from_pretrained(MODEL_NAME_OR_PATH)

        text = tokenizer.encode("konstruowanie sekwencji", add_special_tokens=False)
        text_2 = tokenizer.encode("konstruowanie wielu sekwencji", add_special_tokens=False)

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [0] + text + [2]
        assert encoded_pair == [0] + text + [2] + text_2 + [2]

if __name__=='__main__':
    unittest.main(argv=['', 'HerbertTokenizationTest.test_sequence_builders'])
