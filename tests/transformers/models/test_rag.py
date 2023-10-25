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
""" Testing suite for the PyTorch Rag model. """

import unittest
from datasets import load_dataset

from transformers.testing_utils import require_torch, torch_device
from transformers.utils import is_torch_available, cached_property
from transformers.models.dpr.tokenization_dpr import DPRQuestionEncoderTokenizer

if is_torch_available():
    import torch
    from transformers import BartTokenizer, RagRetriever, AutoConfig, RagConfig, RagSequenceForGeneration

MODEL_NAME_OR_PATH = ["facebook/dpr-question_encoder-single-nq-base", "facebook/bart-large-cnn"]

TOLERANCE = 1e-3

def _assert_tensors_equal(a, b, atol=1e-12, prefix=""):
    """If tensors not close, or a and b arent both tensors, raise a nice Assertion error."""
    if a is None and b is None:
        return True
    try:
        if torch.allclose(a, b, atol=atol):
            return True
        raise
    except Exception:
        msg = f"{a} != {b}"
        if prefix:
            msg = prefix + ": " + msg
        raise AssertionError(msg)

@require_torch
class RagModelIntegrationTests(unittest.TestCase):
    @cached_property
    def sequence_model(self):
        return (
            RagSequenceForGeneration.from_pretrained_question_encoder_generator(
                MODEL_NAME_OR_PATH[0], MODEL_NAME_OR_PATH[1]
            )
            .to(torch_device)
            .eval()
        )

    def get_rag_config(self):
        question_encoder_config = AutoConfig.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        generator_config = AutoConfig.from_pretrained("facebook/bart-large-cnn")
        return RagConfig.from_question_encoder_generator_configs(
            question_encoder_config,
            generator_config,
            bos_token_id=0,
            decoder_start_token_id=2,
            eos_token_id=2,
            is_encoder_decoder=True,
            pad_token_id=1,
            vocab_size=50264,
            title_sep=" / ",
            doc_sep=" // ",
            n_docs=5,
            max_combined_length=300,
            dataset="wiki_dpr",
            dataset_split="train",
            index_name="exact",
            index_path=None,
            use_dummy_dataset=True,
            retrieval_vector_size=768,
            retrieval_batch_size=8,
        )

    def test_rag_sequence_inference(self):
        rag_config = self.get_rag_config()
        rag_decoder_tokenizer = BartTokenizer.from_pretrained(MODEL_NAME_OR_PATH[1])
        rag_question_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            MODEL_NAME_OR_PATH[0]
        )
        rag_retriever = RagRetriever(
            rag_config,
            question_encoder_tokenizer=rag_question_encoder_tokenizer,
            generator_tokenizer=rag_decoder_tokenizer,
        )

        rag_sequence = self.sequence_model
        rag_sequence.set_retriever(rag_retriever)

        input_ids = rag_question_encoder_tokenizer(
            "who sings does he love me with reba", return_tensors="pt"
        ).input_ids
        decoder_input_ids = rag_decoder_tokenizer("Linda Davis", return_tensors="pt").input_ids

        input_ids = input_ids.to(torch_device)
        decoder_input_ids = decoder_input_ids.to(torch_device)

        with torch.no_grad():
            output = rag_sequence(
                input_ids,
                labels=decoder_input_ids,
            )

        expected_shape = torch.Size([5, 5, 50264])
        self.assertEqual(output.logits.shape, expected_shape)

        expected_doc_scores = torch.tensor([[75.0286, 74.4998, 74.0804, 74.0306, 73.9504]]).to(torch_device)
        _assert_tensors_equal(expected_doc_scores, output.doc_scores, atol=TOLERANCE)

        expected_loss = torch.tensor([36.7368]).to(torch_device)
        _assert_tensors_equal(expected_loss, output.loss, atol=TOLERANCE)

if __name__=='__main__':
    unittest.main(argv=['', 'RagModelIntegrationTests.test_rag_sequence_inference'])
