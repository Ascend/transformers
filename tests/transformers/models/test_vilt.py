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
""" Testing suite for the PyTorch Vilt model. """

import unittest

from transformers.utils import is_torch_available, is_vision_available, cached_property
from transformers.testing_utils import torch_device, require_torch, require_vision

if is_torch_available():
    import torch
    from transformers import ViltForQuestionAnswering

if is_vision_available():
    from PIL import Image
    from transformers import ViltProcessor

MODEL_NAME_OR_PATH = "dandelin/vilt-b32-finetuned-vqa"

def prepare_img():
    image = Image.open("000000039769.png")
    return image

@require_torch
@require_vision
class ViltModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_processor(self):
        return ViltProcessor.from_pretrained(MODEL_NAME_OR_PATH) if is_vision_available() else None

    def test_inference_visual_question_answering(self):
        model = ViltForQuestionAnswering.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)

        processor = self.default_processor
        image = prepare_img()
        text = "How many cats are there?"
        inputs = processor(image, text, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        expected_shape = torch.Size((1, 3129))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor([-15.9495, -18.1472, -10.3041]).to(torch_device)

        self.assertTrue(torch.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4))

        vqa_labels = [[2, 3, 155, 800]]
        vqa_scores = [[1.0, 0.3, 0.3, 0.3]]
        labels = torch.zeros(1, model.config.num_labels).to(torch_device)

        for i, (labels_example, scores_example) in enumerate(zip(vqa_labels, vqa_scores)):
            for l, s in zip(labels_example, scores_example):
                labels[i, l] = s

        outputs = model(**inputs, labels=labels)

        self.assertTrue(outputs.loss > 0)

if __name__=='__main__':
    unittest.main(argv=['', 'ViltModelIntegrationTest.test_inference_visual_question_answering'])
