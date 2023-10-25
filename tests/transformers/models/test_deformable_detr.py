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
""" Testing suite for the PyTorch Deformable DETR model. """

import unittest

from transformers.utils import is_torch_available, is_vision_available
from transformers.testing_utils import torch_device, require_torch, require_vision
from transformers.file_utils import cached_property

if is_torch_available():
    import torch
    from transformers import DeformableDetrForObjectDetection

if is_vision_available():
    from PIL import Image
    from transformers import AutoImageProcessor

MODEL_NAME_OR_PATH = "SenseTime/deformable-detr"

def prepare_img():
    image = Image.open("000000039769.png")
    return image

@require_vision
class DeformableDetrModelIntegrationTests(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return AutoImageProcessor.from_pretrained(MODEL_NAME_OR_PATH) if is_vision_available() else None

    def test_inference_object_detection_head(self):
        model = DeformableDetrForObjectDetection.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        encoding = image_processor(images=image, return_tensors="pt").to(torch_device)
        pixel_values = encoding["pixel_values"].to(torch_device)
        pixel_mask = encoding["pixel_mask"].to(torch_device)

        with torch.no_grad():
            outputs = model(pixel_values, pixel_mask)

        expected_shape_logits = torch.Size((1, model.config.num_queries, model.config.num_labels))
        self.assertEqual(outputs.logits.shape, expected_shape_logits)

        expected_logits = torch.tensor(
            [[-9.6645, -4.3449, -5.8705], [-9.7035, -3.8504, -5.0724], [-10.5634, -5.3379, -7.5116]]
        ).to(torch_device)
        expected_boxes = torch.tensor(
            [[0.8693, 0.2289, 0.2492], [0.3150, 0.5489, 0.5845], [0.5563, 0.7580, 0.8518]]
        ).to(torch_device)

        self.assertTrue(torch.allclose(outputs.logits[0, :3, :3], expected_logits, atol=1e-4))

        expected_shape_boxes = torch.Size((1, model.config.num_queries, 4))
        self.assertEqual(outputs.pred_boxes.shape, expected_shape_boxes)
        self.assertTrue(torch.allclose(outputs.pred_boxes[0, :3, :3], expected_boxes, atol=1e-4))

        results = image_processor.post_process_object_detection(
            outputs, threshold=0.3, target_sizes=[image.size[::-1]]
        )[0]
        expected_scores = torch.tensor([0.7999, 0.7894, 0.6331, 0.4720, 0.4382]).to(torch_device)
        expected_labels = [17, 17, 75, 75, 63]
        expected_slice_boxes = torch.tensor([16.5028, 52.8390, 318.2544, 470.7841]).to(torch_device)

        self.assertEqual(len(results["scores"]), 5)
        self.assertTrue(torch.allclose(results["scores"], expected_scores, atol=1e-4))
        self.assertSequenceEqual(results["labels"].tolist(), expected_labels)
        self.assertTrue(torch.allclose(results["boxes"][0, :], expected_slice_boxes))

if __name__=='__main__':
    unittest.main(argv=['', 'DeformableDetrModelIntegrationTests.test_inference_object_detection_head'])
