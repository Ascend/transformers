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
""" Testing suite for the PyTorch Deta model. """

import unittest

from transformers import is_torch_available, is_torchvision_available, is_vision_available
from transformers.file_utils import cached_property
from transformers.testing_utils import require_torchvision, require_vision, torch_device

if is_torch_available():
    import torch

if is_torchvision_available():
    from transformers import DetaForObjectDetection

if is_vision_available():
    from PIL import Image
    from transformers import AutoImageProcessor

MODEL_NAME_OR_PATH = "jozhang97/deta-resnet-50"

def prepare_img():
    image = Image.open("000000039769.png")
    return image

@require_torchvision
@require_vision
class DetaModelIntegrationTests(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return AutoImageProcessor.from_pretrained(MODEL_NAME_OR_PATH) if is_vision_available() else None

    def test_inference_object_detection_head(self):
        model = DetaForObjectDetection.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        expected_shape_logits = torch.Size((1, 300, model.config.num_labels))
        self.assertEqual(outputs.logits.shape, expected_shape_logits)

        expected_logits = torch.tensor(
            [[-7.3978, -2.5406, -4.1668], [-8.2684, -3.9933, -3.8096], [-7.0515, -3.7973, -5.8516]]
        ).to(torch_device)
        expected_boxes = torch.tensor(
            [[0.5043, 0.4973, 0.9998], [0.2542, 0.5489, 0.4748], [0.5490, 0.2765, 0.0570]]
        ).to(torch_device)

        self.assertTrue(torch.allclose(outputs.logits[0, :3, :3], expected_logits, atol=1e-4))

        expected_shape_boxes = torch.Size((1, 300, 4))
        self.assertEqual(outputs.pred_boxes.shape, expected_shape_boxes)
        self.assertTrue(torch.allclose(outputs.pred_boxes[0, :3, :3], expected_boxes, atol=1e-4))

        results = image_processor.post_process_object_detection(
            outputs, threshold=0.3, target_sizes=[image.size[::-1]]
        )[0]
        expected_scores = torch.tensor([0.6392, 0.6276, 0.5546, 0.5260, 0.4706], device=torch_device)
        expected_labels = [75, 17, 17, 75, 63]
        expected_slice_boxes = torch.tensor([40.5866, 73.2107, 176.1421, 117.1751], device=torch_device)

        self.assertTrue(torch.allclose(results["scores"], expected_scores, atol=1e-4))
        self.assertSequenceEqual(results["labels"].tolist(), expected_labels)
        self.assertTrue(torch.allclose(results["boxes"][0, :], expected_slice_boxes))

if __name__=='__main__':
    unittest.main(argv=['', 'DetaModelIntegrationTests.test_inference_object_detection_head'])
