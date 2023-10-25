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
""" Testing suite for the PyTorch YOLOS model. """

import unittest

from transformers.utils import is_torch_available, is_vision_available, cached_property
from transformers.testing_utils import torch_device, require_torch, require_vision

if is_torch_available():
    import torch
    from transformers import YolosForObjectDetection

if is_vision_available():
    from PIL import Image
    from transformers import AutoImageProcessor

MODEL_NAME_OR_PATH = "hustvl/yolos-small"

def prepare_img():
    image = Image.open("000000039769.png")
    return image

@require_torch
@require_vision
class YolosModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return AutoImageProcessor.from_pretrained(MODEL_NAME_OR_PATH) if is_vision_available() else None

    def test_inference_object_detection_head(self):
        model = YolosForObjectDetection.from_pretrained(MODEL_NAME_OR_PATH).to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(inputs.pixel_values)

        expected_shape = torch.Size((1, 100, 92))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice_logits = torch.tensor(
            [[-24.0248, -10.3024, -14.8290], [-42.0392, -16.8200, -27.4334], [-27.2743, -11.8154, -18.7148]],
            device=torch_device,
        )
        expected_slice_boxes = torch.tensor(
            [[0.2559, 0.5455, 0.4706], [0.2989, 0.7279, 0.1875], [0.7732, 0.4017, 0.4462]], device=torch_device
        )
        self.assertTrue(torch.allclose(outputs.logits[0, :3, :3], expected_slice_logits, atol=1e-4))
        self.assertTrue(torch.allclose(outputs.pred_boxes[0, :3, :3], expected_slice_boxes, atol=1e-4))

        results = image_processor.post_process_object_detection(
            outputs, threshold=0.3, target_sizes=[image.size[::-1]]
        )[0]
        expected_scores = torch.tensor([0.9994, 0.9790, 0.9964, 0.9972, 0.9861]).to(torch_device)
        expected_labels = [75, 75, 17, 63, 17]
        expected_slice_boxes = torch.tensor([335.0609, 79.3848, 375.4216, 187.2495]).to(torch_device)

        self.assertEqual(len(results["scores"]), 5)
        self.assertTrue(torch.allclose(results["scores"], expected_scores, atol=1e-4))
        self.assertSequenceEqual(results["labels"].tolist(), expected_labels)
        self.assertTrue(torch.allclose(results["boxes"][0, :], expected_slice_boxes))

if __name__=='__main__':
    unittest.main(argv=['', 'YolosModelIntegrationTest.test_inference_object_detection_head'])
