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
""" Testing suite for the PyTorch Decision_Transformer model. """

import unittest

from transformers.utils import is_torch_available
from transformers.testing_utils import torch_device, require_torch

if is_torch_available():
    import torch
    from transformers import DecisionTransformerModel

MODEL_NAME_OR_PATH = "edbeeching/decision-transformer-gym-hopper-expert"

@require_torch
class DecisionTransformerModelIntegrationTest(unittest.TestCase):
    def test_autoregressive_prediction(self):
        """
        An integration test that performs autoregressive prediction of state, action and return
        from a sequence of state, actions and returns. Test is performed over two timesteps.

        """

        NUM_STEPS = 2
        TARGET_RETURN = 10
        model = DecisionTransformerModel.from_pretrained(MODEL_NAME_OR_PATH)
        model = model.to(torch_device)
        config = model.config
        torch.manual_seed(0)
        state = torch.randn(1, 1, config.state_dim).to(device=torch_device, dtype=torch.float32)

        expected_outputs = torch.tensor(
            [[0.242793, -0.28693074, 0.8742613], [0.67815274, -0.08101085, -0.12952147]], device=torch_device
        )

        returns_to_go = torch.tensor(TARGET_RETURN, device=torch_device, dtype=torch.float32).reshape(1, 1, 1)
        states = state
        actions = torch.zeros(1, 0, config.act_dim, device=torch_device, dtype=torch.float32)
        rewards = torch.zeros(1, 0, device=torch_device, dtype=torch.float32)
        timesteps = torch.tensor(0, device=torch_device, dtype=torch.long).reshape(1, 1)

        for step in range(NUM_STEPS):
            actions = torch.cat([actions, torch.zeros(1, 1, config.act_dim, device=torch_device)], dim=1)
            rewards = torch.cat([rewards, torch.zeros(1, 1, device=torch_device)], dim=1)

            attention_mask = torch.ones(1, states.shape[1]).to(dtype=torch.long, device=states.device)

            with torch.no_grad():
                _, action_pred, _ = model(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    returns_to_go=returns_to_go,
                    timesteps=timesteps,
                    attention_mask=attention_mask,
                    return_dict=False,
                )

            self.assertEqual(action_pred.shape, actions.shape)
            self.assertTrue(torch.allclose(action_pred[0, -1], expected_outputs[step], atol=1e-4))
            state, reward, _, _ = (
                torch.randn(1, 1, config.state_dim).to(device=torch_device, dtype=torch.float32),
                1.0,
                False,
                {},
            )

            actions[-1] = action_pred[0, -1]
            states = torch.cat([states, state], dim=1)
            pred_return = returns_to_go[0, -1] - reward
            returns_to_go = torch.cat([returns_to_go, pred_return.reshape(1, 1, 1)], dim=1)
            timesteps = torch.cat(
                [timesteps, torch.ones((1, 1), device=torch_device, dtype=torch.long) * (step + 1)], dim=1
            )

if __name__=='__main__':
    unittest.main(argv=['', 'DecisionTransformerModelIntegrationTest.test_autoregressive_prediction'])
