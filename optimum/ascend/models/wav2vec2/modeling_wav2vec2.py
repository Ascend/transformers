# Copyright 2023 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright 2021 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
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

from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForPreTrainingOutput


def npu_wav2vec2forpretraining_forward(
    self,
    input_values: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    mask_time_indices: Optional[torch.BoolTensor] = None,
    sampled_negative_indices: Optional[torch.BoolTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, Wav2Vec2ForPreTrainingOutput]:
    """
    Copied from Wav2Vec2ForPreTraining: https://github.com/huggingface/transformers/blob/main/src/tranformers/models/wav2vec2/modeling_wav2vec2.py
    The only difference is:
    - `nn.functional.cross_entropy` is evaluated on cpu
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if mask_time_indices is not None:
        mask_time_indices = mask_time_indices.to(torch.bool)

    outputs = self.wav2vec2(
        input_values,
        attention_mask=attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        mask_time_indices=mask_time_indices,
        return_dict=return_dict,
    )

    # 1. project all transformed features (including masked) to final vq dim
    transformer_features = self.project_hid(outputs[0])

    # 2. quantize all (unmasked) extracted features and project to final vq dim
    extract_features = self.dropout_features(outputs[1])

    if attention_mask is not None:
        # compute reduced attention_mask correponding to feature vectors
        attention_mask = self._get_feature_vector_attention_mask(
            extract_features.shape[1], attention_mask, add_adapter=False
        )

    quantized_features, codevector_perplexity = self.quantizer(
        extract_features, mask_time_indices=mask_time_indices
    )
    quantized_features = self.project_q(quantized_features)

    loss = contrastive_loss = diversity_loss = None
    if sampled_negative_indices is not None:
        batch_size, sequence_length, hidden_size = quantized_features.shape

        # for training, we sample negatives
        # 3. sample K negatives (distractors) quantized states for contrastive loss
        # if attention_mask is passed, make sure that padded feature vectors cannot be sampled
        # sample negative quantized vectors BTC => (BxT)C
        negative_quantized_features = quantized_features.view(-1, hidden_size)[
            sampled_negative_indices.long().view(-1)
        ]
        negative_quantized_features = negative_quantized_features.view(
            batch_size, sequence_length, -1, hidden_size
        ).permute(2, 0, 1, 3)

        # 4. compute logits, corresponding to `logs = sim(c_t, [q_t, \sim{q}_t]) / \kappa`
        # of equation (3) in https://arxiv.org/pdf/2006.11477.pdf
        logits = self.compute_contrastive_logits(
            quantized_features[None, :],
            negative_quantized_features,
            transformer_features,
            self.config.contrastive_logits_temperature,
        )

        # 5. if a negative vector is identical to the positive (i.e. when codebook utilization is low),
        # its cosine similarity will be masked
        neg_is_pos = (quantized_features == negative_quantized_features).all(-1)

        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")

        # 6. compute contrastive loss \mathbf{L}_m = cross_entropy(logs) =
        # -log(exp(sim(c_t, q_t)/\kappa) / \sum_{\sim{q}} exp(sim(c_t, \sim{q})/\kappa))
        logits = logits.transpose(0, 2).reshape(-1, logits.size(0))
        target = ((1 - mask_time_indices.long()) * -100).transpose(0, 1).flatten()

        contrastive_loss = nn.functional.cross_entropy(
            logits.double().to("cpu"), target.to("cpu"), reduction="sum"
        ).float().to(target.device)
        # 7. compute diversity loss: \mathbf{L}_d
        num_codevectors = self.config.num_codevectors_per_group * self.config.num_codevector_groups
        diversity_loss = ((num_codevectors - codevector_perplexity) / num_codevectors) * mask_time_indices.sum()

        # 8. \mathbf{L} = \mathbf{L}_m + \alpha * \mathbf{L}_d
        loss = contrastive_loss + self.config.diversity_loss_weight * diversity_loss

    if not return_dict:
        if loss is not None:
            return (loss, transformer_features, quantized_features, codevector_perplexity) + outputs[2:]
        return (transformer_features, quantized_features, codevector_perplexity) + outputs[2:]

    return Wav2Vec2ForPreTrainingOutput(
        loss=loss,
        projected_states=transformer_features,
        projected_quantized_states=quantized_features,
        codevector_perplexity=codevector_perplexity,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        contrastive_loss=contrastive_loss,
        diversity_loss=diversity_loss,
    )
