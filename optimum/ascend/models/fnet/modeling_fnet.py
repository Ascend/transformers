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

def npu_fnetbasicfouriertransform_forward(self, hidden_states):
    # NOTE: We do not use torch.vmap as it is not integrated into PyTorch stable versions.
    # Interested users can modify the code to use vmap from the nightly versions, getting the vmap from here:
    # https://pytorch.org/docs/master/generated/torch.vmap.html. Note that fourier transform methods will need
    # change accordingly.

    outputs = self.fourier_transform(hidden_states.to("cpu")).real.to(hidden_states.device)
    return (outputs,)