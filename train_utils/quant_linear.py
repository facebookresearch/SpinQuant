# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch._tensor import Tensor


class QuantizeLinear(nn.Linear):
    def forward(
        self,
        input: Tensor,
        R1=None,
        R2=None,
        transpose=False,
    ) -> Tensor:
        # quantize weight
        if R1 is not None:
            dtype = self.weight.dtype
            if not transpose:
                weight = (self.weight.to(torch.float64) @ R1.to(torch.float64)).to(
                    dtype
                )
            else:
                weight = (R1.T.to(torch.float64) @ self.weight.to(torch.float64)).to(
                    dtype
                )
            if R2 is not None:
                # Each head dim = 128 for Llama model
                had_dim = R2.shape[0]
                dtype = weight.dtype
                if transpose:
                    W_ = weight
                    init_shape = W_.shape
                    temp = W_.reshape(-1, init_shape[-1] // had_dim, had_dim)
                    temp = temp.to(torch.float64) @ R2.to(torch.float64)
                    weight = temp.reshape(init_shape)
                else:
                    W_ = weight.t()
                    transposed_shape = W_.shape
                    temp = W_.reshape(-1, transposed_shape[-1] // had_dim, had_dim)
                    temp = temp.to(torch.float64) @ R2.to(torch.float64)
                    weight = temp.reshape(transposed_shape).t()
            weight = weight.to(dtype)
        else:
            weight = self.weight
        if hasattr(self, "quantizer"):
            dtype = weight.dtype
            self.quantizer.find_params(weight.data)
            weight = self.quantizer.quantize(weight).to(dtype)

        return nn.functional.linear(input, weight, self.bias)
