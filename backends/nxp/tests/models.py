# Copyright (c) 2024 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch


class Conv2dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = torch.nn.Conv2d(
            in_channels=4, out_channels=8, kernel_size=3, bias=True, stride=2, dilation=1
        )

    def forward(self, x):
        return self.conv(x)


class SoftmaxModule(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.softmax = torch.nn.Softmax(dim=dim)

    def forward(self, x):
        return self.softmax(x)


class SoftmaxConvModule(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.conv = Conv2dModule()
        self.softmax = SoftmaxModule(dim=dim)

    def forward(self, x):
        x = self.conv(x)
        return self.softmax(x)

class LinearModule(torch.nn.Module):
    def __init__(self, bias: bool):
        super().__init__()
        self.linear = torch.nn.Linear(32, 16, bias=bias)

    def forward(self, x):
        return self.linear(x)
