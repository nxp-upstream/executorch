# Copyright (c) 2024 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Quantizer for Neutron NPU.

import torch

from torch.ao.quantization.quantizer import Quantizer
from executorch.backends.arm.quantizer.arm_quantizer import  ArmQuantizer

class NeutronQuantizer(ArmQuantizer):

    def __init__(self):
        super().__init__()

    def transform_for_annotation(
            self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        return model


    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        return super().annotate(model)


    def validate(self, model: torch.fx.GraphModule) -> None:
        return super().validate(model)

