# Copyright (c) 2024 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Quantizer for Neutron NPU.

from typing import List, Type

import torch
from torch import fx
from torch.ao.quantization.quantizer.composable_quantizer import ComposableQuantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import QuantizationConfig
from torch.ao.quantization.quantizer import QuantizationSpec
from torch.ao.quantization.observer import HistogramObserver, MinMaxObserver, PerChannelMinMaxObserver

from executorch.backends.cadence.aot.quantizer.patterns import (
    AddmmPattern,
    Conv1dPattern,
    Conv2dPattern,
    LinearPattern,
)
from executorch.backends.cadence.aot.quantizer.quantizer import CadenceGenericQuantizer

# Quantization Specification used by Neutron NPU
act_qspec = QuantizationSpec(
    dtype=torch.int8,
    quant_min=-128,
    quant_max=127,
    qscheme=torch.per_tensor_affine,
    is_dynamic=False,
    observer_or_fake_quant_ctr=HistogramObserver.with_args(eps=2 ** -12),
)

wgt_qspec = QuantizationSpec(
    dtype=torch.int8,
    quant_min=-128,
    quant_max=127,
    qscheme=torch.per_channel_symmetric,
    is_dynamic=False,
    observer_or_fake_quant_ctr=PerChannelMinMaxObserver,
    ch_axis=0
)

wgt_fc_qspec = QuantizationSpec(
    dtype=torch.int8,
    quant_min=-128,
    quant_max=127,
    qscheme=torch.per_tensor_symmetric,
    is_dynamic=False,
    observer_or_fake_quant_ctr=MinMaxObserver,
)
# Bias Quantization Specification is as follows:
#   dtype = torch.int32
#   quant_min, quant_max - full int32 range
#   qcheme = torch.per_channel_symetric (for Conv), torch.per_tensor_symetric for Addmmn ==> i.e. zero_point = 0
#   scale = input_scale * weight_scale
# Is set by the *PatternQuantizer directly.
bias_qspec = None

class NeutronQuantizer(ComposableQuantizer):
    def __init__(self):
        static_qconfig = QuantizationConfig(
            act_qspec,
            act_qspec,
            wgt_qspec,
            None,
        )
        static_fc_qconfig = QuantizationConfig(
            act_qspec,
            act_qspec,
            wgt_fc_qspec,
            None
        )
        super().__init__(
            [
                CadenceGenericQuantizer(AddmmPattern(), static_fc_qconfig), # TODO need to be verified, not use by CifarNet
                CadenceGenericQuantizer(Conv1dPattern(), static_qconfig),
                CadenceGenericQuantizer(Conv2dPattern(), static_qconfig),
                CadenceGenericQuantizer(LinearPattern(), static_fc_qconfig),
            ]
        )

    def transform_for_annotation(
            self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        return model

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        return super().annotate(model)

    def validate(self, model: torch.fx.GraphModule) -> None:
        return super().validate(model)
