# Copyright (c) 2024 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch.fx import Node

from executorch.backends.nxp.backend.ir.converter.node_converter import NodeConverter
from executorch.backends.nxp.backend.ir.converter.quantization_utils import set_quantization_parameters_to_tensor


class QDQQuantizeConverter(NodeConverter):

    def convert(self, node: Node):
        assert "cluster" in node.meta, "Attempt to convert Quantize node that is not part of the cluster!"

        from_tensor = self.builder.tensor_for_name(node.name)
        to_tensor = self.builder.tensor_for_name(node.args[0].name)

        assert node.args[5] == torch.int8, "Only INT8 conversion is currently supported"
        scale = np.array(node.args[1], dtype=np.float32)
        zero_point = np.array(node.args[2], dtype=np.int8)

        set_quantization_parameters_to_tensor(to_tensor, scale, zero_point, 0)

        # Change type so we pass check tensor similarity check when redirecting
        to_tensor.type = from_tensor.type
        self.builder.redirect_tensor(from_tensor, to_tensor)
