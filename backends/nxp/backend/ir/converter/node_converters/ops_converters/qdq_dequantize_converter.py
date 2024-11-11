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


class QDQDequantizeConverter(NodeConverter):

    def convert(self, node: Node):
        assert "cluster" in node.meta, "Attempt to convert Dequantize node that is not part of the cluster!"

        from_tensor = self.builder.tensor_for_name(node.name)
        to_tensor = self.builder.tensor_for_name(node.args[0].name)

        assert node.args[5] == torch.int8, "Only INT8 conversion is currently supported"
        scale = np.array(node.args[1], dtype=np.float32)
        zero_point = np.array(node.args[2], dtype=np.int8)

        if self.context.parameters_mapping.get(node.args[0].name, None) is None:
            # Convert dequantize as identity op (Transpose that will be removed) because
            # input tensor is input of the model and don't have static data. If we do redirection
            # here we will change input name of the model.
            t_op = self._create_tflite_op_with_io_tensors(node)

            set_quantization_parameters_to_tensor(to_tensor, scale, zero_point, 0)
            set_quantization_parameters_to_tensor(from_tensor, scale, zero_point, 0)
            from_tensor.type = to_tensor.type

            self.builder.turn_operator_to_identity(t_op)
            self.builder.append_operators([t_op])
        else:
            # Dequantize consumes tensor with static data -> convert as a tensor
            set_quantization_parameters_to_tensor(to_tensor, scale, zero_point, 0)

            # Change type so we pass check tensor similarity check when redirecting
            from_tensor.type = to_tensor.type
            self.builder.redirect_tensor(from_tensor, to_tensor)
