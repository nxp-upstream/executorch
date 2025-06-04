# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.nxp.backend.ir.converter.node_converter import NodeConverter
from executorch.backends.nxp.backend.ir.converter.quantization_utils import (
    propagate_quantization,
)
from torch.fx import Node
from torch.nn import Parameter


class GetItemConverter(NodeConverter):

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
    ) -> bool:
        return True

    def convert(self, node: Node):
        """Skip the `GetItem` node, as it serves no purpose in the IR."""
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)

        # It is possible (and common) for the `GetItem` to be a part of a QDQ cluster. In these cases, it consumes the
        #  output of the main compute operator, and it is followed by a `Quantize` operator, which specifies the output
        #  quantization parameters of the cluster. So the input of the `GetItem` is float32, and the output is
        #  quantized. Therefore, the quantization must be propagated from the output to the input.
        input_ = t_op.tmp_inputs[0]
        output = t_op.tmp_outputs[0]
        if input_.quantization is None and output.quantization is not None:
            input_.type = output.type
            propagate_quantization(from_tensor=output, to_tensor=input_)

        self.builder.turn_operator_to_identity(t_op)
        self.builder.append_operators([t_op])
