# Copyright (c) 2024 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from torch.fx import Node

from executorch.backends.nxp.backend.ir.converter.conversion import common
from executorch.backends.nxp.backend.ir.converter.conversion.common import try_get_input, OpsList
from executorch.backends.nxp.backend.ir.converter.node_converter import NodeConverter
from executorch.backends.nxp.backend.ir.lib.tflite import Padding
from executorch.backends.nxp.backend.ir.lib.tflite.TensorType import TensorType
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import conv_2d_options


class ConvolutionConverter(NodeConverter):

    def _convert_unpadded_2D(self, stride, dilation, t_op) -> OpsList:
        # Prepare the input and output tensors. To replace them, assign to t_op.tmp_inputs/tmp_outputs directly.
        output_tensor = t_op.tmp_outputs[0]
        input_tensor = t_op.tmp_inputs[0]
        weight_tensor = t_op.tmp_inputs[1]

        if (bias_tensor := try_get_input(t_op, 2)) is None:
            # Operator has no bias. Convolution aten op can omit it, TFLite can't.
            output_channels = weight_tensor.shape.vector[0]

            if weight_tensor.type == TensorType.FLOAT32:
                bias_type = np.dtype(np.float32)
            elif weight_tensor.type in [TensorType.INT8, TensorType.UINT8]:
                bias_type = np.dtype(np.int32)
            else:
                raise NotImplementedError(f"Convolution node with unsupported weight type: {weight_tensor.type}")

            bias_tensor = self.builder.create_zeros_tensor([output_channels], "zero_bias", bias_type, True)

        # Assign the operator its TFLite inputs and outputs
        t_op.tmp_inputs = [input_tensor, weight_tensor, bias_tensor]
        t_op.tmp_outputs = [output_tensor]

        ops_list = OpsList()
        ops_list.middle_op = t_op

        # Convert the builtin options
        common.assign_2d_strides(t_op.builtin_options, stride)
        common.assign_2d_dilations(t_op.builtin_options, dilation)

        return ops_list

    def _convert_2d_conv(self, stride, dilation, t_op: tflite_model.Operator) -> list[tflite_model.Operator]:
        t_op.builtin_options = conv_2d_options.Conv2D()
        t_op.builtin_options.padding = Padding.Padding.VALID
        ops_list = self._convert_unpadded_2D(stride, dilation, t_op)
        # TODO Lukas: Try to reuse padding calculation from onnx2tflite
        # padding, pad_op = conv_utils.build_input_tensor_padding(conv_attributes, t_op, self.builder)
        # t_op.builtin_options.padding = padding

        # if pad_op is not None:
        #     conversion_result.ops_list.add_pre(pad_op)

        return ops_list.flatten()

    def convert(self, node: Node):
        x = node.args[0]
        weight = node.args[1]
        bias: Node | None = node.args[2]
        stride = node.args[3]
        padding = node.args[4]
        dilation = node.args[5]
        is_transposed = node.args[6]
        output_padding = node.args[7]
        groups = node.args[8]

        assert padding == [0, 0], "'padding' attribute not yet supported"
        assert not is_transposed, "'is_transposed' attribute not yet supported"
        assert output_padding == [0, 0], "'output_padding' attribute not yet supported"
        assert groups == 1, "'groups' attribute not yet supported"

        t_op = self._create_tflite_op_with_io_tensors(node)
        ops_to_add = self._convert_2d_conv(stride, dilation, t_op)
        self.builder.append_operators(ops_to_add)
