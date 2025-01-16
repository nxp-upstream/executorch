# Copyright (c) 2024 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.fx import Node

from executorch.backends.nxp.backend.ir.converter.conversion import common
from executorch.backends.nxp.backend.ir.converter.node_converter import NodeConverter, Target
from executorch.backends.nxp.backend.ir.lib.tflite import Padding
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import max_pool_2d_options


class Maxpool2dConverter(NodeConverter):
    """ Convert 'max_pool2d' operator to TFLite 'MaxPool2D'.
        NOTE: max_pool2d_with_indices is a different operator and is unsupported.
    """
    supported_targets = [Target.RT700]

    @staticmethod
    def _is_supported_in_IR(node: Node) -> bool:
        n_args = len(node.args)

        padding = node.args[3] if n_args >= 4 else [0, 0]
        dilation = node.args[4] if n_args >= 5 else [1, 1]
        ceil_mode = node.args[5] if n_args == 6 else False

        if padding != [0, 0] or \
            any(dil != 1 for dil in dilation) or \
            ceil_mode:
            return False

        return True

    # noinspection PyMethodMayBeStatic
    def _convert_2d_max_pool(self, kernel_size, stride, padding, t_op: tflite_model.Operator
                             ) -> list[tflite_model.Operator]:
        t_op.builtin_options = max_pool_2d_options.MaxPool2D()
        t_op.builtin_options.padding = Padding.Padding.VALID
        # t_op.tmp_inputs and t_op.tmp_outputs don't need any changes

        common.assign_2d_strides(t_op.builtin_options, stride)
        t_op.builtin_options.filter_h = kernel_size[0]
        t_op.builtin_options.filter_w = kernel_size[1]

        return [t_op]

    # Maxpool2d Node format: (Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False)
    def convert(self, node: Node):
        self.assert_convertible(node)

        n_args = len(node.args)

        kernel_size = node.args[1]
        stride = node.args[2]
        padding = node.args[3] if n_args >= 4 else [0, 0]

        t_op = self._create_tflite_op_with_io_tensors(node)
        ops_to_add = self._convert_2d_max_pool(kernel_size, stride, padding, t_op)
        self.builder.append_operators(ops_to_add)
