# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from executorch.backends.nxp.backend.ir.converter.conversion.common import OpsList
from executorch.backends.nxp.backend.ir.converter.node_converter import NodeConverter
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import transpose_options
from torch.fx import Node


class PermuteCopyConverter(NodeConverter):

    def convert(self, node: Node):
        """ Convert the `aten.permute_copy` operator to TFLite `Transpose`. """
        t_op = self._append_io_tensors_and_get_tflite_op(node)
        t_op.builtin_options = transpose_options.Transpose()

        output_tensor = t_op.tmp_outputs[0]
        x = t_op.tmp_inputs[0]

        perm = np.array(node.args[1], 'int32')
        perm_tensor = self.builder.create_tensor_for_data(perm, 'perm')

        # Assign the operator its TFLite inputs and outputs
        t_op.tmp_inputs = [x, perm_tensor]
        t_op.tmp_outputs = [output_tensor]

        ops_to_add = OpsList(middle_op=t_op)

        self._append_operators(ops_to_add.flatten())
