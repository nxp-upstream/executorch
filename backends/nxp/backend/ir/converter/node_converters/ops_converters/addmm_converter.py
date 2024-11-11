# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.nxp.backend.ir.converter.conversion.common import OpsList
from executorch.backends.nxp.backend.ir.converter.node_converter import NodeConverter
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import fully_connected_options
from torch.fx import Node


class AddMMConverter(NodeConverter):
    """ Convert the `aten.addmm` operator to TFLite `FullyConnected` with a bias input. """

    def convert(self, node: Node):
        t_op = self._append_io_tensors_and_get_tflite_op(node)
        t_op.builtin_options = fully_connected_options.FullyConnected(keep_num_dims=True)

        assert len(t_op.tmp_inputs) == 3, f'`aten.addmm` has an unexpected number of inputs ({len(t_op.tmp_inputs)}).'
        bias = t_op.tmp_inputs[0]
        x = t_op.tmp_inputs[1]
        w = t_op.tmp_inputs[2]
        y = t_op.tmp_outputs[0]

        # Assign the operator its TFLite inputs and outputs
        t_op.tmp_inputs = [x, w, bias]
        t_op.tmp_outputs = [y]

        ops = OpsList(middle_op=t_op)

        # The `aten.addmm` uses main input with shape [M, N] and the weights have the shape [N, O].
        # TFLite `FullyConnected` requires the weights to have shape [O, N] (if the main input has shape [M, N]).
        # Insert a `Transpose` operator to permute the weights to achieve correct conversion. (The `Transpose` will not
        #  be present in the output model if the weights are static.)
        assert w.rank == 2, f'`aten.addmm` has weights with rank `{w.rank}`, which is not supported.'
        ops.add_pre(self.builder.create_transpose_operator_before(t_op, 1, [1, 0]))

        self._append_operators(ops.flatten())
