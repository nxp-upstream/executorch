
# Copyright (c) 2024 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.fx import Node

from executorch.backends.nxp.backend.ir.converter.node_converter import NodeConverter
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import softmax_options
from executorch.backends.nxp.backend.ir.converter.conversion import translator


class SoftmaxConverter(NodeConverter):

    def _normalize_dim(self, dim, rank):
        # convert negative index to positive
        if dim < 0:
            dim += rank
        return dim

    def convert(self, node: Node):
        x = node.args[0]
        dim = node.args[1]

        rank = len(x.meta["val"].shape)
        dim = self._normalize_dim(dim, rank)

        t_op = self._create_tflite_op_with_io_tensors(node)
        if t_op.tmp_inputs[0].tensor_format.is_channels_last():
            dim = translator.create_channels_last_to_channels_first_permutation(rank)[dim]

        assert dim == rank - 1, "'dim' different to last dim not yet supported"
        t_op.builtin_options = softmax_options.Softmax(beta=1.0)

        self.builder.append_operators([t_op])
