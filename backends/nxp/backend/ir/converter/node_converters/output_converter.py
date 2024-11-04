# Copyright (c) 2024 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.fx import Node

from executorch.backends.nxp.backend.ir.converter.node_converter import NodeConverter


class OutputConverter(NodeConverter):

    def convert(self, node: Node):
        # (TODO Lukas) We currently ignore output nodes because it's only mapping from last function's
        # output to some additional node (%output = %aten_conv). We need to implement some logic to map
        # such tensors (or insert some NOP in between them) in TFLite if necessary.
        pass
