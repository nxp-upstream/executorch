# Copyright (c) 2024 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.fx import Node

from executorch.backends.nxp.backend.ir.converter.node_converter import NodeConverter


class PlaceholderConverter(NodeConverter):

    def convert(self, node: Node):
        if node.name in self.context.parameters_mapping:
            self._append_as_static_tensor(node)
        else:
            self._append_as_fake_tensor(node)

