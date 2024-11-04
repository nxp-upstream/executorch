# Copyright (c) 2024 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.fx import Node

from executorch.backends.nxp.backend.ir import logger
from executorch.backends.nxp.backend.ir.converter.node_converter import NodeConverter
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters.convolution_converter import \
    ConvolutionConverter
from executorch.exir.dialects._ops import ops as exir_ops


class CallFunctionConverter(NodeConverter):
    method_converters = {
        exir_ops.edge.aten.convolution.default: ConvolutionConverter
    }

    def convert(self, node: Node):
        if node.target in self.method_converters:
            self.method_converters[node.target](self.context).convert(node)
        else:
            logger.e(logger.Code.NOT_IMPLEMENTED, f"Converter for '{node.target.__name__}' not implemented!")
