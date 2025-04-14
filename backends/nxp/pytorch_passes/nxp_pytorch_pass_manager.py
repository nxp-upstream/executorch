# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Iterable

from torch.fx import GraphModule

from executorch.backends.nxp.pytorch_passes.fuse_batch_norm_with_conv_pass import FuseBatchNormWithConvPass
from executorch.backends.nxp.pytorch_passes.fuse_batch_norm_with_linear_pass import FuseBatchNormWithLinearPass
from executorch.backends.nxp.pytorch_passes.nxp_pytorch_pass import NXPPyTorchPass


class NXPPyTorchPassManager:
    """ Class iteratively calls provided passes which inherit from the `NXPPyTorchPass` class. """

    def __init__(self, module: GraphModule, passes: Iterable[type[NXPPyTorchPass]] | None = None):
        self.module = module
        self.passes = passes or [  # New passes should be added here.
            FuseBatchNormWithConvPass,
            FuseBatchNormWithLinearPass
        ]

    def _clean_up_graph_module(self):
        self.module.graph.eliminate_dead_code()
        self.module.recompile()

    def run(self) -> GraphModule:
        """ Iteratively apply all available passes for as long as they are changing the graph. """
        graph_module = self.module
        hard_limit = 10  # Empirical value.
        overall_made_changes = False

        self._clean_up_graph_module()

        for _ in range(hard_limit):
            for pass_class in self.passes:
                try:
                    pass_ = pass_class(graph_module)
                    made_changes = pass_.run()
                    overall_made_changes = overall_made_changes or made_changes
                    self._clean_up_graph_module()

                except Exception as e:
                    logging.warning(f'An exception occurred during the pre-processing pass `{pass_class}`. '
                                    'Please report this issue.\n' + str(e))

            if not overall_made_changes:
                break

        return graph_module
