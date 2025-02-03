# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Iterable

import itertools
from torch.fx import GraphModule

from executorch.backends.nxp.pytorch_passes.fuse_batch_norm_with_conv_pass import FuseBatchNormWithConvPass
from executorch.backends.nxp.pytorch_passes.nxp_pytorch_pass import NXPPyTorchPass


class NXPPyTorchPassManager:
    """ Class iteratively calls provided passes which inherit from the `NXPPyTorchPass` class. """

    def __init__(self, module: GraphModule, passes: Iterable[type[NXPPyTorchPass]] | None = None):
        self.module = module
        self.passes = passes or [  # New passes should be added here.
            FuseBatchNormWithConvPass,
        ]

    def _clean_up_graph_module(self):
        self.module.graph.eliminate_dead_code()
        self.module.recompile()

    def run(self) -> GraphModule:
        """ Iteratively apply all available passes for as long as they are changing the graph. """
        graph_module = self.module
        num_passes = len(self.passes)
        hard_limit = 10 * num_passes  # Empirical value.
        num_passes_since_last_change = 0

        self._clean_up_graph_module()

        # Cycle through all passes as long as they are making changes.
        for i, pass_class in enumerate(itertools.cycle(self.passes)):
            try:
                pass_ = pass_class(graph_module)
                made_changes = pass_.run()
                self._clean_up_graph_module()

                if made_changes:
                    num_passes_since_last_change = 0
                else:
                    num_passes_since_last_change += 1

                if num_passes_since_last_change >= num_passes or i >= hard_limit:
                    break

            except Exception as e:
                logging.warning(f'An exception occurred during the pre-processing pass `{pass_class}`. '
                                'Please report this issue.\n' + str(e))

        return graph_module
