# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod, ABC

from torch.fx import GraphModule
from torch.nn.parameter import Parameter


class NXPPyTorchPass(ABC):
    """ Abstract parent class for pre-processing passes on the aten dialect level. """

    def __init__(self, module: GraphModule) -> None:
        super().__init__()
        self.module = module

    @abstractmethod
    def run(self) -> bool:
        """ Execute the pass and return a bool indicating if any changes have been made. """
        pass

    def get_tensor_constant_from_node(self, node) -> Parameter | None:
        """ Get the static data from a given node. If it doesn't have any data, return `None`. """
        if node is None or node.op != 'get_attr':
            return None

        target_atoms = node.target.split('.')
        attr_itr = self.module
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                return None
            attr_itr = getattr(attr_itr, atom)
        return attr_itr
