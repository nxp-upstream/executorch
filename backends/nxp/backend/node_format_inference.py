# Copyright (c) 2024 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import abc
from enum import Enum

from torch import Node
from torch.export import ExportedProgram
from typing_extensions import Tuple

# (TODO Lukas) Can we found ops somewhere else?
from executorch.exir.dialects._ops import ops as exir_ops

logger = logging.getLogger(__name__)


class NodeFormat(Enum):
    # Node's output in NCHW format
    CHANNELS_FIRST = 0

    # Node's output format has no meaning
    FORMATLESS = 1

    # Format has not been identified
    NONE = 2

    def is_channels_first(self) -> bool:
        return self == NodeFormat.CHANNELS_FIRST


class NodeFormatInference:
    # Dictionary with Edge Aten ops that always use channels first format.
    # The op in the dictionary is mapped to a dictionary, which holds indices to input nodes
    # that are always channels first.
    ops_with_channels_first_nodes = {
        exir_ops.edge.aten.convolution.default: {"inputs": [0, 1]},
    }

    # A set of Edge Aten ops, which have the ability to change the format (for example - input nodes
    # are channels first but output is formatless).
    ops_that_can_change_tensor_format = {
        # TODO ("transpose", "reshape", etc.)
    }

    _node_format_mapping: dict[Node, NodeFormat]

    _type_changed_during_last_run: bool

    # Mapping between Node and its ancestors (inputs)
    _node_inputs: dict[Node, list[Node]]

    # Mapping between Node and its children (outputs)
    _node_outputs: dict[Node, list[Node]]

    def __init__(self, edge_program: ExportedProgram):
        self._edge_program = edge_program

        self._nodes = edge_program.graph.nodes
        self._node_format_mapping = {}
        self._node_inputs = {node: node.all_input_nodes for node in edge_program.graph.nodes}
        self._node_outputs = {node: list(node.users.keys()) for node in edge_program.graph.nodes}

        self._type_changed_during_last_run = False

    def identify_node_formats(self) -> dict[Node, NodeFormat]:
        self._type_changed_during_last_run = True

        # Re-run format inference until there are no changes
        while self._type_changed_during_last_run:
            self._type_changed_during_last_run = False

            for node in self._nodes:
                self._infer_format_of_nodes(node)

        return self._node_format_mapping

    def _infer_format_of_nodes(self, node: Node):
        op_type = self._get_node_op_type(node)

        if op_type in self.ops_with_channels_first_nodes:
            self._handle_node_which_uses_channels_first_format(node)
        elif op_type in self.ops_that_can_change_tensor_format:
            if op_type in ["transpose"]:
                self._assign_format_to_node(node, NodeFormat.FORMATLESS)
            else:
                logger.error(f"Node format inference for node type: {op_type} not found!")
        else:
            self._handle_node_which_can_use_any_node_format(node)

    def _assign_format_to_node(self, node: Node, node_format: NodeFormat):
        """
        Assign format to node, but only if it's not channels first.
        """
        old_node_format = self._get_node_format(node)

        if old_node_format is NodeFormat.CHANNELS_FIRST:
            # Once CHANNEL_FIRST was assigned, we don't want to reassign
            return

        if old_node_format != node_format:
            self._type_changed_during_last_run = True

        self._node_format_mapping[node] = node_format

    def _get_node_op_type(self, node: Node) -> str | None:
        """
        Get node's operation type or None if node is not callable function.
        """
        if node.op == "call_function":
            return node.target

        return None

    def _handle_node_which_uses_channels_first_format(self, node: Node):
        """
        Function for assigning format to nodes that require channels first input (Conv, MaxPool etc.)
        """
        op_type = self._get_node_op_type(node)

        for index, ancestor_node in enumerate(self._node_inputs[node]):
            # Go through input nodes and assign them correct format
            if index in self.ops_with_channels_first_nodes[op_type]["inputs"]:
                self._assign_format_to_node(ancestor_node, NodeFormat.CHANNELS_FIRST)

                # We need to propagate channels first format up to already visited nodes
                self._propagate_channels_first_format_up(ancestor_node)
            else:
                self._assign_format_to_node(ancestor_node, NodeFormat.FORMATLESS)

        # (TODO Lukas): It is expected here, that CHANNELS_FIRST node always produces CHANNELS_FIRST output.
        # Validate the assumption.
        self._assign_format_to_node(node, NodeFormat.CHANNELS_FIRST)

    def _handle_node_which_can_use_any_node_format(self, node: Node):
        """
        Function for assigning format to nodes that don't care about format (Softmax, Abs).
        It stays formatless if there is no surrounding channels first ancestor/child node.
        """
        if not self._node_produces_or_consumes_channels_first_format(node):
            # Nor inputs or current node are channels first -> assign everything to formatless
            for processed_node in self._node_inputs[node] + [node]:
                self._assign_format_to_node(processed_node, NodeFormat.FORMATLESS)

        else:
            # Node produces or consumes channels first content
            for processed_node in self._node_inputs[node] + [node]:
                is_0d_to_2d = self._node_product_has_0_to_2_dimensions(processed_node)

                if self._get_node_format(processed_node).is_channels_first():
                    # Node output already channel first
                    continue
                elif is_0d_to_2d:
                    # Node has less than 3 dimensions so it cannot be considered CHANNELS_FIRST
                    self._assign_format_to_node(processed_node, NodeFormat.FORMATLESS)
                else:
                    # Node has more than 2D output -> make it channels first
                    self._assign_format_to_node(processed_node, NodeFormat.CHANNELS_FIRST)
                    self._propagate_channels_first_format_up(processed_node)

    def _propagate_channels_first_format_up(self, node: Node):
        if self._node_is_placeholder(node):
            # Input or buffer node -> there is no parent node so we can end propagation here
            self._assign_format_to_node(node, NodeFormat.CHANNELS_FIRST)
            return

        if node in self.ops_that_can_change_tensor_format:
            # Propagation ends here because processed node changing format.
            return

        for ancestor_node in self._node_inputs[node]:
            # Propagate channels first to ancestor nodes
            self._infer_format_of_nodes(ancestor_node)

    def _node_product_has_0_to_2_dimensions(self, node: Node) -> bool:
        assert "val" in node.meta, f"Node '{node.name}' doesn't contain 'val' metadata!"

        node_value_meta = node.meta["val"]

        # (TODO Lukas): Some nodes contains multiple value metadata (MaxPool, ...). Find out why.
        if isinstance(node_value_meta, tuple):
            node_value_meta = node_value_meta[0]
        elif isinstance(node_value_meta, list):
            node_value_meta = node_value_meta[0]

        node_output_rank = len(node_value_meta.shape)

        return 0 <= node_output_rank <= 2

    def _node_produces_or_consumes_channels_first_format(self, node) -> bool:
        """
        Check if node itself produces output in channels first format or consumes it from ancestor node.
        """
        if self._get_node_format(node).is_channels_first():
            return True

        input_nodes = self._node_inputs[node]
        return any(
            self._get_node_format(ancestor_node).is_channels_first() for ancestor_node in input_nodes)

    def _get_node_format(self, node):
        return self._node_format_mapping.get(node, NodeFormat.NONE)

    def _node_is_placeholder(self, node: Node):
        return node.op == "placeholder"
