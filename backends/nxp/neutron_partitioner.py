# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Partitioner for the NXP Neutron NPU

import logging
import operator
from typing import final, List

import torch
from executorch.backends.nxp.nxp_backend import NeutronBackend
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export.exported_program import ExportedProgram
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner

from torch.fx.passes.operator_support import OperatorSupportBase

# These operators are directly supported by Neutron Kernels
NeutronSupportedOperatorsList = [
    # exir_ops.edge.aten._softmax.default,
    # exir_ops.edge.aten.abs.default,
    # exir_ops.edge.aten.add.Scalar,
    # exir_ops.edge.aten.bmm.default,
    # exir_ops.edge.aten.clamp.default,
    # exir_ops.edge.aten.constant_pad_nd.default,
    exir_ops.edge.aten.convolution.default,
    # exir_ops.edge.aten.leaky_relu.default,
    # exir_ops.edge.aten.logit.default,
    exir_ops.edge.aten.max_pool2d_with_indices.default,
    # exir_ops.edge.aten.mean.dim,
    # exir_ops.edge.aten.mm.default,
    # exir_ops.edge.aten.mul.Scalar,
    # exir_ops.edge.aten.relu.default,
    # exir_ops.edge.aten.slice_copy.Tensor,
    # exir_ops.edge.aten.sub.Scalar,
    # exir_ops.edge.aten.tanh.default,
    # operator.getitem,
]

class NeutronSupportedOperators(OperatorSupportBase):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        """
        Check if the PyTorch op that gets called is supported for Neutron
        or if it is part of a QDQ cluster.
        """
        is_neutron_supported_op = node.op == "call_function" and node.target in NeutronSupportedOperatorsList
        is_part_of_the_cluster = "cluster" in node.meta

        # TODO(Lukas) Naive approach. This will be replaced by 'getitem' recognition in QDQ clustering
        is_maxpool_getitem = (node.name.startswith("getitem") and
                              node.all_input_nodes[0].target == exir_ops.edge.aten.max_pool2d_with_indices.default)

        return is_neutron_supported_op or is_part_of_the_cluster or is_maxpool_getitem

@final
class NeutronPartitioner(Partitioner):
    def __init__(self, compile_spec: List[CompileSpec]) -> None:
        self.delegation_spec = DelegationSpec(NeutronBackend.__name__, compile_spec)

    def is_quant_node(self, node: torch.fx.node.Node):
        return node.target in {
            exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor,
        }
    
    def is_dequant_node(self, node: torch.fx.node.Node):
        return node.target in {
            exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor,
        }

    def tag_clusters(self, nodes):
        """
        Identifies clusters of nodes that involve quantisation and dequantisation 
        operations. It tags these nodes with a cluster name, which can be used
        later for partitioning and optimising the graph.

        Clustering is the process of grouping nodes in the computation graph that are related
        to quantisation and dequantisation operations. This is useful for optimising the graph
        for execution on specialized hardware.
        """
        def get_dequant_inputs(node):
            """
            This function returns all the dequant operators which produce inputs to the node.
            However, if the operator has 3 inputs and only one comes from dequant, the function
            will return true and consequently the code condition `if dequant_inputs:` will be true.

            This is done to handle the unexpected behavior of the NeutronQuantizer with the bias tensor (EIEX-66).
            """
            return [
                input_node for input_node in node.args
                if isinstance(input_node, torch.fx.node.Node) and self.is_dequant_node(input_node)
            ]

        def get_quant_outputs(node):
            """
            Retrieve the quantised outputs of a given node.

            This function examines the outputs of the provided node to identify
            quantised nodes. It also checks if the output operation is a call to the
            `operator.getitem` function and then inspects the operator's output to
            find quantised nodes.
            """
            quant_outputs = []
            for user in node.users:
                if user.op == "call_function" and user.target == operator.getitem:
                    for grandchild in user.users:
                        if self.is_quant_node(grandchild):
                            quant_outputs.append(grandchild)
                elif self.is_quant_node(user):
                    quant_outputs.append(user)
            return quant_outputs

        def tag_node_and_related(node, cluster_name, dequant_inputs, quant_outputs):
            # Tags a node and its related dequant and quant nodes with a specified cluster name
            logging.info(f"Tagging node {node} as {cluster_name}")
            node.meta["cluster"] = cluster_name
            for dequant_node in dequant_inputs:
                dequant_node.meta["cluster"] = cluster_name
            for quant_node in quant_outputs:
                quant_node.meta["cluster"] = cluster_name

        for node in nodes:
            if node.op == "call_function":
                dequant_inputs = get_dequant_inputs(node)
                quant_outputs = get_quant_outputs(node)
                if dequant_inputs and quant_outputs:
                    cluster_name = f"{node.name}_cluster"
                    tag_node_and_related(node, cluster_name, dequant_inputs, quant_outputs)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        # Run the CapabilityBasedPartitioner to return the largest possible
        # subgraphs containing the nodes with the tags
        logging.info("NeutronPartitioner::partition")
        partition_tags = {}

        graph_module = exported_program.graph_module
        nodes = list(graph_module.graph.nodes)

        self.tag_clusters(nodes)
        
        graph_module.recompile()

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            NeutronSupportedOperators(),
            allows_single_node_partition=True,
        )

        partition_list = capability_partitioner.propose_partitions()
        for partition in partition_list:
            for node in partition.nodes:
                delegation_tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = delegation_tag
                partition_tags[delegation_tag] = self.delegation_spec

        tag_constant_data(exported_program)
        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )
