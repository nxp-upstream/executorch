# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Partitioner for the NXP Neutron NPU

import logging
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

    # QDQ ops
    exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
]

class NeutronSupportedOperators(OperatorSupportBase):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        # check if the PyTorch op get called is supported for Neutron
        return node.op == "call_function" and node.target in NeutronSupportedOperatorsList

@final
class NeutronPartitioner(Partitioner):
    def __init__(self, compile_spec: List[CompileSpec]) -> None:
        self.delegation_spec = DelegationSpec(NeutronBackend.__name__, compile_spec)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        # Run the CapabilityBasedPartitioner to return the largest possible
        # subgraphs containing the nodes with the tags
        logging.info("NeutronPartitioner::partition")
        partition_tags = {}

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
