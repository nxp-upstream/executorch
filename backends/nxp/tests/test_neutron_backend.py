# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.nxp.neutron_partitioner import NeutronPartitioner
from executorch.backends.nxp.nxp_backend import generate_neutron_compile_spec
from executorch.backends.nxp.tests.models import Conv2dModule
from executorch.examples.nxp.aot_neutron_compile import post_training_quantize
from executorch.examples.portable import export_to_edge


def to_lowered_edge_program_manager(model: torch.nn.Module, input_shape: tuple):
    calibration_inputs = [(torch.randn(input_shape),), (torch.randn(input_shape),)]
    example_input = (torch.ones(*input_shape),)

    exir_program_aten = torch._export.capture_pre_autograd_graph(model, example_input)
    exir_program_aten_quant = post_training_quantize(exir_program_aten, calibration_inputs)
    edge_program_manager = export_to_edge(exir_program_aten_quant, example_input)

    partitioner = NeutronPartitioner(generate_neutron_compile_spec("rt700"))

    edge_program_manager = edge_program_manager.to_backend(partitioner)
    return edge_program_manager


def test_neutron_backend__single_conv_model():
    edge_program_manager = to_lowered_edge_program_manager(Conv2dModule(bias=False), (1, 4, 32, 32))
    lowered_module = edge_program_manager.exported_program().graph_module.lowered_module_0
    assert len(lowered_module.processed_bytes) != 0  # The Neutron microcode, weights and kernels have been written here
