# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch

from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops, convert_run_compare
from executorch.exir.dialects._ops import ops as exir_ops


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(42)
    np.random.seed(23)


ExecutorchDelegateCall = torch.ops.higher_order.executorch_call_delegate
Pad = exir_ops.edge.aten.pad.default


class ReflectionPadModule(torch.nn.Module):

    def __init__(self, padding: int | tuple[int, ...]):
        super().__init__()
        self.reflection_pad = torch.nn.ReflectionPad2d(padding)

    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        return self.reflection_pad(x)


@pytest.mark.parametrize(
    "padding",
    [
        (1, 1, 1, 1)
    ],
)
def test_convert_reflection_pad(mocker, padding):
    input_shape = (1, 8, 5, 6)
    model = ReflectionPadModule(padding)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
    delegated_ep = to_quantized_edge_program(
        model,
        input_shape,
        use_neutron_for_format_conversion=False
    ).exported_program()

    # Make sure the `pad` was delegated.
    assert graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
    assert not graph_contains_any_of_ops(delegated_ep.graph, [Pad])

    # Verify correct behavior of the converted NeutronIR model.
    edge_partition = converter_spy.call_args.args[1]
    neutron_ir_partition, _ = converter_spy.spy_return

    input_data = (
        np.random.random(input_shape).astype(np.float32) * 256.0 - 128.0
    ).astype(np.int8)

    # Make sure the tested program contains the `reflection_pad`.
    assert graph_contains_any_of_ops(edge_partition.graph, [Pad])
    assert all(n.args[2] == "reflect" for n in edge_partition.graph.nodes if n.target == Pad)

    convert_run_compare(
        edge_partition,
        tfl_model=neutron_ir_partition,
        input_data=input_data,
    )

#
# def test_convert_reflection_pad__channels_last(mocker):
#     model = ReflectionPadConvModule()
#     input_shape = (1, 3, 4, 5)
#
#     converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
#     delegated_ep = to_quantized_edge_program(
#         model, input_shape, use_neutron_for_format_conversion=False
#     ).exported_program()
#
#     # Make sure the `reflection_pad` was delegated.
#     assert graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
#     assert not graph_contains_any_of_ops(delegated_ep.graph, [ReflectionPad])
#
#     # Verify correct behavior of the converted NeutronIR model.
#     intermediate_ep = converter_spy.call_args.args[1]
#     neutron_ir_model, _ = converter_spy.spy_return
#
#     input_data = (
#         np.random.random(input_shape).astype(np.float32) * 256.0 - 128.0
#     ).astype(np.int8)
#
#     # Make sure the tested program contains the `reflection_pad`.
#     assert graph_contains_any_of_ops(intermediate_ep.graph, [ReflectionPad])
#
#     convert_run_compare(
#         intermediate_ep,
#         tfl_model=neutron_ir_model,
#         input_data=input_data,
#         tflite_input_preprocess=ToChannelLastPreprocess(),
#         tflite_output_preprocess=ToChannelFirstPreprocess(),
#     )
