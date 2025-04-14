# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import numpy as np
import pytest
import torch
from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters.view_copy_converter import \
    ViewCopyConverter
from torch import nn
from torch.export import ExportedProgram

from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters import AddMMConverter, MMConverter
from executorch.backends.nxp.pytorch_passes.fuse_batch_norm_with_conv_pass import FuseBatchNormWithConvPass
from executorch.backends.nxp.pytorch_passes.fuse_batch_norm_with_linear_pass import FuseBatchNormWithLinearPass
from executorch.backends.nxp.pytorch_passes.nxp_pytorch_pass_manager import NXPPyTorchPassManager
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import OverrideTargetSupportCheck


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(42)
    np.random.seed(23)


class BatchNormModule(torch.nn.Module):
    def __init__(self, input_rank: int, num_features: int, eps: float = 1e-5):
        super().__init__()
        match input_rank - 2:
            case 1:
                self.batch_norm = nn.BatchNorm1d(num_features, eps)
            case 2:
                self.batch_norm = nn.BatchNorm2d(num_features, eps)
            case 3:
                self.batch_norm = nn.BatchNorm3d(num_features, eps)
            case _:
                raise ValueError
        self.eval()

    def forward(self, x):
        return self.batch_norm(x)


class ConvBatchNormModule(torch.nn.Module):
    def __init__(self, bias: bool, input_rank: int, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, bias=bias)
        self.batch_norm = BatchNormModule(input_rank, num_features, eps)
        self.eval()

    def forward(self, x):
        x = self.conv(x)
        return self.batch_norm(x)


class LinearBatchNormModule(torch.nn.Module):
    def __init__(self, bias: bool, input_rank: int, fc_in_features: int, fc_out_features: int, eps: float = 1e-5):
        super().__init__()
        self.linear = torch.nn.Linear(fc_in_features, fc_out_features, bias=bias)
        self.batch_norm = BatchNormModule(input_rank, fc_out_features, eps)
        self.eval()

    def forward(self, x):
        x = self.linear(x)
        return self.batch_norm(x)


@pytest.mark.parametrize('bias', [True, False], ids=lambda x: 'Bias' if x else 'No bias')
@pytest.mark.parametrize('input_shape', [[4, 6, 8], [2, 4, 6, 8]], ids=lambda x: f'{len(x)}D')
def test_batch_norm_conv_fusing(bias: bool, input_shape: list[int]):
    example_input = (torch.ones(*input_shape),)

    module = ConvBatchNormModule(bias, len(input_shape), 4)
    graph_module = torch._export.capture_pre_autograd_graph(module, example_input)

    # Fuse the BatchNorm with Conv.
    pass_manager = NXPPyTorchPassManager(deepcopy(graph_module), [FuseBatchNormWithConvPass])
    transformed_module = pass_manager.run()

    # Make sure the fusion worked.
    og_nodes = list(graph_module.graph.nodes)
    transformed_nodes = list(transformed_module.graph.nodes)

    assert len(og_nodes) == (14 if bias else 13)
    assert og_nodes[9 if bias else 8].target.__name__ == '_native_batch_norm_legit_no_training.default'

    assert len(transformed_nodes) == 5
    assert not any(node.op == 'call_function' and 'batch_norm' in node.target.__name__ for node in transformed_nodes)

    # Verify that the behavior has not changed.
    input_data = torch.randn(input_shape, dtype=torch.float32)
    out1 = graph_module(input_data).detach().numpy()
    out2 = transformed_module(input_data).detach().numpy()
    assert np.allclose(out1, out2, atol=3.e-7)


@pytest.mark.parametrize('bias', [True, False], ids=lambda x: 'Bias' if x else 'No bias')
def test_batch_norm_linear_fusing(bias: bool):
    input_shape = (2, 4, 6, 8)
    example_input = (torch.ones(*input_shape),)

    module = LinearBatchNormModule(bias, 4, input_shape[-1], input_shape[1])
    graph_module = torch._export.capture_pre_autograd_graph(module, example_input)

    # Fuse the BatchNorm with Linear.
    pass_manager = NXPPyTorchPassManager(deepcopy(graph_module), [FuseBatchNormWithLinearPass])
    transformed_module = pass_manager.run()

    # Make sure the fusion worked.
    og_nodes = list(graph_module.graph.nodes)
    transformed_nodes = list(transformed_module.graph.nodes)

    assert len(og_nodes) == (14 if bias else 13)
    assert og_nodes[3 if bias else 2].target.__name__ == 'linear.default'

    assert len(transformed_nodes) == 5
    assert not any(node.op == 'call_function' and 'batch_norm' in node.target.__name__ for node in transformed_nodes)

    # Verify that the behavior has not changed.
    input_data = torch.randn(input_shape, dtype=torch.float32)
    out1 = graph_module(input_data).detach().numpy()
    out2 = transformed_module(input_data).detach().numpy()
    assert np.allclose(out1, out2, atol=1.2e-7)


@pytest.mark.parametrize('bias', [True, False], ids=lambda x: 'Bias' if x else 'No bias')
def test_batch_norm_conv_fusing__full_pipeline__1d(bias: bool):
    input_shape = [4, 6, 8]
    module = ConvBatchNormModule(bias, len(input_shape), 4)

    edge_program = to_quantized_edge_program(module, tuple(input_shape)).exported_program()
    nodes = list(edge_program.graph.nodes)

    assert len(nodes) == 13  # 1D Conv currently isn't delegated, because it doesn't get quantized.
    assert not any(node.op == 'call_function' and 'batch_norm' in node.target.__name__ for node in nodes)


@pytest.mark.parametrize('bias', [True, False], ids=lambda x: 'Bias' if x else 'No bias')
def test_batch_norm_conv_fusing__full_pipeline__2d(bias: bool):
    input_shape = [1, 4, 6, 8]
    module = ConvBatchNormModule(bias, len(input_shape), 4)

    edge_program = to_quantized_edge_program(module, tuple(input_shape)).exported_program()
    nodes = list(edge_program.graph.nodes)

    assert len(nodes) == 7
    assert not any(node.op == 'call_function' and 'batch_norm' in node.target.__name__ for node in nodes)


@pytest.mark.parametrize('bias', [True, False], ids=lambda x: 'Bias' if x else 'No bias')
def test_batch_norm_linear_fusing__full_pipeline(bias: bool):
    input_shape = (2, 4, 6, 8)
    module = LinearBatchNormModule(bias, 4, input_shape[-1], input_shape[1])

    # Don't delegate the Linear node, because there seems to be a bug with the NeutronConverter/NeutronPartitioner.
    #  But that doesn't affect the validity of this test.
    with OverrideTargetSupportCheck(AddMMConverter, new_target_support_check=lambda *_: False):
        with OverrideTargetSupportCheck(MMConverter, new_target_support_check=lambda *_: False):
            with OverrideTargetSupportCheck(ViewCopyConverter, new_target_support_check=lambda *_: False):
                edge_program = to_quantized_edge_program(module, tuple(input_shape)).exported_program()
                nodes = list(edge_program.graph.nodes)

    assert len(nodes) == 18
    assert not any(node.op == 'call_function' and 'batch_norm' in node.target.__name__ for node in nodes)
