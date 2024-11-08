import torch

from executorch.backends.nxp.neutron_partitioner import NeutronPartitioner
from executorch.backends.nxp.nxp_backend import generate_neutron_compile_spec
from executorch.examples.nxp.aot_neutron_compile import post_training_quantize
from executorch.examples.portable import export_to_edge


class Conv2dNoBiasModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = torch.nn.Conv2d(
            in_channels=4, out_channels=8, kernel_size=3, bias=False, stride=2, dilation=1
        )

    def forward(self, x):
        return self.conv(x)


def test_conv2d_partitioner():
    calibration_inputs = [(torch.randn((1, 4, 32, 32)),), (torch.randn((1, 4, 32, 32)),)]
    model = Conv2dNoBiasModule()
    example_input = (torch.ones(1, 4, 32, 32),)

    exir_program_aten = torch._export.capture_pre_autograd_graph(model, example_input)
    exir_program_aten_quant = post_training_quantize(exir_program_aten, calibration_inputs)
    edge_program_manager = export_to_edge(exir_program_aten_quant, example_input)

    partitioner = NeutronPartitioner(generate_neutron_compile_spec("rt700"))
    edge_program = edge_program_manager.to_backend(partitioner)

    # Get subgraph (module) that is delegated to neutron
    lowered_module = edge_program.exported_program().graph_module.lowered_module_0
    nodes = list(lowered_module.original_module.graph.nodes)

    assert len(nodes) == 7

    q_x_node = nodes[1]
    dq_w_node = nodes[2]
    dq_x_node = nodes[3]
    conv_node = nodes[4]
    q_y_node = nodes[5]

    assert "cluster" not in q_x_node.meta
    assert dq_w_node.meta["cluster"] == "aten_convolution_default_cluster"
    assert dq_x_node.meta["cluster"] == "aten_convolution_default_cluster"
    assert conv_node.meta["cluster"] == "aten_convolution_default_cluster"
    assert q_y_node.meta["cluster"] == "aten_convolution_default_cluster"
    