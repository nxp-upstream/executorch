import numpy as np
import torch
from torch.export import ExportedProgram

from executorch.backends.nxp.backend.edge_program_converter import EdgeProgramToIRConverter
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions
from executorch.backends.nxp.backend.ir.lib.tflite.Model import Model
from executorch.backends.nxp.neutron_partitioner import NeutronPartitioner
from executorch.backends.nxp.nxp_backend import generate_neutron_compile_spec
from executorch.backends.nxp.tests.executors import TFLiteExecutor, EdgeProgramExecutor, convert_run_compare, \
    ToNHWCPreprocess
from executorch.backends.nxp.tests.models import ConvFCSoftmaxModule
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


def to_lowered_edge_program_manager(model: torch.nn.Module, input_shape: tuple):
    calibration_inputs = [(torch.randn(input_shape),), (torch.randn(input_shape),)]
    example_input = (torch.ones(*input_shape),)

    exir_program_aten = torch._export.capture_pre_autograd_graph(model, example_input)
    exir_program_aten_quant = post_training_quantize(exir_program_aten, calibration_inputs)
    edge_program_manager = export_to_edge(exir_program_aten_quant, example_input)

    partitioner = NeutronPartitioner(generate_neutron_compile_spec("rt700"))

    edge_program_manager = edge_program_manager.to_backend(partitioner)
    return edge_program_manager


def test_conv2d_no_bias__qdq_clustering():
    model = Conv2dNoBiasModule()
    input_shape = (1, 4, 32, 32)

    # Run conversion
    edge_program_manager = to_lowered_edge_program_manager(model, input_shape, )
    # Get subgraph (module) that is delegated to neutron
    lowered_module = edge_program_manager.exported_program().graph_module.lowered_module_0
    nodes = list(lowered_module.original_module.graph.nodes)

    assert len(nodes) == 7

    q_x_node = nodes[1]
    dq_w_node = nodes[2]
    dq_x_node = nodes[3]
    conv_node = nodes[4]
    dq_y_node = nodes[5]
    q_y_node = nodes[6]

    assert "cluster" not in q_x_node.meta
    assert dq_w_node.meta["cluster"] == "aten_convolution_default_cluster"
    assert dq_x_node.meta["cluster"] == "aten_convolution_default_cluster"
    assert conv_node.meta["cluster"] == "aten_convolution_default_cluster"
    assert dq_y_node.meta["cluster"] == "aten_convolution_default_cluster"
    assert "cluster" not in q_y_node.meta


def test_conv2d__lowered_program_and_tflite_output_match(mocker):
    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    model = Conv2dNoBiasModule()
    input_shape = (1, 4, 32, 32)

    # Run conversion
    _ = to_lowered_edge_program_manager(model, input_shape)

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    tflite_model = Model.GetRootAs(tflite_flatbuffers_model)
    sub_graph = tflite_model.Subgraphs(0)

    assert sub_graph.OperatorsLength() == 1
    assert sub_graph.Operators(0).BuiltinOptionsType() == BuiltinOptions.Conv2DOptions

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (torch.randn(input_shape, dtype=torch.float32) * 50).type(torch.int8).detach().numpy()
    input_data_tflite = np.transpose(input_data, [0, 2, 3, 1])

    # Execute program and TFLite model
    program_executor = EdgeProgramExecutor(exported_program)
    tflite_executor = TFLiteExecutor(model_content=tflite_flatbuffers_model)

    output_edge = program_executor.inference(input_data)
    output_tflite = tflite_executor.inference(input_data_tflite)

    output_tflite = np.transpose(output_tflite, [0, 3, 1, 2])

    # Outputs difference is smaller than 1 (rounding error in quantization)
    assert np.max(np.abs(output_edge - output_tflite)) <= 1


def test_conv_fc_softmax__lowered_program_and_tflite_output_match(mocker):
    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    model = ConvFCSoftmaxModule()
    input_shape = (1, 4, 5, 5)

    # Run conversion
    _ = to_lowered_edge_program_manager(model, input_shape)

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    # Capture generated model
    tflite_flatbuffers_model, _ = converter_spy.spy_return

    # No Transpose ops in produced TFLite model
    tflite_subgraph = Model.GetRootAs(tflite_flatbuffers_model).Subgraphs(0)

    assert tflite_subgraph.OperatorsLength() == 4
    assert tflite_subgraph.Operators(0).BuiltinOptionsType() == BuiltinOptions.Conv2DOptions
    assert tflite_subgraph.Operators(1).BuiltinOptionsType() == BuiltinOptions.ReshapeOptions
    assert tflite_subgraph.Operators(2).BuiltinOptionsType() == BuiltinOptions.FullyConnectedOptions
    assert tflite_subgraph.Operators(3).BuiltinOptionsType() == BuiltinOptions.SoftmaxOptions

    # Verify outputs of program and TFLite model
    input_data = (torch.randn(input_shape, dtype=torch.float32)).type(torch.int8).detach().numpy()
    convert_run_compare(exported_program, input_data=input_data, tflite_input_preprocess=ToNHWCPreprocess())
