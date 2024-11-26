import pytest
import torch

from executorch import exir
from executorch.backends.nxp.backend.edge_program_converter import EdgeProgramToIRConverter
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import convert_run_compare, ToNCHWPreprocess, ToNHWCPreprocess
from executorch.backends.nxp.tests.models import Conv2dModule
from torch.export import ExportedProgram


def test_conv2d_conversion():
    model = Conv2dModule()

    example_input = (torch.ones(1, 4, 32, 32),)
    exir_program = torch.export.export(model, example_input)
    edge_program = exir.to_edge(exir_program).exported_program()

    torch.manual_seed(23)
    input_data = torch.randn((1, 4, 32, 32), dtype=torch.float32).detach().numpy()

    convert_run_compare(edge_program, input_data, tflite_input_preprocess=ToNHWCPreprocess(),
                        tflite_output_preprocess=ToNCHWPreprocess(), atol=5e-7)


@pytest.mark.parametrize("model, input_shape", [
    pytest.param(Conv2dModule(in_channels=8, out_channels=32, kernel_size=5,
                              bias=True, stride=1, dilation=1), (1, 8, 32, 32), id="conv layer 1"),
    pytest.param(Conv2dModule(in_channels=32, out_channels=32, kernel_size=5,
                              bias=True, stride=1, dilation=1), (1, 32, 32, 32), id="conv layer 2"),
    pytest.param(Conv2dModule(in_channels=32, out_channels=64, kernel_size=5,
                              bias=True, stride=1, dilation=1), (1, 32, 32, 32), id="conv layer 3"),
])
def test_conv2d_quant_conversion(mocker, model, input_shape):
    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    _ = to_quantized_edge_program(model, input_shape)

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    torch.manual_seed(23)
    input_data = (torch.randn(input_shape, dtype=torch.float32) * 50).type(torch.int8).detach().numpy()

    convert_run_compare(exported_program, tflite_input_preprocess=ToNHWCPreprocess(), tfl_model=tflite_flatbuffers_model,
                        tflite_output_preprocess=ToNCHWPreprocess(), input_data=input_data, atol=1.)
