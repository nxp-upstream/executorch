import pytest
import torch

from executorch import exir
from executorch.backends.nxp.tests.executors import convert_run_compare, ToNCHWPreprocess, ToNHWCPreprocess
from executorch.backends.nxp.tests.models import SoftmaxModule, SoftmaxConvModule
from executorch.backends.nxp.backend.edge_program_converter import EdgeProgramToIRConverter
from executorch.backends.nxp.backend.ir.conversion_config import ConversionConfig


@pytest.mark.parametrize("input_shape,dim", [
    pytest.param((10,), -1, id="1D,dim=-1"),
    pytest.param((10,), 0, id="1D,dim=0"),
    pytest.param((10, 32), -1, id="2D,dim=-1"),
    pytest.param((10, 32), 1, id="2D,dim=1"),
    pytest.param((10, 32, 32), -1, id="3D,dim=-1"),
    pytest.param((10, 32, 32), 2, id="3D,dim=2"),
    pytest.param((10, 32, 32, 8), -1, id="4D,dim=-1"),
    pytest.param((10, 32, 32, 8), 3, id="4D,dim=3"),
    pytest.param((10, 32, 32, 8, 8), -1, id="5D,dim=-1"),
    pytest.param((10, 32, 32, 8, 8), 4, id="5D,dim=4"),
])
def test_softmax_conversion(input_shape, dim):
    model = SoftmaxModule(dim)

    example_input = (torch.ones(input_shape),)
    exir_program = torch.export.export(model, example_input)
    edge_program_manager = exir.to_edge(exir_program)

    torch.manual_seed(23)
    input_data = torch.randn(input_shape, dtype=torch.float32).detach().numpy()

    convert_run_compare(edge_program_manager.exported_program(), input_data=input_data, atol=5e-7)

@pytest.mark.parametrize("input_shape,dim", [
    pytest.param((10, 4, 32, 32), 1, id="4D,dim=1"),
    pytest.param((10, 4, 16, 16), -3, id="4D,dim=-3"),
])
def test_softmax_conversion_channel_last(input_shape, dim):
    model = SoftmaxConvModule(dim)

    example_input = (torch.ones(input_shape),)
    exir_program = torch.export.export(model, example_input)
    edge_program_manager = exir.to_edge(exir_program)

    torch.manual_seed(23)
    input_data = torch.randn(input_shape, dtype=torch.float32).detach().numpy()

    convert_run_compare(edge_program_manager.exported_program(), tflite_input_preprocess=ToNHWCPreprocess(),
                        tflite_output_preprocess=ToNCHWPreprocess(), input_data=input_data, atol=5e-7)


@pytest.mark.parametrize("input_shape,dim", [
    pytest.param((10, 32), 0, id="2D,dim=0"),
    pytest.param((10, 32, 32), 1, id="3D,dim=1"),
    pytest.param((10, 32, 32, 8), 2, id="4D,dim=2"),
    pytest.param((10, 32, 32, 8, 8), 3, id="5D,dim=3"),
    pytest.param((10, 32, 32, 8, 8), 2, id="5D,dim=2"),
])
def test_softmax_conversion_unsupported_dims(input_shape, dim):
    model = SoftmaxModule(dim)

    example_input = (torch.ones(input_shape),)
    exir_program = torch.export.export(model, example_input)
    edge_program_manager = exir.to_edge(exir_program)

    with pytest.raises(AssertionError, match="'dim' different to last dim not yet supported"):
        EdgeProgramToIRConverter().convert_program(edge_program_manager.exported_program(), ConversionConfig())
