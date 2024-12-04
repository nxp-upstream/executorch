import pytest
import torch

from executorch.backends.nxp.tests.executors import convert_run_compare, ToNCHWPreprocess, ToNHWCPreprocess
from executorch.backends.nxp.tests.executorch_pipeline import to_edge_program
from executorch.backends.nxp.tests.models import ConstantPadNDModule, ConstantPadNDConvModule


@pytest.mark.parametrize("constant", [0.0, 42., -13.37])
def test_constant_pad_nd_conversion__specific_constant(constant):
    input_shape = [2, 4, 6, 8]
    paddings = [1, 2, 3, 4]

    edge_program = to_edge_program(ConstantPadNDModule(paddings, constant), input_shape).exported_program()

    torch.manual_seed(23)
    input_data = torch.randn(input_shape, dtype=torch.float32).detach().numpy()

    convert_run_compare(edge_program, input_data)


def test_constant_pad_nd_conversion__default_constant():
    input_shape = [2, 4, 6, 8]
    paddings = [1, 2, 3, 4]

    edge_program = to_edge_program(ConstantPadNDModule(paddings), input_shape).exported_program()

    torch.manual_seed(23)
    input_data = torch.randn(input_shape, dtype=torch.float32).detach().numpy()

    convert_run_compare(edge_program, input_data)


@pytest.mark.parametrize("input_shape, paddings", [
    pytest.param([2], list(range(2)), id='1D, padding H'),

    pytest.param([2, 4], list(range(2)), id='2D, padding H'),
    pytest.param([2, 4], list(range(4)), id='2D, padding N, H'),

    pytest.param([2, 4, 6], list(range(2)), id='3D, padding H'),
    pytest.param([2, 4, 6], list(range(4)), id='3D, padding C, H'),
    pytest.param([2, 4, 6], list(range(6)), id='3D, padding N, C, H'),

    pytest.param([2, 4, 6, 8], list(range(2)), id='4D, padding W'),
    pytest.param([2, 4, 6, 8], list(range(4)), id='4D, padding H, W'),
    pytest.param([2, 4, 6, 8], list(range(6)), id='4D, padding C, H, W'),
    pytest.param([2, 4, 6, 8], list(range(8)), id='4D, padding N, C, H, W'),

    pytest.param([1, 2, 3, 4, 5], list(range(2)), id='5D, padding D'),
    pytest.param([1, 2, 3, 4, 5], list(range(4)), id='5D, padding W, D'),
    pytest.param([1, 2, 3, 4, 5], list(range(6)), id='5D, padding H, W, D'),
    pytest.param([1, 2, 3, 4, 5], list(range(8)), id='5D, padding C, H, W, D'),
    pytest.param([1, 2, 3, 4, 5], list(range(10)), id='5D, padding N, C, H, W, D'),
])
def test_constant_pad_nd_conversion__format_less(input_shape, paddings):
    edge_program = to_edge_program(ConstantPadNDModule(paddings), input_shape).exported_program()

    torch.manual_seed(23)
    input_data = torch.randn(input_shape, dtype=torch.float32).detach().numpy()

    convert_run_compare(edge_program, input_data)


@pytest.mark.parametrize("input_shape, paddings", [
    pytest.param([2, 4, 6, 8], list(range(2)), id='4D, padding W'),
    pytest.param([2, 4, 6, 8], list(range(4)), id='4D, padding H, W'),
    pytest.param([2, 1, 6, 8], [1, 2, 3, 4, 2, 1], id='4D, padding C, H, W'),
    pytest.param([2, 1, 6, 8], [1, 2, 3, 4, 2, 1, 5, 6], id='4D, padding N, C, H, W'),
])
def test_constant_pad_nd_conversion__channels_first(input_shape, paddings):
    edge_program = to_edge_program(ConstantPadNDConvModule(paddings), input_shape).exported_program()  # Extra `Conv` after the padding.

    torch.manual_seed(23)
    input_data = torch.randn(input_shape, dtype=torch.float32).detach().numpy()

    convert_run_compare(edge_program, input_data, tflite_input_preprocess=ToNHWCPreprocess(),
                        tflite_output_preprocess=ToNCHWPreprocess())
