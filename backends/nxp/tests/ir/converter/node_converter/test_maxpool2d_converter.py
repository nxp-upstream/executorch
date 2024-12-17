import pytest
import torch

from executorch import exir
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import convert_run_compare, ToNCHWPreprocess, ToNHWCPreprocess
from executorch.backends.nxp.tests.models import Maxpool2dModule
from executorch.backends.xnnpack.passes import RemoveGetItemPass, XNNPACKPassManager
from executorch.exir.verification.verifier import EXIREdgeDialectVerifier
import pytest

@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.seed()

def test_maxpool2d_conversion():
    model = Maxpool2dModule()

    example_input = (torch.ones(1, 4, 32, 32),)
    exir_program = torch.export.export(model, example_input)
    edge_program = exir.to_edge(exir_program).exported_program()

    # We need to create custom model verifier with max_pool2d added as exception.
    # Otherwise, we get violation that this op is not part of ATen Core ops.
    edge_program._verifier = EXIREdgeDialectVerifier(
        class_only=True,
        exception_list=[torch.ops.aten.max_pool2d.default]
    )

    # Remove MaxPool-related "getitem" nodes from graph
    edge_program = XNNPACKPassManager(edge_program, [RemoveGetItemPass]).transform()

    torch.manual_seed(23)
    input_data = torch.randn((1, 4, 32, 32), dtype=torch.float32).detach().numpy()

    convert_run_compare(edge_program, input_data, tflite_input_preprocess=ToNHWCPreprocess(),
                        tflite_output_preprocess=ToNCHWPreprocess())

def test_maxpool2d_lowered_program_and_tflite_output_match():
    model = Maxpool2dModule()
    input_shape = (1, 4, 32, 32)

    edge_program_manager = to_quantized_edge_program(model, input_shape)
    exported_program = edge_program_manager.exported_program()

    # We need to create custom model verifier with max_pool2d added as exception.
    # Otherwise, we get violation that this op is not part of ATen Core ops.
    exported_program._verifier = EXIREdgeDialectVerifier(
        class_only=True,
        exception_list=[torch.ops.aten.max_pool2d.default]
    )

    # Remove MaxPool-related "getitem" nodes from graph
    exported_program = XNNPACKPassManager(exported_program, [RemoveGetItemPass]).transform()

    torch.manual_seed(23)
    input_data = torch.randn((1, 4, 32, 32), dtype=torch.float32).detach().numpy()

    # Verify outputs of program and TFLite model
    convert_run_compare(exported_program, input_data, tflite_input_preprocess=ToNHWCPreprocess(),
                        tflite_output_preprocess=ToNCHWPreprocess(), atol=5e-7)


'''
This test is expected to fail with Error: [Code.NOT_IMPLEMENTED] - Converter for
    'aten.max_pool2d_with_indices.default' not implemented!
'''
def test_maxpool2d_no_quantize():
    model = Maxpool2dModule()

    example_input = (torch.ones(1, 4, 32, 32),)
    exir_program = torch.export.export(model, example_input)
    edge_program = exir.to_edge(exir_program).exported_program()

    torch.manual_seed(23)
    input_data = torch.randn((1, 4, 32, 32), dtype=torch.float32).detach().numpy()

    with pytest.raises(Exception) as e:
        convert_run_compare(edge_program, input_data, tflite_input_preprocess=ToNHWCPreprocess(),
                        tflite_output_preprocess=ToNCHWPreprocess(), atol=5e-7)
    
    assert e.value.error_code.name == "NOT_IMPLEMENTED"
    assert e.value.msg == "Converter for 'aten.max_pool2d_with_indices.default' not implemented!"
