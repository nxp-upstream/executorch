import pytest
import torch

from executorch import exir
from executorch.backends.nxp.tests.executors import convert_run_compare, ToNCHWPreprocess, ToNHWCPreprocess
from executorch.backends.nxp.tests.models import Maxpool2dModule
from executorch.backends.xnnpack.passes import RemoveGetItemPass, XNNPACKPassManager
from executorch.exir.verification.verifier import EXIREdgeDialectVerifier

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
