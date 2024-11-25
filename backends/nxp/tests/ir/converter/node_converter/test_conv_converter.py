import torch

from executorch import exir
from executorch.backends.nxp.tests.executors import convert_run_compare, ToNCHWPreprocess, ToNHWCPreprocess
from executorch.backends.nxp.tests.models import Conv2dModule


def test_conv2d_conversion():
    model = Conv2dModule()

    example_input = (torch.ones(1, 4, 32, 32),)
    exir_program = torch.export.export(model, example_input)
    edge_program = exir.to_edge(exir_program).exported_program()

    torch.manual_seed(23)
    input_data = torch.randn((1, 4, 32, 32), dtype=torch.float32).detach().numpy()

    convert_run_compare(edge_program, input_data, tflite_input_preprocess=ToNHWCPreprocess(),
                        tflite_output_preprocess=ToNCHWPreprocess(), atol=5e-7)
