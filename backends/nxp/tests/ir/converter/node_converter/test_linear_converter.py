import numpy as np
import pytest
import torch

from executorch.backends.nxp.tests.executorch_pipeline import to_edge_program
from executorch.backends.nxp.tests.executors import convert_run_compare
from executorch.backends.nxp.tests.models import LinearModule


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


def test_linear_conversion__with_bias():
    input_shape = (10, 32)
    edge_program = to_edge_program(LinearModule(bias=True), input_shape).exported_program()

    input_data = np.random.random(input_shape).astype(np.float32)

    convert_run_compare(edge_program, input_data=input_data, atol=1.e-6)

def test_linear_conversion__without_bias():
    input_shape = (10, 32)
    edge_program = to_edge_program(LinearModule(bias=True), input_shape).exported_program()

    input_data = np.random.random(input_shape).astype(np.float32)

    convert_run_compare(edge_program, input_data=input_data, atol=1.e-6)
