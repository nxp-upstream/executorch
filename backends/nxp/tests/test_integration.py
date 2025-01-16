import os.path
import pathlib

import pytest
import torch

from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_executorch_program
from executorch.backends.nxp.tests.models import ConvFCSoftmaxModule
from executorch.exir.backend.utils import get_delegation_info

_CURRENT_DIR = pathlib.Path(__file__).parent
_PROJECT_DIR = _CURRENT_DIR.parent.parent.parent

QUANTIZED_OPS_AOT_LIB_PATH = _PROJECT_DIR / "cmake-build-debug" / "kernels" / "quantized" / "libquantized_ops_aot_lib.so"


@pytest.mark.skipif(not os.path.exists(QUANTIZED_OPS_AOT_LIB_PATH), reason="Quant OPS AoT library file not found.")
def test_conv_fc_softmax__to_executorch_program():
    torch.ops.load_library(str(QUANTIZED_OPS_AOT_LIB_PATH))

    model = ConvFCSoftmaxModule()
    input_shape = (1, 4, 5, 5)

    exec_prog = to_quantized_executorch_program(model, input_shape)

    program = exec_prog.exported_program()
    assert program.graph_module.lowered_module_0, "There is no lowered module with Neutron microcode."

    delegation_info = get_delegation_info(program.graph_module)
    assert delegation_info.num_delegated_subgraphs == 1
    assert delegation_info.num_non_delegated_nodes == 11
    assert delegation_info.num_delegated_nodes == 13

    for node in program.graph.nodes:
        # Make sure Convolution and AddMM are delegated
        assert "convolution" not in node.name
        assert "addmm" not in node.name
