import glob
import pathlib

import torch

from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_executorch_program
from executorch.backends.nxp.tests.models import ConvFCSoftmaxModule
from executorch.devtools.backend_debug import get_delegation_info

_CURRENT_DIR = pathlib.Path(__file__).parent
_PROJECT_DIR = _CURRENT_DIR.parent.parent.parent


def _get_quantized_ops_aot_lib_path(
        default_path_pattern="pip-out/temp*/cmake-out/kernels/quantized/libquantized_ops_aot_lib.so"
):
    lib_path = glob.glob(f"{_PROJECT_DIR}/{default_path_pattern}")

    if len(lib_path) < 1:
        raise RuntimeError("Unable to find 'libquantized_ops_aot_lib'. Make sure you've built project "
                           "with './install_requirements.sh' or provided correct path.")
    return lib_path[0]


def test_conv_fc_softmax__to_executorch_program():
    torch.ops.load_library(str(_get_quantized_ops_aot_lib_path()))

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
