import glob
import pathlib

import torch

from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.exported_program_vizualize import exported_program_to_dot
from executorch.backends.nxp.tests.models import Conv2dReLUModule
from executorch.examples.nxp.cifar_net.cifar_net import CifarNet
from executorch.examples.nxp.models.mlperf_tiny import KeywordSpotting, VisualWakeWords
from executorch.exir import ExecutorchBackendConfig
from executorch.exir.passes.quantize_io_pass import get_config_method_name

_CURRENT_DIR = pathlib.Path(__file__).parent
_PROJECT_DIR = _CURRENT_DIR.parent.parent.parent.parent.parent


def _get_quantized_ops_aot_lib_path(
        default_path_pattern="pip-out/temp*/cmake-out/kernels/quantized/libquantized_ops_aot_lib.so"
):
    lib_path = glob.glob(f"{_PROJECT_DIR}/{default_path_pattern}")

    if len(lib_path) < 1:
        raise RuntimeError("Unable to find 'libquantized_ops_aot_lib'. Make sure you've built project "
                           "with './install_requirements.sh' or provided correct path.")
    return lib_path[0]


def test_remove_io_quant_ops_pass__conv_relu():
    torch.ops.load_library(str(_get_quantized_ops_aot_lib_path()))

    model = Conv2dReLUModule()
    model.eval()

    input_shape = (1, 4, 32, 32)
    edge_program_manager = to_quantized_edge_program(model, input_shape, remove_quant_io_ops=True)

    exec_prog = edge_program_manager.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )

    exported_program_to_dot(exec_prog.exported_program(), "conv_relu.dot")

    nodes = list(exec_prog.exported_program().graph.nodes)
    assert nodes[0].meta["val"].dtype == torch.int8, "Input tensor doesn't have type INT8."
    assert nodes[2].name == 'executorch_call_delegate'
    assert nodes[4].meta["val"][0].dtype == torch.int8, "Output tensor doesn't have type INT8."

    assert get_config_method_name(None, "input", 0, "scale") in exec_prog._config_methods
    assert get_config_method_name(None, "input", 0, "zp") in exec_prog._config_methods
    assert get_config_method_name(None, "output", 0, "scale") in exec_prog._config_methods
    assert get_config_method_name(None, "output", 0, "zp") in exec_prog._config_methods


def test_remove_io_quant_ops_pass__cifarnet():
    torch.ops.load_library(str(_get_quantized_ops_aot_lib_path()))

    model = CifarNet().get_eager_model()
    input_shape = (1, 3, 32, 32)
    edge_program_manager = to_quantized_edge_program(model, input_shape, remove_quant_io_ops=True)

    exec_prog = edge_program_manager.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )

    nodes = list(exec_prog.exported_program().graph.nodes)
    assert len(nodes) == 17
    assert nodes[0].meta["val"].dtype == torch.int8, "Input tensor doesn't have type INT8."
    assert nodes[16].meta["val"][0].dtype == torch.int8, "Output tensor doesn't have type INT8."

    assert get_config_method_name(None, "input", 0, "scale") in exec_prog._config_methods
    assert get_config_method_name(None, "input", 0, "zp") in exec_prog._config_methods
    assert get_config_method_name(None, "output", 0, "scale") in exec_prog._config_methods
    assert get_config_method_name(None, "output", 0, "zp") in exec_prog._config_methods


def test_remove_io_quant_ops_pass__kws():
    torch.ops.load_library(str(_get_quantized_ops_aot_lib_path()))

    kws = KeywordSpotting()
    model = kws.get_eager_model()
    edge_program_manager = to_quantized_edge_program(model, kws._input_shape, remove_quant_io_ops=True)

    exec_prog = edge_program_manager.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )

    nodes = list(exec_prog.exported_program().graph.nodes)
    assert len(nodes) == 26
    assert nodes[0].meta["val"].dtype == torch.int8, "Input tensor doesn't have type INT8."
    assert nodes[25].meta["val"][0].dtype == torch.int8, "Output tensor doesn't have type INT8."

    assert get_config_method_name(None, "input", 0, "scale") in exec_prog._config_methods
    assert get_config_method_name(None, "input", 0, "zp") in exec_prog._config_methods
    assert get_config_method_name(None, "output", 0, "scale") in exec_prog._config_methods
    assert get_config_method_name(None, "output", 0, "zp") in exec_prog._config_methods


def test_remove_io_quant_ops_pass__vww():
    torch.ops.load_library(str(_get_quantized_ops_aot_lib_path()))

    vww = VisualWakeWords()
    model = vww.get_eager_model()
    edge_program_manager = to_quantized_edge_program(model, vww._input_shape, remove_quant_io_ops=True)

    exec_prog = edge_program_manager.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )

    nodes = list(exec_prog.exported_program().graph.nodes)
    assert len(nodes) == 5
    assert nodes[0].meta["val"].dtype == torch.int8, "Input tensor doesn't have type INT8."
    assert nodes[4].meta["val"][0].dtype == torch.int8, "Output tensor doesn't have type INT8."

    assert get_config_method_name(None, "input", 0, "scale") in exec_prog._config_methods
    assert get_config_method_name(None, "input", 0, "zp") in exec_prog._config_methods
    assert get_config_method_name(None, "output", 0, "scale") in exec_prog._config_methods
    assert get_config_method_name(None, "output", 0, "zp") in exec_prog._config_methods
