import numpy as np
from torch.fx import Node, Graph

from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters import ViewCopyConverter
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import EdgeProgramExecutor, OverrideSupportedTargets
from executorch.backends.nxp.tests.models import ConvFCFCSoftmaxModuleWithoutReshape
from executorch.exir.dialects._ops import ops as exir_ops


def _is_view_copy(node_: Node) -> bool:
    return node_.op == 'call_function' and node_.target == exir_ops.edge.aten.view_copy.default


def _is_dequantize(node_: Node) -> bool:
    return node_.op == 'call_function' and node_.target.__name__ == 'quantized_decomposed.dequantize_per_tensor.default'


def _is_quantize(node_: Node) -> bool:
    return node_.op == 'call_function' and node_.target.__name__ == 'quantized_decomposed.quantize_per_tensor.default'


def _assert_nodes_form_a_view_copy_qdq_cluster(graph: Graph, node_indices: list[int]):
    assert len(node_indices) == 3

    nodes = list(graph.nodes)
    assert _is_dequantize(dequantize := nodes[node_indices[0]])
    assert _is_view_copy(view_copy := nodes[node_indices[1]])
    assert _is_quantize(quantize := nodes[node_indices[2]])

    # Make sure the nodes are properly connected.
    assert view_copy.args[0] == dequantize
    assert quantize.args[0] == view_copy


def test_moving_view_copy_into_separate_qdq_clusters():
    model = ConvFCFCSoftmaxModuleWithoutReshape()
    input_shape = (1, 4, 3, 33)

    # Prohibit `view_copy` conversion for the testing purposes.
    with OverrideSupportedTargets(ViewCopyConverter, new_targets=[]):
        epm = to_quantized_edge_program(model, input_shape, target='imxrt700')
        exported_program = epm.exported_program()

        nodes = list(exported_program.graph_module.graph.nodes)
        assert len(nodes) == 28
        _assert_nodes_form_a_view_copy_qdq_cluster(exported_program.graph, [7, 8, 9])
        _assert_nodes_form_a_view_copy_qdq_cluster(exported_program.graph, [12, 13, 14])
        _assert_nodes_form_a_view_copy_qdq_cluster(exported_program.graph, [15, 16, 17])
        _assert_nodes_form_a_view_copy_qdq_cluster(exported_program.graph, [20, 21, 22])

        # Make sure the program is runnable.
        input_data = np.random.random(input_shape).astype('float32')
        program_executor = EdgeProgramExecutor(exported_program)
        program_executor.inference(input_data)
