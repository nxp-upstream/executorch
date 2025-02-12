import torch
from executorch.backends.nxp.backend.edge_program_converter import EdgeProgramToIRConverter
from executorch.backends.nxp.backend.ir.converter.node_converter import Target
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.models import Conv2dModule

def test_operator_selector_mechanism():
    model = Conv2dModule(bias=False)
    input_shape = (1, 4, 32, 32)

    operators_not_to_delegate = ["aten::convolution"]

    edge_program_manager = to_quantized_edge_program(model, input_shape, target=Target.IGNORE, operators_not_to_delegate=operators_not_to_delegate)

    exported_program = edge_program_manager.exported_program()

    for node in exported_program.graph.nodes:
        if node.name == "aten_convolution_default":
            assert "delegation_tag" not in node.meta
