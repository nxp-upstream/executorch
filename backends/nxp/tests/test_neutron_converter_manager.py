import pytest
import torch

from executorch import exir
from executorch.backends.nxp.backend.edge_program_converter import EdgeProgramToIRConverter
from executorch.backends.nxp.backend.neutron_converter_manager import NeutronConverterManager
from executorch.backends.nxp.tests.models import Conv2dModule


def test_conv2d_neutron_conversion():
    pytest.importorskip("neutron_converter_wrapper")

    model = Conv2dModule()

    example_input = (torch.ones(1, 4, 32, 32),)
    exir_program = torch.export.export(model, example_input)
    edge_program_manager = exir.to_edge(exir_program)

    edge_program_converter = EdgeProgramToIRConverter()
    tflite_model, _ = edge_program_converter.convert_program(edge_program_manager.exported_program())

    neutron_converter_manager = NeutronConverterManager()
    neutron_model = neutron_converter_manager.convert(tflite_model, "imxrt700")

    assert len(neutron_model), "Produced NeutronGraph-based TFLite model has zero length!"
