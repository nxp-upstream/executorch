# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Main implementation of AoT flow to partition and preprocess for Neutron target
# backends.
#

import logging
from typing import final, List, Optional

from torch.export.exported_program import ExportedProgram

from executorch.backends.nxp.backend.edge_program_converter import EdgeProgramToIRConverter
from executorch.backends.nxp.backend.ir.tensor_formatting import TensorFormat
from executorch.backends.nxp.backend.neutron_converter_manager import NeutronConverterManager
from executorch.backends.nxp.neutron_node_extraction import extract_artifacts_from_neutron_node
from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec


class NeutronCompileSpecBuilder:
    def __init__(self):
        self.compile_spec: List[CompileSpec] = []
        self.compiler_flags = []
        self.output_format = None

    def neutron_compile_spec(
        self,
        config: str,
        extra_flags: Optional[str] = None,
    ):
        """
        Generate compile spec for Neutron NPU

        Args:
            config: Neutron accelerator configuration, e.g. rt700
            extra_flags: Extra flags for the Neutron compiler
        """
        assert (
            self.output_format is None
        ), f"Output format already set to f{self.output_format}"
        self.output_format = "tflite"
        self.compiler_flags = [

        ]
        if extra_flags is not None:
            self.compiler_flags.append(extra_flags)

        return self

    def build(self):
        """
        Generate a list of compile spec objects from the builder
        """
        if self.output_format == "tflite":
            self.compile_spec += [
                CompileSpec("output_format", "tflite".encode()),
                CompileSpec("compile_flags", " ".join(self.compiler_flags).encode()),
            ]

        return self.compile_spec


def generate_neutron_compile_spec(
    config: str,
    system_config: Optional[str] = None,
    extra_flags: Optional[str] = None,
) -> List[CompileSpec]:
    return (
        NeutronCompileSpecBuilder()
        .neutron_compile_spec(
            config,
            extra_flags=extra_flags,
        )
        .build()
    )


@final
class NeutronBackend(BackendDetails):

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_spec: List[CompileSpec],
    ) -> PreprocessResult:
        logging.info("NeutronBackend::preprocess")

        logging.debug(f"NeutronBackend preprocessing graph:\n{edge_program.graph}")

        output_format = ""
        compile_flags = []
        binary = bytes()
        for spec in compile_spec:
            if spec.key == "output_format":
                output_format = spec.value.decode()
            if spec.key == "compile_flags":
                compile_flags.append(spec.value.decode())

        # Check that the output format is set in the compile spec
        if not output_format:
            raise RuntimeError("output format is required")

        for node in edge_program.graph.nodes:
            if node.op == "call_function":
                logging.debug(f"Operator to be processed: {node.target}")

        # Serialize and return the program.
        if output_format == "tflite":
            # Convert the edge program to TFLite.
            tflite_model, io_formats = EdgeProgramToIRConverter().convert_program(edge_program)
            for tensor, tensor_format in io_formats.items():
                if tensor_format == TensorFormat.CHANNELS_LAST:
                    channel_last_format = b'1'
                else:
                    channel_last_format = b'0'

                compile_spec.append(CompileSpec(tensor, channel_last_format))

            # Call the neutron converter with the TFLite model.
            neutron_model = NeutronConverterManager().convert(tflite_model)

            # Extract the Neutron microcode, weights and kernels from the Neutron Node in the `neutron_model`.
            payload = extract_artifacts_from_neutron_node(neutron_model)
            binary = payload.processed_bytes

        else:
            raise RuntimeError(f"Unknown format {output_format}")

        return PreprocessResult(processed_bytes=binary)
