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

from executorch.backends.nxp.backend.edge_program_converter import EdgeProgramToIRConverter
from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch.export.exported_program import ExportedProgram
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
            tflite_model = EdgeProgramToIRConverter().convert_program(edge_program)
            logging.debug("Here we serialize to TFLite for the Neutron Convertor")
            # TODO: Actually serialize to TFLite for Neutron Convertor
        else:
            raise RuntimeError(f"Unknown format {output_format}")

        return PreprocessResult(processed_bytes=binary)
