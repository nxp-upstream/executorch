# Copyright (c) 2024 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import flatbuffers
from torch import Node
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind
from torch.nn.parameter import Parameter

import executorch.backends.nxp.backend.ir.logger as logger
from executorch.backends.nxp.backend.ir.conversion_config import ConversionConfig
from executorch.backends.nxp.backend.ir.conversion_context import ConversionContext
from executorch.backends.nxp.backend.ir.converter.builder.model_builder import ModelBuilder
from executorch.backends.nxp.backend.ir.converter.node_converters.call_function_converter import CallFunctionConverter
from executorch.backends.nxp.backend.ir.converter.node_converters.output_converter import OutputConverter
from executorch.backends.nxp.backend.ir.converter.node_converters.placeholder_converter import PlaceholderConverter
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model
from executorch.backends.nxp.backend.node_format_inference import NodeFormatInference, NodeFormat

class EdgeProgramToIRConverter:
    """
    Converter from convertion of ExportedProgram in Edge dialect to IR (TFLite Flatbuffers).
    """
    conversion_context: ConversionContext | None

    def __init__(self):
        self.conversion_context = None

    def convert_program(self, edge_program: ExportedProgram, conversion_config = ConversionConfig()) -> bytes:
        """
        Convert ExportedProgram in Edge dialect to IR (TFLite flatbuffers) as bytes.

        :param edge_program: Converter ExportedProgram.
        :param conversion_config: ConversionConfig instance.
        :return: TFLite flatbuffers as bytes.
        """
        node_formats = NodeFormatInference(edge_program).identify_node_formats()
        parameters_mapping = self._map_inputs_to_parameters(edge_program)

        cc = self._build_conversion_context(parameters_mapping, node_formats, conversion_config)

        # Program conversion
        self._process_nodes(edge_program.graph.nodes, cc)
        self._assign_model_io_to_subgraph(edge_program.graph_signature, cc)

        # TFLite model generation
        internal_tflite_model = cc.tflite_builder.finish()
        flatbuffers_builder = flatbuffers.Builder()
        internal_tflite_model.gen_tflite(flatbuffers_builder)

        return bytes(flatbuffers_builder.Output())

    def _process_nodes(self, nodes: list[Node], conversion_context: ConversionContext):
        """
        Go through program nodes and append their TFLite's sibling into ModelBuilder.

        :param nodes: Program's nodes.
        :param conversion_context: ConversionConxtext instance.
        """
        node_converters = {
            "placeholder": PlaceholderConverter,
            "output": OutputConverter,
            "call_function": CallFunctionConverter,
        }

        for node in nodes:
            if node.op not in node_converters:
                logger.e(logger.Code.UNSUPPORTED_NODE, f"Node with op type '{node.op}' is not supported.")

            node_converters[node.op](conversion_context).convert(node)


    def _map_inputs_to_parameters(self, edge_program: ExportedProgram) -> dict[str, Parameter]:
        """
        Create mapping between program parameters (input nodes & static data nodes) and their names.

        :param edge_program: EdgeProgram instance.
        :return: Mapping from parameter name to parameter instance.
        """
        result_map = {}

        for input_spec in edge_program.graph_signature.input_specs:
            if input_spec.kind == InputKind.PARAMETER:
                result_map[input_spec.arg.name] = edge_program.state_dict[input_spec.target]

        return result_map

    def _build_conversion_context(
            self,
            parameters_mapping: dict,
            node_formats: dict[Node, NodeFormat],
            conversion_config: ConversionConfig = ConversionConfig(),
    ) -> ConversionContext:
        tflite_builder = ModelBuilder(3, "TFLite from EdgeProgram", conversion_config)

        # Add "sentinel" buffer (defined in schema.fbs)
        tflite_builder.build_empty_buffer()

        context = ConversionContext(tflite_builder, conversion_config, parameters_mapping, node_formats)

        return context

    def _assign_model_io_to_subgraph(self, graph_signature, conversion_context):
        """
        Assign model's inputs/outputs to SubGraph.

        :param graph_signature: Instance of GraphSignature.
        :param conversion_context: Conversion context.
        """
        model_builder = conversion_context.tflite_builder
        model_builder.get_sub_graph().inputs = tflite_model.SubGraphInputs()
        for input_name in graph_signature.user_inputs:
            tensor = model_builder.tensor_for_name(input_name)
            model_builder.get_sub_graph().inputs.tmp_inputs.append(tensor)

        model_builder.get_sub_graph().outputs = tflite_model.SubGraphOutputs()
        for output_name in graph_signature.user_outputs:
            tensor = model_builder.tensor_for_name(output_name)
            model_builder.get_sub_graph().outputs.tmp_outputs.append(tensor)
