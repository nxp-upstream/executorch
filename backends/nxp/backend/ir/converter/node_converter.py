#
# Copyright 2024 NXP
#
# License: LA_OPT_NXP_Software_License
# See the LICENSE_LA_OPT_NXP_Software_License for more details.
#
from abc import ABC, abstractmethod
from typing import Tuple

from torch import Node

from executorch.backends.nxp.backend.ir.conversion_context import ConversionContext
from executorch.backends.nxp.backend.ir.converter.builder import model_builder
from executorch.backends.nxp.backend.ir.converter.conversion import translator
from executorch.backends.nxp.backend.ir.tensor_formatting import TensorFormat
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model


class NodeConverter(ABC):
    """
        Classes which implement conversion of torch.Node to TFLite should inherit from this class and overwrite the
         'convert()' method.
    """
    context: ConversionContext

    def __init__(self, context: ConversionContext):
        self.context = context

    @abstractmethod
    def convert(self, node: Node):
        """ Convert the torch.Node in 'node' to TFLite and append changes to ModelBuilder.

            Classes which implement conversion for individual operators must overwrite this method.

        :param node: torch.Node to convert.
        """
        pass

    @property
    def builder(self) -> model_builder.ModelBuilder:
        """
        Get instance of TFLite ModelBuilder from conversion context.
        :return: ModelBuilder instance.
        """
        return self.context.tflite_builder

    def _append_io_tensors_and_get_tflite_op(self, node: Node) -> tflite_model.Operator:
        """
        Append node's inputs/output as tensors and create TFLite op wrapper with such tensors
        added into 'tmp_inputs' and 'tmp_outputs'.

        :param node: Node instance.
        :return: TFLite operator with assigned input/output tensors.
        """
        t_operator = tflite_model.Operator()

        # Initialize node's inputs
        t_operator.inputs = tflite_model.OperatorInputs()
        for ancestor_node in node.all_input_nodes:
            assert self.context.tflite_builder.tensor_exists(ancestor_node.name)
            t_operator.tmp_inputs.append(self.context.tflite_builder.tensor_for_name(ancestor_node.name))

        # Add node's output as a new tensor
        t_operator.outputs = tflite_model.OperatorOutputs()
        if not self.context.tflite_builder.tensor_exists(node.name):
            self._append_as_fake_tensor(node)
        t_operator.tmp_outputs.append(self.context.tflite_builder.tensor_for_name(node.name))

        return t_operator

    def _append_operators(self, ops_to_add: list[tflite_model.Operator]):
        """
        Append list of TFLite operators to created model via ModelBuilder.

        :param ops_to_add: List of operators to be added.
        """
        for op in ops_to_add:
            if op.builtin_options is not None:
                op.opcode_index = self.builder.op_code_index_for_op_type(
                    op.builtin_options.operator_type,
                    op.tmp_version
                )

            elif op.custom_options is not None:
                op.opcode_index = self.builder.op_code_index_for_op_type(
                    op.custom_options.operator_type,
                    op.tmp_version,
                    op.custom_options.custom_code
                )

            self.builder.check_and_append_operator(op)

    def _append_as_fake_tensor(self, node: Node):
        """
        Append node into ModelBuilder as tensor without data (FakeTensor). Can be used
        for activations and output tensors.

        :param node: Node instance.
        """
        if self.builder.tensor_exists(node.name):
            return
        assert not self.builder.tensor_exists(node.name), f"Tensor '{node.name}' already added!"

        tensor = node.meta["val"]
        if isinstance(tensor, Tuple):
            tensor = tensor[0]  # Fake tensor
        _type = translator.convert_data_type(tensor.dtype)
        shape = list(tensor.shape)

        if self.context.node_formats[node].is_channels_first():
            shape = translator.dims_to_channels_last(shape)

        tensor = self.builder.create_empty_tensor(node.name, _type, shape)
        tensor.tensor_format = TensorFormat.from_node_format(self.context.node_formats[node])

    def _append_as_static_tensor(self, node: Node):
        """
        Append node into ModelBuilder as tensor with data (static). Can be used for weights,
        permutations etc.

        :param node: Node instance.
        """
        assert not self.builder.tensor_exists(node.name), f"Tensor '{node.name}' already added!"

        tensor = self.context.parameters_mapping[node.name]
        data = tensor.data.numpy()

        if self.context.node_formats[node].is_channels_first():
            data = translator.convert_data_to_channels_last(data)

        tensor = self.builder.create_tensor_for_data(data, node.name)
        tensor.tensor_format = TensorFormat.from_node_format(self.context.node_formats[node])
