#
# Copyright 2024 NXP
#
# License: LA_OPT_NXP_Software_License
# See the LICENSE_LA_OPT_NXP_Software_License for more details.
#
from abc import ABC, abstractmethod

from torch.fx import Node

from executorch.backends.nxp.backend.ir.conversion_context import ConversionContext
from executorch.backends.nxp.backend.ir.converter.builder.aten_model_builder_director import AtenModelBuilderDirector
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
    def builder(self) -> AtenModelBuilderDirector:
        """
        Get instance of TFLite ModelBuilder from conversion context.
        :return: AtenModelBuilderDirector instance.
        """
        return self.context.tflite_builder

    def _create_tflite_op_with_io_tensors(self, node: Node) -> tflite_model.Operator:
        """
        Create TFLite op wrapper with input/output tensors added into 'tmp_inputs' and 'tmp_outputs'.

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
        assert self.context.tflite_builder.tensor_exists(node.name)
        t_operator.outputs = tflite_model.OperatorOutputs()
        t_operator.tmp_outputs.append(self.context.tflite_builder.tensor_for_name(node.name))

        return t_operator
