# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.fx import Node
from torch.nn import Parameter

from executorch.backends.nxp.backend.custom_delegation_options import CustomDelegationOptions
from executorch.backends.nxp.backend.ir.converter.conversion import translator
from executorch.backends.nxp.backend.ir.converter.conversion.common import OpsList
from executorch.backends.nxp.backend.ir.converter.node_converter import NodeConverter, Target
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options.concatenation_options import Concatenation


def _get_shape(node: torch.fx.Node) -> list[int]:
    return node.meta['val'].shape


class CatConverter(NodeConverter):

    @staticmethod
    def _get_normalized_dim(node: torch.fx.Node) -> int:
        dim = node.args[1] if len(node.args) >= 2 else 0  # Default `dim` value.
        rank = len(_get_shape(node))
        if dim < 0:
            dim += rank

        if not (0 <= dim < rank):
            raise RuntimeError('`Cat` operator has invalid `dim`.')

        return dim

    @staticmethod
    def _is_supported_on_target(
        node: Node,
        target: Target,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions
    ) -> bool:
        match target:
            case Target.RT700:
                dim = CatConverter._get_normalized_dim(node)

                # neutron-library/src/utils/NeutronLibraryInterrogation.cpp#1491
                if dim == 0:
                    return False

                # Neutron requires the channels to be a multiple of `8`. The channels could either be the second or the
                #  last dimension, depending on the formats of the node. The format, however, cannot be determined
                #  during conversion, as it depends on what other nodes are delegated.
                input_channels = [
                                     # The second dimension is the channels in PyTorch. If the inputs/output are not channels first, it
                                     #  will still be the channels in the IR.
                                     _get_shape(input_)[1] for input_ in node.all_input_nodes
                                 ] + [
                                     # If the inputs/outputs are channels first, the last dimension will be the channels.
                                     _get_shape(input_)[-1] for input_ in node.all_input_nodes
                                 ]
                if any((input_channel % 8) != 0 for input_channel in input_channels):
                    # neutron-library/src/utils/NeutronLibraryInterrogation.cpp#1492
                    return False

                output_channels = [_get_shape(node)[1], _get_shape(node)[-1]]
                # neutron-library/src/utils/NeutronLibraryInterrogation.cpp#1493
                if any((out_c % 8) != 0 for out_c in output_channels):
                    return False

                if len(node.all_input_nodes) < 2:  # Not supported on Neutron
                    # TODO Try to skip the operator if this case is realistic.
                    return False

                return True

            case _:
                return False

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions
    ) -> bool:
        return True

    def convert(self, node: Node):
        """ Convert the 'aten.cat' operator to TFLite 'Concatenation'. """
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)

        dim = self._get_normalized_dim(node)  # Also checks the validity of `dim`.

        if t_op.tmp_inputs[0].tensor_format.is_channels_last():
            dim = translator.create_channels_last_to_channels_first_permutation(t_op.tmp_inputs[0].rank)[dim]

        ops = OpsList(middle_op=t_op)

        if t_op.is_qdq_quantized():
            # The IR requires all inputs to have the same quantization parameters as the output.
            #  https://ai.google.dev/edge/litert/models/quantization_spec
            # The quantizer should quantize the operator so that this case does not happen, but we cannot rely on it.
            output = t_op.tmp_outputs[0]

            output_q_params = list(output.quantization.scale), list(output.quantization.zero_point)
            for input_index, input_ in enumerate(t_op.tmp_inputs):
                if input_.quantization != output.quantization:
                    ops.add_pre(
                        self.builder.create_quantize_operator_before(t_op, input_index, input_.type, *output_q_params)
                    )

        t_op.builtin_options = Concatenation(dim)
        self.builder.append_operators(ops.flatten())
