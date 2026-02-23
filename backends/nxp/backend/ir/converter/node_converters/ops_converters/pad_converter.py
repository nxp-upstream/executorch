# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import  Sequence
from executorch.backends.nxp.backend.ir.lib.tflite.MirrorPadMode import MirrorPadMode

import numpy as np

from executorch.backends.nxp.backend import edge_helper
from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    sub_options, mirror_pad_options,
)
from executorch.backends.nxp.backend.ir.tflite_generator.tflite_model import (
    Quantization,
    Scale,
    ZeroPoint,
)
from torch.fx import Node
from torch.nn import Parameter

from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec


class PadConverter(NodeConverter):

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        paddings = node.args[1]
        if not (isinstance(paddings, int) or isinstance(paddings, Sequence)):
            return False

        input_shape = node.args[0].meta['val'].shape
        spatial_dims = input_shape[2:]
        if isinstance(paddings, int):
            if any(paddings >= dim for dim in spatial_dims):
                # Neutron restriction: the padding must always be smaller than the size of the padded dimension.
                return False

        elif isinstance(paddings, Sequence):
            # If `paddings` is a sequence, it contains left and right padding for every spatial dimension.
            #  torch.nn.ReflectionPad2d documentation states: (padding_leftpadding_left, padding_rightpadding_right,
            #  padding_toppadding_top, padding_bottompadding_bottom)
            if len(paddings) != len(spatial_dims) * 2:
                return False



        return True

    @staticmethod
    def _is_supported_on_target(
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        mode = node.args[2]
        if mode != "reflect":
            return False

        # TODO

        return True

    def _convert_reflect_mode(self, node: Node):
        t_op = self._create_tflite_op_with_io_tensors(node)
        x = t_op.tmp_inputs[0]

        paddings = node.args[1]
        paddings = np.array([[0,0], [1,1], [1,1], [0,0]], dtype="int32")


        # The Neutron IR MirrorPad requires the `paddings` to be an input tensor, not an argument.
        paddings_tensor = self.builder.create_tensor_for_data(
            paddings, "paddings"
        )

        # Neutron only supports the `REFLECT` mode (`0`).
        t_op.builtin_options = mirror_pad_options.MirrorPad(MirrorPadMode.REFLECT)
        t_op.tmp_inputs = [x, paddings_tensor]

        self.builder.append_operators([t_op])


    def convert(self, node: Node):
        """Convert 'aten.pad.default' with mode="reflect" to NeutronIR `ReflectionPad`.

        The ExecuTorch schema is
            aten::pad(
                Tensor self,
                SymInt[] pad,
                str mode="constant",
                float? value=None
            ) -> Tensor
        """
        self.assert_convertible(node)

        mode = node.args[2]
        if mode == "reflect":
            self._convert_reflect_mode(node)

        else:
            # Should never happen.
            raise ValueError(f"`aten.pad.default` node {node} was incorrectly selected for delegation. There is an "
                             "issue wit the NXP backend. Please report this.")




