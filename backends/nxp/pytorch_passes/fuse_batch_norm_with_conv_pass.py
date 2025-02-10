# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.export.unflatten import _AttrKind, _assign_attr
from torch.fx import Node
from torch.nn.utils import fuse_conv_bn_weights

from executorch.backends.nxp.pytorch_passes.nxp_pytorch_pass import NXPPyTorchPass


class FuseBatchNormWithConvPass(NXPPyTorchPass):
    """ The executorch batch normalization carries out the following computation [1].

            (x - mean) / (var + eps) * W + B

        Which can be expressed as

            x * (W / sqrt(var + eps)) + (B - mean * (W / sqrt(var + eps)))

        So the batch norm can be done as 1 multiplication and 1 addition, provided that the parameters are static,
         and the terms can be precomputed. If there is a `Conv` operator before the batch normalization, this scale and
         bias can be statically integrated into the weights and bias of the `Conv`, which allows the batch norm to be
         completely removed.
         
        
                                   │
                     ┌─────────────▼─────────────┐
                     │ aten.conv1d | aten.conv2d │
                     └─────────────┬─────────────┘
                                   │                                                                │
             ┌─────────────────────▼─────────────────────┐        replace with        ┌─────────────▼─────────────┐
             │ aten._native_batch_norm_legit_no_training │       ──────────────►      │ aten.conv1d | aten.conv2d │
             └─────────────────────┬─────────────────────┘                            └─────────────┬─────────────┘
                                   │                                                                ▼
                             ┌─────▼──────┐
                             │ getitem(0) │
                             └─────┬──────┘
                                   ▼

        [1] https://github.com/pytorch/executorch/blob/v0.5.0-rc2/kernels/portable/cpu/op_native_batch_norm.cpp#L118-L128
    """

    def run(self) -> bool:
        def _is_batch_norm(node_: Node) -> bool:
            return node_.op == "call_function" and node_.target == torch.ops.aten._native_batch_norm_legit_no_training.default

        def _is_conv(node_: Node):
            return node_.op == "call_function" and node_.target in (
                torch.ops.aten.conv1d.default,
                torch.ops.aten.conv2d.default
            )

        def _is_getitem(node_: Node) -> bool:
            return node_.op == "call_function" and node_.target.__name__ == "getitem"

        made_changes = False

        if not any(map(_is_batch_norm, self.module.graph.nodes)):
            return made_changes  # No batch norm nodes in the model.

        for node in self.module.graph.nodes:
            if not _is_batch_norm(node):
                continue  # Not BatchNorm.

            bn_node = node
            if not all(_is_getitem(user) and user.args[1] == 0 for user in bn_node.users):
                # Nodes other than just `getitem(0)` follow after the BatchNorm. Probably `getitem` nodes accessing
                #  other outputs of the BN. After the fusion with a Conv op, only the first output can be accessed.
                continue

            if not _is_conv(bn_node.args[0]):
                continue  # Something other than a Conv node comes before the BatchNorm.

            conv_node = bn_node.args[0]
            conv_weight_node = conv_node.args[1]
            conv_bias_node = conv_node.args[2] if len(conv_node.args) > 2 else None

            # conv args: input, weight, bias, stride, padding, dilation, ...
            conv_w = self.get_tensor_constant_from_node(conv_weight_node)
            conv_b = self.get_tensor_constant_from_node(conv_bias_node)

            # batch norm legit no training args: input, weight, bias, running mean, running var, momentum, eps
            bn_w = self.get_tensor_constant_from_node(bn_node.args[1])
            bn_b = self.get_tensor_constant_from_node(bn_node.args[2])
            bn_rm = self.get_tensor_constant_from_node(bn_node.args[3])
            bn_rv = self.get_tensor_constant_from_node(bn_node.args[4])
            bn_eps = bn_node.args[6]

            if any(t is None for t in (conv_w, bn_rm, bn_rv)):  # The other inputs can be None.
                continue  # The data is not static. Leave this BatchNorm as is (probably a rare case).
            fused_weight, fused_bias = fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b)

            # Update the weight and bias for Conv.
            conv_args = list(conv_node.args)
            if len(conv_args) == 2:
                # Fill in the default bias argument.
                conv_args.append(None)

            weight_attr_name = conv_weight_node.target
            _assign_attr(fused_weight, self.module, weight_attr_name, _AttrKind.PARAMETER)

            if conv_bias_node is not None:
                bias_attr_name = conv_bias_node.target
                _assign_attr(fused_bias, self.module, str(bias_attr_name), _AttrKind.PARAMETER)
            else:
                # The Conv doesn't have a bias. Create a new one.
                bias_attr_name = weight_attr_name + "_bias"
                _assign_attr(fused_bias, self.module, bias_attr_name, _AttrKind.PARAMETER)
                with self.module.graph.inserting_before(conv_node):
                    get_bias_node = self.module.graph.get_attr(bias_attr_name)

                conv_args[2] = get_bias_node

            conv_node.args = tuple(conv_args)

            # Replace the uses of the BatchNorm with the Conv.
            for user in bn_node.users:
                user.replace_all_uses_with(conv_node)

            made_changes = True

        return made_changes
