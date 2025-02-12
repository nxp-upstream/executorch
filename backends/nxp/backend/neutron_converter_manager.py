# Copyright (c) 2024 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from neutron_converter_wrapper import neutron_converter


class NeutronConverterManager:
    """
    Manager for conversion of TFLite model in flatbuffers format into TFLite model that
    contains NeutronGraph nodes.
    """

    def convert(self, tflite_model: bytes, target: str) -> bytes:
        cctx = neutron_converter.CompilationContext()
        cctx.targetOpts = neutron_converter.getNeutronTarget(target)
        model_converted = neutron_converter.convertModel(list(tflite_model), cctx)

        return bytes(model_converted)
