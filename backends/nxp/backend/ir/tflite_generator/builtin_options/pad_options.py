#
# Copyright 2023 NXP
#
# License: LA_OPT_NXP_Software_License
# See the LICENSE_LA_OPT_NXP_Software_License for more details.
#

import flatbuffers

from executorch.backends.nxp.backend.ir.lib.tflite import BuiltinOptions, BuiltinOperator, PadOptions
from executorch.backends.nxp.backend.ir.tflite_generator.meta import meta


class Pad(meta.BuiltinOptions):

    def __init__(self) -> None:
        super().__init__(BuiltinOptions.BuiltinOptions.PadOptions, BuiltinOperator.BuiltinOperator.PAD)

    def gen_tflite(self, builder: flatbuffers.Builder):
        PadOptions.Start(builder)

        return PadOptions.End(builder)
