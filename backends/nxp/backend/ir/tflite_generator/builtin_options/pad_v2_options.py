#
# Copyright 2023 NXP
#
# License: LA_OPT_NXP_Software_License
# See the LICENSE_LA_OPT_NXP_Software_License for more details.
#

import flatbuffers

from executorch.backends.nxp.backend.ir.lib.tflite import BuiltinOptions, BuiltinOperator, PadV2Options
from executorch.backends.nxp.backend.ir.tflite_generator.meta import meta


class PadV2(meta.BuiltinOptions):

    def __init__(self) -> None:
        super().__init__(BuiltinOptions.BuiltinOptions.PadV2Options, BuiltinOperator.BuiltinOperator.PADV2)

    def gen_tflite(self, builder: flatbuffers.Builder):
        PadV2Options.Start(builder)

        return PadV2Options.End(builder)
