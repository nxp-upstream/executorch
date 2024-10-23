#
# Copyright 2023 NXP
#
# License: LA_OPT_NXP_Software_License
# See the LICENSE_LA_OPT_NXP_Software_License for more details.
#


import flatbuffers as fb

import executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator as libBuiltinOperator
import executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions as libBuiltinOptions
import executorch.backends.nxp.backend.ir.lib.tflite.SelectV2Options as libSelectV2Options
import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta


class SelectV2(meta.BuiltinOptions):

    def __init__(self) -> None:
        super().__init__(libBuiltinOptions.BuiltinOptions.SelectV2Options,
                         libBuiltinOperator.BuiltinOperator.SELECT_V2)

    def gen_tflite(self, builder: fb.Builder):
        libSelectV2Options.Start(builder)
        return libSelectV2Options.End(builder)
