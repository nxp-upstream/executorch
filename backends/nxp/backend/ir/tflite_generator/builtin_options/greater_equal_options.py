#
# Copyright 2024 NXP
#
# License: LA_OPT_NXP_Software_License
# See the LICENSE_LA_OPT_NXP_Software_License for more details.
#

import flatbuffers as fb

import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
from executorch.backends.nxp.backend.ir.lib.tflite import GreaterEqualOptions
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import BuiltinOperator
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions


class GreaterEqual(meta.BuiltinOptions):

    def __init__(self) -> None:
        super().__init__(BuiltinOptions.GreaterEqualOptions, BuiltinOperator.GREATER_EQUAL)

    def gen_tflite(self, builder: fb.Builder):
        GreaterEqualOptions.Start(builder)

        return GreaterEqualOptions.End(builder)
