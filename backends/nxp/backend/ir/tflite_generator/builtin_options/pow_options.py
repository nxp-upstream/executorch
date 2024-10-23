#
# Copyright 2024 NXP
#
# License: LA_OPT_NXP_Software_License
# See the LICENSE_LA_OPT_NXP_Software_License for more details.
#

import flatbuffers

from executorch.backends.nxp.backend.ir.lib.tflite import PowOptions
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import BuiltinOperator
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions
from executorch.backends.nxp.backend.ir.tflite_generator.meta import meta


class Pow(meta.BuiltinOptions):

    def __init__(self) -> None:
        super().__init__(BuiltinOptions.PowOptions, BuiltinOperator.POW)

    def gen_tflite(self, builder: flatbuffers.Builder):
        PowOptions.Start(builder)

        return PowOptions.End(builder)
