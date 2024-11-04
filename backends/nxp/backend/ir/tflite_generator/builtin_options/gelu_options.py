#
# Copyright 2024 NXP
#
# License: LA_OPT_NXP_Software_License
# See the LICENSE_LA_OPT_NXP_Software_License for more details.
#


import flatbuffers as fb

import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
from executorch.backends.nxp.backend.ir.lib.tflite import GeluOptions
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import BuiltinOperator
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions


class Gelu(meta.BuiltinOptions):
    approximate: bool

    def __init__(self, approximate: bool) -> None:
        super().__init__(BuiltinOptions.GeluOptions, BuiltinOperator.GELU)
        self.approximate = approximate

    def gen_tflite(self, builder: fb.Builder):
        GeluOptions.Start(builder)

        GeluOptions.AddApproximate(builder, self.approximate)

        return GeluOptions.End(builder)
