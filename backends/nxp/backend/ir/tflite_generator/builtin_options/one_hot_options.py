#
# Copyright 2024 NXP
#
# License: LA_OPT_NXP_Software_License
# See the LICENSE_LA_OPT_NXP_Software_License for more details.
#

import flatbuffers as fb

import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
from executorch.backends.nxp.backend.ir.lib.tflite import OneHotOptions
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import BuiltinOperator
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions


class OneHot(meta.BuiltinOptions):
    axis: int

    def __init__(self, axis: int) -> None:
        super().__init__(BuiltinOptions.OneHotOptions, BuiltinOperator.ONE_HOT)
        self.axis = axis

    def gen_tflite(self, builder: fb.Builder):
        OneHotOptions.Start(builder)

        OneHotOptions.AddAxis(builder, self.axis)

        return OneHotOptions.End(builder)
