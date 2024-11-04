#
# Copyright 2024 NXP
#
# License: LA_OPT_NXP_Software_License
# See the LICENSE_LA_OPT_NXP_Software_License for more details.
#

import flatbuffers as fb

import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
from executorch.backends.nxp.backend.ir.lib.tflite import LogicalAndOptions
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import BuiltinOperator
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions


class LogicalAnd(meta.BuiltinOptions):

    def __init__(self) -> None:
        super().__init__(BuiltinOptions.LogicalAndOptions, BuiltinOperator.LOGICAL_AND)

    def gen_tflite(self, builder: fb.Builder):
        LogicalAndOptions.Start(builder)

        return LogicalAndOptions.End(builder)
