#
# Copyright 2023 NXP
#
# License: LA_OPT_NXP_Software_License
# See the LICENSE_LA_OPT_NXP_Software_License for more details.
#
"""
    slice_options

Representation of the TFLite operator 'Slice'.
"""

import flatbuffers as fb

import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
from executorch.backends.nxp.backend.ir.lib.tflite import SliceOptions as libSliceOptions
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import BuiltinOperator
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions


class Slice(meta.BuiltinOptions):
    def __init__(self) -> None:
        super().__init__(BuiltinOptions.SliceOptions, BuiltinOperator.SLICE)

    def gen_tflite(self, builder: fb.Builder):
        libSliceOptions.Start(builder)
        return libSliceOptions.End(builder)
