#
# Copyright 2023 NXP
#
# License: LA_OPT_NXP_Software_License
# See the LICENSE_LA_OPT_NXP_Software_License for more details.
#
import flatbuffers as fb

from executorch.backends.nxp.backend.ir.lib.tflite import ReducerOptions
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import BuiltinOperator
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions
from executorch.backends.nxp.backend.ir.tflite_generator.meta import meta


class Mean(meta.BuiltinOptions):
    keep_dims: bool

    def __init__(self, keep_dims: bool) -> None:
        super().__init__(BuiltinOptions.ReducerOptions, BuiltinOperator.MEAN)
        self.keep_dims = keep_dims

    def gen_tflite(self, builder: fb.Builder):
        ReducerOptions.Start(builder)

        ReducerOptions.AddKeepDims(builder, self.keep_dims)

        return ReducerOptions.End(builder)
