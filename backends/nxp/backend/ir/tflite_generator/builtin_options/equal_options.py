#
# Copyright 2023 NXP
#
# License: LA_OPT_NXP_Software_License
# See the LICENSE_LA_OPT_NXP_Software_License for more details.
#
import flatbuffers as fb

import executorch.backends.nxp.backend.ir.lib.tflite.EqualOptions as libEqualOptions
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import BuiltinOperator
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions
from executorch.backends.nxp.backend.ir.tflite_generator.meta import meta


class Equal(meta.BuiltinOptions):

    def __init__(self) -> None:
        super().__init__(BuiltinOptions.EqualOptions, BuiltinOperator.EQUAL)

    def gen_tflite(self, builder: fb.Builder):
        libEqualOptions.Start(builder)

        return libEqualOptions.End(builder)
