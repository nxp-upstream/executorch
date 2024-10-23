#
# Copyright 2023 NXP
#
# License: LA_OPT_NXP_Software_License
# See the LICENSE_LA_OPT_NXP_Software_License for more details.
#
import flatbuffers as fb

import executorch.backends.nxp.backend.ir.lib.tflite.SplitOptions as libSplitOptions
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import BuiltinOperator
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions
from executorch.backends.nxp.backend.ir.tflite_generator.meta import meta


class Split(meta.BuiltinOptions):
    num_splits: int

    def __init__(self, num_splits: int) -> None:
        super().__init__(BuiltinOptions.SplitOptions, BuiltinOperator.SPLIT)
        self.num_splits = num_splits

    def gen_tflite(self, builder: fb.Builder):
        libSplitOptions.Start(builder)
        
        libSplitOptions.AddNumSplits(builder, self.num_splits)

        return libSplitOptions.End(builder)
