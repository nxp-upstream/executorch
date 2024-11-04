#
# Copyright 2023 NXP
#
# License: LA_OPT_NXP_Software_License
# See the LICENSE_LA_OPT_NXP_Software_License for more details.
#

import flatbuffers as fb

import executorch.backends.nxp.backend.ir.lib.tflite.GatherOptions as libGatherOptions
import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import BuiltinOperator
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions


class Gather(meta.BuiltinOptions):
    axis: int
    batch_dims: int

    def __init__(self, axis: int, batch_dims: int = 0) -> None:
        super().__init__(BuiltinOptions.GatherOptions, BuiltinOperator.GATHER)
        self.axis = axis
        self.batch_dims = batch_dims

    def gen_tflite(self, builder: fb.Builder):
        libGatherOptions.Start(builder)

        libGatherOptions.AddAxis(builder, self.axis)
        libGatherOptions.AddBatchDims(builder, self.batch_dims)

        return libGatherOptions.End(builder)
