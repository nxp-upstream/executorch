#
# Copyright 2024 NXP
#
# License: LA_OPT_NXP_Software_License
# See the LICENSE_LA_OPT_NXP_Software_License for more details.
#

import flatbuffers as fb

import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
from executorch.backends.nxp.backend.ir.lib.tflite import SpaceToDepthOptions
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import BuiltinOperator
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions


class SpaceToDepth(meta.BuiltinOptions):
    block_size: int

    def __init__(self, block_size: int) -> None:
        super().__init__(BuiltinOptions.SpaceToDepthOptions, BuiltinOperator.SPACE_TO_DEPTH)
        self.block_size = block_size

    def gen_tflite(self, builder: fb.Builder):
        SpaceToDepthOptions.Start(builder)

        SpaceToDepthOptions.SpaceToDepthOptionsAddBlockSize(builder, self.block_size)

        return SpaceToDepthOptions.End(builder)
