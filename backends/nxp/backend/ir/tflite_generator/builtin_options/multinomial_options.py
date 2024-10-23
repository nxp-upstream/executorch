#
# Copyright 2024 NXP
#
# License: LA_OPT_NXP_Software_License
# See the LICENSE_LA_OPT_NXP_Software_License for more details.
#

import flatbuffers as fb

import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta
from executorch.backends.nxp.backend.ir.lib.tflite import RandomOptions
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import BuiltinOperator
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions


class Multinomial(meta.BuiltinOptions):
    seed: int
    seed2: int

    def __init__(self, seed: int, seed2: int) -> None:
        super().__init__(BuiltinOptions.RandomOptions, BuiltinOperator.MULTINOMIAL)
        self.seed = seed
        self.seed2 = seed2

    def gen_tflite(self, builder: fb.Builder):
        RandomOptions.Start(builder)

        RandomOptions.AddSeed(builder, self.seed)
        RandomOptions.AddSeed2(builder, self.seed2)

        return RandomOptions.End(builder)
