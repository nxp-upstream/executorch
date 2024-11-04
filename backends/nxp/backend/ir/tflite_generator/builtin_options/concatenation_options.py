#
# Copyright 2023 NXP
#
# License: LA_OPT_NXP_Software_License
# See the LICENSE_LA_OPT_NXP_Software_License for more details.
#

import flatbuffers as fb

import executorch.backends.nxp.backend.ir.lib.tflite.ActivationFunctionType as libActivationFunctionType
import executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator as libBuiltinOperator
import executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions as libBuiltinOptions
import executorch.backends.nxp.backend.ir.lib.tflite.ConcatenationOptions as libConcatenationOptions
import executorch.backends.nxp.backend.ir.tflite_generator.meta.meta as meta


class Concatenation(meta.BuiltinOptions):
    axis: int
    fused_activation_function: libActivationFunctionType.ActivationFunctionType

    def __init__(self, axis: int,
                 fused_activation_function: libActivationFunctionType.ActivationFunctionType = libActivationFunctionType.ActivationFunctionType.NONE) -> None:
        super().__init__(libBuiltinOptions.BuiltinOptions.ConcatenationOptions,
                         libBuiltinOperator.BuiltinOperator.CONCATENATION)
        self.axis = axis
        self.fused_activation_function = fused_activation_function

    def gen_tflite(self, builder: fb.Builder):
        libConcatenationOptions.Start(builder)

        libConcatenationOptions.AddAxis(builder, self.axis)
        libConcatenationOptions.AddFusedActivationFunction(builder, self.fused_activation_function)

        return libConcatenationOptions.End(builder)
