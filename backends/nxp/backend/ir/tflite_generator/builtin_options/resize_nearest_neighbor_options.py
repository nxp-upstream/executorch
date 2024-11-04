#
# Copyright 2024 NXP
#
# License: LA_OPT_NXP_Software_License
# See the LICENSE_LA_OPT_NXP_Software_License for more details.
#

import flatbuffers as fb

from executorch.backends.nxp.backend.ir.lib.tflite import ResizeNearestNeighborOptions
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import BuiltinOperator
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions
from executorch.backends.nxp.backend.ir.tflite_generator.meta import meta


# noinspection SpellCheckingInspection
class ResizeNearestNeighbor(meta.BuiltinOptions):
    align_corners: bool
    half_pixel_centers: bool

    def __init__(self, align_corners: bool, half_pixel_centers: bool) -> None:
        super().__init__(BuiltinOptions.ResizeNearestNeighborOptions, BuiltinOperator.RESIZE_NEAREST_NEIGHBOR)
        self.align_corners = align_corners
        self.half_pixel_centers = half_pixel_centers

    def gen_tflite(self, builder: fb.Builder):
        ResizeNearestNeighborOptions.Start(builder)

        ResizeNearestNeighborOptions.AddAlignCorners(builder, self.align_corners)
        ResizeNearestNeighborOptions.AddHalfPixelCenters(builder, self.half_pixel_centers)

        return ResizeNearestNeighborOptions.End(builder)
