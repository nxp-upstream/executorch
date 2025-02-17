# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch

from executorch.examples.nxp.models.mlperf_tiny.mlperf_tiny_model import MLPerfTinyModel


class AnomalyDetection(MLPerfTinyModel):
    _input_dim: int = 5 * 128
    __input_shape: tuple = (1, 640)

    def __init__(self):
        super().__init__()
        self.__dts_path = os.path.join(self._calibration_data_path, "anomaly_detection/calibration_data.xz")

    @property
    def _dts_path(self):
        return self.__dts_path

    @property
    def _input_shape(self):
        return self.__input_shape

    def get_eager_model(self) -> torch.nn.Module:
        return self._model_manager.get_model("anomaly_detection", input_dimension=self._input_dim)
