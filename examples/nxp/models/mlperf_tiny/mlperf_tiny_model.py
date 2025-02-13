# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from abc import abstractmethod

import torch
from torch.utils.data import DataLoader

from executorch.examples.models import model_base
from executorch.examples.nxp.models.mlperf_tiny.calibration_dataset import CalibrationDataset
from executorch.examples.nxp.models.model_manager import ModelManager


class MLPerfTinyModel(model_base.EagerModelBase):
    def __init__(self):
        self._batch_size = 1
        self._dataset = None
        self._model_manager = ModelManager()
        self._num_workers = 4
        mlperf_tiny_path = os.path.dirname(os.path.abspath(__file__))
        self._calibration_data_path = os.path.join(mlperf_tiny_path, "../../../../data/calibration_data")

    @property
    @abstractmethod
    def _dts_path(self):
        pass

    @property
    @abstractmethod
    def _input_shape(self):
        pass

    @abstractmethod
    def _collate_fn(self, data: torch.Tensor):
        pass

    @abstractmethod
    def get_eager_model(self) -> torch.nn.Module:
        pass

    def get_example_inputs(self) -> tuple[torch.Tensor]:
        return (torch.randn(self._input_shape, dtype=torch.float32),)

    def _get_data_loader(self):
        self._init_dataset()
        data_loader = DataLoader(self._dataset, batch_size=self._batch_size,
                                 collate_fn=self._collate_fn,
                                 num_workers=self._num_workers, pin_memory=True)
        return data_loader

    def _init_dataset(self):
        if self._dataset is None:
            self._dataset = CalibrationDataset(self._dts_path)
