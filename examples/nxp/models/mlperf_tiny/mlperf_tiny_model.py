# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
from abc import abstractmethod
from typing import Iterator

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

    @staticmethod
    def _collate_fn(data: list[tuple]):
        data, labels = zip(*data)
        return torch.stack(list(data)), torch.tensor(list(labels))

    def get_calibration_inputs(self, batch_size: int = 1) -> Iterator[tuple[torch.Tensor]]:
        self._batch_size = batch_size
        data_loader = self._get_data_loader()
        get_first = lambda a, b: (a,)
        return itertools.starmap(get_first, iter(data_loader))

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
            os.makedirs(os.path.dirname(self._dts_path), exist_ok=True)
            if not os.path.exists(self._dts_path):
                raise FileNotFoundError("Calibration data file not found! For more info, follow README.md.")
            self._dataset = CalibrationDataset(self._dts_path)
