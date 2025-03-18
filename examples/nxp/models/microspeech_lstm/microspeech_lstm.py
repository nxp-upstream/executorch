# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
from typing import Iterator

import torch
from torch import nn
from torch.utils.data import DataLoader

from executorch.examples.models import model_base


class MicroSpeechLSTM(model_base.EagerModelBase):

    def __init__(self):
        self._weights_file = os.path.join(os.path.dirname(__file__), 'microspeech_lstm_best.pth')
        calibration_dataset_path = os.path.join(os.path.dirname(__file__), 'calibration_data.pt')
        self._calibration_dataset = torch.load(calibration_dataset_path, map_location=torch.device('cpu'))

    @staticmethod
    def _collate_fn(data: list[tuple]):
        data, labels = zip(*data)
        return torch.stack(list(data)), torch.tensor(list(labels))

    def get_eager_model(self) -> torch.nn.Module:
        lstm_module = MicroSpeechLSTMModule()
        lstm_module.load_state_dict(torch.load(self._weights_file, weights_only=True, map_location=torch.device('cpu')))

        return lstm_module

    def get_example_inputs(self) -> tuple[torch.Tensor]:
        sample = self._calibration_dataset[0]

        return (sample[0].unsqueeze(0),)  # use only sample data, not label

    def get_calibration_inputs(self, batch_size: int = 1) -> Iterator[tuple[torch.Tensor]]:
        """
        Get LSTM's calibration input. Batch size is ignored and set to 1 because hidden state
        has to be initialized to the size of batch.

        :param batch_size: Ignored.
        :return: Iterator with input calibration dataset samples.
        """
        data_loader = DataLoader(self._calibration_dataset, batch_size=1)
        return itertools.starmap(lambda data, label: (data,), iter(data_loader))


class MicroSpeechLSTMModule(nn.Module):

    def __init__(self, input_size=80, output_size=3, features_size=128):
        super(MicroSpeechLSTMModule, self).__init__()
        self.lstm = nn.LSTM(input_size, features_size, batch_first=True)
        self.linear = nn.Linear(features_size, output_size)

    def forward(self, x):
        _, (x, _) = self.lstm(x)
        x = x.squeeze(0)
        x = self.linear(x)
        x = nn.functional.softmax(x, dim=1)
        return x
