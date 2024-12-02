from typing import Sequence

import numpy as np
import torch
from torch import nn

from executorch.backends.nxp.backend.ir.converter.builder.model_builder import ModelBuilder
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options.conv_2d_options import Conv2D
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options.reshape_options import Reshape
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options.transpose_options import Transpose
from executorch.backends.nxp.tests.executorch_pipeline import to_edge_program
from executorch.backends.nxp.tests.executors import convert_run_compare, ToNHWCPreprocess, ToNCHWPreprocess


class FormatlessToChannelsFirstModule(nn.Module):
    def __init__(self, channels: int, new_shape: Sequence[int]):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 2, bias=True)
        self.new_shape = new_shape

    def forward(self, x):
        x = torch.reshape(x, self.new_shape)
        x = self.conv(x)
        return x


class FormatlessToFormatlessModule(nn.Module):
    def __init__(self, new_shape: Sequence[int]):
        super().__init__()
        self.new_shape = new_shape

    def forward(self, x):
        x = torch.reshape(x, self.new_shape)
        return x


class ConvReshapeModule(nn.Module):
    def __init__(self, channels: int, new_shape: Sequence[int]):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 2, bias=True)
        self.new_shape = new_shape

    def forward(self, x):
        x = self.conv(x)
        x = torch.reshape(x, self.new_shape)
        return x


def test__channels_first_to_2d(mocker):
    input_shape = [2, 4, 7, 9]
    new_shape = [12, 32]  # Mix up the dimensions for a thorough test.

    torch_model = ConvReshapeModule(channels=input_shape[1], new_shape=new_shape)
    edge_program = to_edge_program(torch_model, input_shape).exported_program()

    torch.manual_seed(23)
    input_data = np.random.random(input_shape).astype('float32')

    converter_spy = mocker.spy(ModelBuilder, "finish")

    convert_run_compare(edge_program, input_data, tflite_input_preprocess=ToNHWCPreprocess())

    tflite_model = converter_spy.spy_return
    ops = tflite_model.sub_graphs[0].operators.vector
    assert len(ops) == 3
    assert isinstance(ops[0].builtin_options, Conv2D)
    assert isinstance(ops[1].builtin_options, Transpose)
    assert isinstance(ops[2].builtin_options, Reshape)


def test__channels_first_to_4d(mocker):
    input_shape = [2, 4, 6, 8]
    new_shape = [7, 4, 2, 5]

    torch_model = ConvReshapeModule(channels=input_shape[1], new_shape=new_shape)
    edge_program = to_edge_program(torch_model, input_shape).exported_program()

    torch.manual_seed(23)
    input_data = np.random.random(input_shape).astype('float32')

    converter_spy = mocker.spy(ModelBuilder, "finish")

    convert_run_compare(edge_program, input_data, tflite_input_preprocess=ToNHWCPreprocess())

    tflite_model = converter_spy.spy_return
    ops = tflite_model.sub_graphs[0].operators.vector
    assert len(ops) == 3
    assert isinstance(ops[0].builtin_options, Conv2D)
    assert isinstance(ops[1].builtin_options, Transpose)
    assert isinstance(ops[2].builtin_options, Reshape)


def test__formatless_to_channels_first(mocker):
    input_shape = [12, 32]
    new_shape = [2, 4, 6, 8]  # Mix up the dimensions for a thorough test.

    torch_model = FormatlessToChannelsFirstModule(channels=new_shape[1], new_shape=new_shape)
    edge_program = to_edge_program(torch_model, input_shape).exported_program()

    torch.manual_seed(23)
    input_data = np.random.random(input_shape).astype('float32')

    converter_spy = mocker.spy(ModelBuilder, "finish")

    convert_run_compare(edge_program, input_data, tflite_output_preprocess=ToNCHWPreprocess())

    tflite_model = converter_spy.spy_return
    ops = tflite_model.sub_graphs[0].operators.vector
    assert len(ops) == 3
    assert isinstance(ops[0].builtin_options, Reshape)
    assert isinstance(ops[1].builtin_options, Transpose)
    assert isinstance(ops[2].builtin_options, Conv2D)


def test__formatless_to_formatless(mocker):
    input_shape = [12, 32]
    new_shape = [2, 4, 6, 8]

    torch_model = FormatlessToFormatlessModule(new_shape=new_shape)
    edge_program = to_edge_program(torch_model, input_shape).exported_program()

    torch.manual_seed(23)
    input_data = np.random.random(input_shape).astype('float32')

    converter_spy = mocker.spy(ModelBuilder, "finish")

    convert_run_compare(edge_program, input_data)

    tflite_model = converter_spy.spy_return
    ops = tflite_model.sub_graphs[0].operators.vector
    assert len(ops) == 1  # No extra Transpose ops.
    assert isinstance(ops[0].builtin_options, Reshape)
