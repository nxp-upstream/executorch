# Copyright (c) 2024 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Tests for NeutronQuantizer.

import torch
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e

from executorch.backends.nxp.quantizer.neutron_quantizer import NeutronQuantizer
import executorch.backends.nxp.tests.models as models

def _get_target_name(node):
    return node._pretty_print_target(node.target)
def test_quantizer_conv2d():
    model = models.Conv2dModule()
    model.eval()

    example_input = (torch.ones(1, 4, 32, 32),)
    quantizer = NeutronQuantizer()
    exir_program = torch._export.capture_pre_autograd_graph(model, example_input)

    m = prepare_pt2e(exir_program, quantizer)
    m(*example_input)
    m = convert_pt2e(m)
    # print(f"Quantized model: {m}")
    nodes = list(m.graph.nodes)

    assert len(nodes) == 11
    assert nodes[7].name == "conv2d"
    # [0]: Input, [1] : weights, [2]: bias
    assert _get_target_name(nodes[7].args[0]) == 'torch.ops.quantized_decomposed.dequantize_per_tensor.default'
    assert _get_target_name(nodes[7].args[1]) == 'torch.ops.quantized_decomposed.dequantize_per_tensor.default'
    assert _get_target_name(nodes[7].args[2]) == 'torch.ops.quantized_decomposed.dequantize_per_tensor.default'
    assert _get_target_name(nodes[8]) == 'torch.ops.quantized_decomposed.quantize_per_tensor.default'
    assert nodes[8].args[0].name == "conv2d"

def test_quantizer_linear():
    model = models.LinearModule(bias=True)
    model.eval()

    example_input = (torch.ones(10, 32),)
    quantizer = NeutronQuantizer()
    exir_program = torch._export.capture_pre_autograd_graph(model, example_input)

    m = prepare_pt2e(exir_program, quantizer)
    m(*example_input)
    m = convert_pt2e(m)
    # print(f"Quantized model: {m}")
    nodes = list(m.graph.nodes)

    assert len(nodes) == 11
    assert nodes[7].name == "linear"
    # [0]: Input, [1] : weights, [2]: bias
    assert _get_target_name(nodes[7].args[0]) == 'torch.ops.quantized_decomposed.dequantize_per_tensor.default'
    assert _get_target_name(nodes[7].args[1]) == 'torch.ops.quantized_decomposed.dequantize_per_tensor.default'
    assert _get_target_name(nodes[7].args[2]) == 'torch.ops.quantized_decomposed.dequantize_per_tensor.default'
    assert _get_target_name(nodes[8]) == 'torch.ops.quantized_decomposed.quantize_per_tensor.default'
    assert nodes[8].args[0].name == "linear"

def test_quantizer_maxpool2d():
    model = models.Conv2dAndMaxPool2DModule()
    model.eval()

    example_input = (torch.ones(1, 8, 32, 32))
    quantizer = NeutronQuantizer()
    exir_program = torch._export.capture_pre_autograd_graph(model, example_input)

    m = prepare_pt2e(exir_program, quantizer)
    m(*example_input)
    m = convert_pt2e(m)
    # print(f"Quantized model: {m}")
    nodes = list(m.graph.nodes)

    assert len(nodes) == 14
    # Check if QDQ pattern:
    assert nodes[10].name == "max_pool2d"
    assert _get_target_name(nodes[10].args[0]) == 'torch.ops.quantized_decomposed.dequantize_per_tensor.default'
    assert _get_target_name(nodes[11]) == 'torch.ops.quantized_decomposed.quantize_per_tensor.default'
    assert nodes[11].args[0].name == "max_pool2d"

    # Check if input and output quantization is same
    input_quant = nodes[10].args[0].args[1:]
    output_quant = nodes[11].args[1:]
    assert input_quant == output_quant

def test_quantizer_softmax():
    model = models.SoftmaxModule(dim=0)
    model.eval()

    example_input = (torch.ones(1, 10))
    quantizer = NeutronQuantizer()
    exir_program = torch._export.capture_pre_autograd_graph(model, example_input)

    m = prepare_pt2e(exir_program, quantizer)
    m(*example_input)
    m = convert_pt2e(m)
    # print(f"Quantized model: {m}")
    nodes = list(m.graph.nodes)

    assert len(nodes) == 7
    # Check if QDQ pattern:
    assert nodes[3].name == "softmax"
    assert _get_target_name(nodes[3].args[0]) == 'torch.ops.quantized_decomposed.dequantize_per_tensor.default'
    assert _get_target_name(nodes[4]) == 'torch.ops.quantized_decomposed.quantize_per_tensor.default'
    assert nodes[4].args[0].name == "softmax"

    # Check output quantization
    scale, zp, _, _, dtype  = nodes[4].args[1:]
    assert scale == 1.0/256.0
    assert zp == -128
    assert dtype == torch.int8

def test_quantizer_single_maxpool2d():
    model = models.MaxPool2dModule()
    model.eval()

    example_input = (torch.ones(1, 4, 32, 32))
    quantizer = NeutronQuantizer()
    exir_program = torch._export.capture_pre_autograd_graph(model, example_input)

    m = prepare_pt2e(exir_program, quantizer)
    m(*example_input)
    m = convert_pt2e(m)
    # print(f"Quantized model: {m}")
    nodes = list(m.graph.nodes)

    assert len(nodes) == 3
    assert nodes[1].name == "max_pool2d"
    assert "quantization_annotation" not in nodes[1].meta
