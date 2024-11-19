# Copyright (c) 2024 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import struct

import numpy as np

from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import BuiltinOperator
from executorch.backends.nxp.backend.ir.lib.tflite.Model import Model
from executorch.exir.backend.backend_details import PreprocessResult


def extract_artifacts_from_neutron_node(tflite_flatbuffer_or_path: bytes | str) -> PreprocessResult:
    """ Extract the payload (microcode, weights, kernels) from the Neutron Node in the given TFLite model.
        The model can be provided as a binary flatbuffer, or a path to a `.tflite` model.

        The return format is a `PreprocessResult` object, and its `processed_bytes` attribute contains the serialized
         binary data of the following C struct:
         struct NeutronBinary {
            uint8[] microcode;
            uint8[] weights;
            uint8[] kernels;
        }

        The individual components must be aligned to 16 bytes.
    """

    if isinstance(tflite_flatbuffer_or_path, str):
        with open(tflite_flatbuffer_or_path, 'rb') as f:
            flatbuffer = f.read()
    else:
        flatbuffer = tflite_flatbuffer_or_path

    model = Model.GetRootAs(flatbuffer, 0)
    assert model.SubgraphsLength() == 1, f'The model has `{model.SubgraphsLength()}` SubGraphs instead of `1`.'

    sub_graph = model.Subgraphs(0)

    if sub_graph.OperatorsLength() != 1:
        logging.warning(f'Model has `{sub_graph.OperatorsLength()}` Operators instead of `1`.')

        # TODO Raise an exception in the future, because the graph should only contain the 1 node. Multiple nodes
        #  indicate an issue with the Partitioner.
        # raise RuntimeError(f'Model has `{sub_graph.OperatorsLength()}` Operators instead of `1`.')

    neutron_node = None
    opcodes = [model.OperatorCodes(i) for i in range(model.OperatorCodesLength())]
    for i in range(sub_graph.OperatorsLength()):
        opcode = opcodes[sub_graph.Operators(i).OpcodeIndex()]
        if opcode.BuiltinCode() == BuiltinOperator.CUSTOM and opcode.CustomCode() == b'NeutronGraph':
            # Found the NeutronNode.
            neutron_node = sub_graph.Operators(i)
            break

    assert neutron_node is not None, 'The provided model does not contain a Neutron Node.'

    # The last 3 input tensors of the Neutron Node contain:
    #   1. Neutron Microcode
    #   2. Neutron Weights
    #   3. Neutron Kernels
    assert neutron_node.InputsLength() >= 3, \
        f'The Neutron Node only has `{neutron_node.GetInputsLen()}` inputs. Expected at least `3`.'
    microcode_idx, weights_idx, kernels_idx = neutron_node.InputsAsNumpy()[-3:]

    microcode_buffer_idx = sub_graph.Tensors(microcode_idx).Buffer()
    weights_buffer_idx = sub_graph.Tensors(weights_idx).Buffer()
    kernels_buffer_idx = sub_graph.Tensors(kernels_idx).Buffer()

    microcode = model.Buffers(microcode_buffer_idx).DataAsNumpy()
    weights = model.Buffers(weights_buffer_idx).DataAsNumpy()
    kernels = model.Buffers(kernels_buffer_idx).DataAsNumpy()

    assert microcode.dtype == weights.dtype == kernels.dtype == np.dtype('uint8'), \
        'The Neutron Node uses unexpected data types.'

    # Align to 16B (according to commit 008bdc17670).
    alignment = 16

    def padding_format_string_for_array(array: np.ndarray) -> str:
        """ Create a padding format string for the given array, which will add 0s at the end for correct alignment.
            E.g. the string '10x' represents adding 10 bytes of '0' padding.
        """
        assert array.dtype == np.dtype('uint8')

        overflow = array.size % alignment
        if overflow == 0:
            return ''

        # Overflow 1 means padding 15, so use `alignment - overflow` padding.
        return f'{alignment - overflow}x'

    def format_string_for_array(array: np.ndarray) -> str:
        """ Create a format string which will represent the provided array. It also handles the necessary alignment.
            E.g. for array [1,2,3] we get '3s13x', because '3s' means string of 3 bytes, and `13x` means adding 13 bytes
             of '0' padding at the end (for 16B alignment).
        """
        assert array.dtype == np.dtype('uint8')

        return f'{array.size}s{padding_format_string_for_array(array)}'

    # The resulting payload should be structured as a binary in the format defined in the function header.
    payload = struct.pack(
        format_string_for_array(microcode) + format_string_for_array(weights) + format_string_for_array(kernels),
        microcode.tobytes(), weights.tobytes(), kernels.tobytes()
    )

    return PreprocessResult(processed_bytes=payload)
