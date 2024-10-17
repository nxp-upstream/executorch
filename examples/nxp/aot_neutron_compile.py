# Copyright (c) 2024 NXP
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script to compile the model for the NXP Neutron NPU

import argparse
import logging
import io

import torch
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.quantization.utils import is_per_channel

from executorch.backends.arm.quantizer.arm_quantizer import get_symmetric_quantization_config
from executorch.backends.nxp.quantizer.neutron_quantizer import NeutronQuantizer
from executorch.examples.models import MODEL_NAME_TO_MODEL
from executorch.examples.models.model_factory import EagerModelFactory
from executorch.examples.portable import export_to_edge, save_pte_program
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig
import executorch.exir as exir
from executorch.examples.nxp.cifar_net.cifar_net import CifarNet

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

def get_model_and_inputs_from_name(model_name: str):
    """Given the name of an example pytorch model, return it, example inputs and calibration inputs (can be None)

    Raises RuntimeError if there is no example model corresponding to the given name.
    """
    calibration_inputs = None #TBD
    # Case 1: Model is defined in this file
    if model_name in models.keys():
        m = models[model_name]()
        model = m.get_eager_model()
        example_inputs = (next(m.get_example_inputs())[0],) #TODO (Robert): Needs to redesign the CifarNet example.
    # Case 2: Model is defined in executorch/examples/models/
    elif model_name in MODEL_NAME_TO_MODEL.keys():
        logging.warning(
            "Using a model from examples/models not all of these are currently supported"
        )
        model, example_inputs, _ = EagerModelFactory.create_model(*MODEL_NAME_TO_MODEL[model_name])
    else:
        raise RuntimeError(
            f"Model '{model_name}' is not a valid name. Use --help for a list of available models."
        )

    return model, example_inputs, calibration_inputs

models = {
    "cifar10": CifarNet,
}

def post_training_quantize(model, calibration_inputs):
    # Based on executorch.examples.arm.aot_amr_compiler.quantize
    logging.info("Quantizing model")
    logging.debug(f"Original model: {model}")
    quantizer = NeutronQuantizer()
    operator_config = get_symmetric_quantization_config(is_per_channel=False)
    quantizer.set_global(operator_config)
    m = prepare_pt2e(model, quantizer)
    # Calibration:
    m(*calibration_inputs)
    m = convert_pt2e(m)
    logging.debug(f"Quantized model: {m}")
    return m



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help=f"Provide model name. Valid ones: {set(list(models.keys())+list(MODEL_NAME_TO_MODEL.keys()))}",
    )
    parser.add_argument(
        "-d",
        "--delegate",
        action="store_true",
        required=False,
        default=False,
        help="Flag for producing ArmBackend delegated model",
    )
    parser.add_argument(
        "-q",
        "--quantize",
        action="store_true",
        required=False,
        default=False,
        help="Produce a quantized model",
    )
    parser.add_argument(
        "-s",
        "--so_library",
        required=False,
        default=None,
        help="Provide path to so library. E.g., cmake-out/kernels/quantized/libquantized_ops_aot_lib.so. "
             "To build it update the CMake arguments: -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON"
             " -DEXECUTORCH_BUILD_KERNELS_QUANTIZED_AOT=ON and build the quantized_ops_aot_lib package",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Set the logging level to debug."
    )

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format=FORMAT, force=True)


    # 1. pick model from one of the supported lists
    model, example_inputs, calibration_inputs = get_model_and_inputs_from_name(args.model_name)
    model = model.eval()

    # 2. Export the model to ATEN
    # pre-autograd export is used in the quantization example here
    # https://pytorch.org/docs/stable/quantization.html#prototype-pytorch-2-export-quantization.
    # Eventually this will become torch.export.export(), but the quantization implementation does not support
    # it yet.
    exir_program_aten = torch._export.capture_pre_autograd_graph(model, example_inputs)

    # 3. Quantize if required
    if args.quantize:
        if args.quantize and not args.so_library:
            logging.warning(
              "Quantization enabled without supplying path to libcustom_ops_aot_lib using -s flag."
                + "This is required for running quantized models with unquantized input."
            )
        if calibration_inputs is None:
            logging.warning(f"No calibration inputs available, using the example inputs instead")
            calibration_inputs = example_inputs
        exir_program_aten = post_training_quantize(exir_program_aten, calibration_inputs)

        # For quantization we need to build the quantized_ops_aot_lib.so. To build it update the CMake arguments:
        # -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON
        # -DEXECUTORCH_BUILD_KERNELS_QUANTIZED_AOT=ON
        # and build the quantized_ops_aot_lib package
        # Then run with --so_library <path_to_the_library>
        if args.so_library is not None:
            logging.debug(f"Loading custom operator library {args.so_library}")
            torch.ops.load_library(args.so_library)

    # 4. Export to edge program
    # edge_program = exir.to_edge(exir_program_aten)
    edge_program = export_to_edge(exir_program_aten, example_inputs) # TODO for now reusing the default for edge_compile_config
                                                         # (compared to Arm)
    logging.debug(f"Exported graph:\n{edge_program.exported_program().graph}")

    # 5. Delegate to Neutron
    if args.delegate is True:
        logging.warning("Neutron Delegate is not available yet")

    # 6. Export to ExecuTorch program
    try:
        exec_prog = edge_program.to_executorch(
            config=ExecutorchBackendConfig(
                extract_delegate_segments=False, extract_constant_segment=False
            )
        )
    except RuntimeError as e:
        if "Missing out variants" in str(e.args[0]):
            raise RuntimeError(
                e.args[0]
                + ".\nThis likely due to an external so library not being loaded. Supply a path to it with the -s flag."
            ).with_traceback(e.__traceback__) from None
        else:
            raise e

    def executorch_program_to_str(ep, verbose=False):
        f = io.StringIO()
        ep.dump_executorch_program(out=f, verbose=verbose)
        return f.getvalue()

    logging.debug(f"Executorch program (short):\n{executorch_program_to_str(exec_prog)}")
    logging.debug(f"Executorch program (complete):\n{executorch_program_to_str(exec_prog, verbose=True)}")

    # 7. Serialize to *.pte
    model_name = f"{args.model_name}" + (
        "_arm_delegate" if args.delegate is True else ""
    )
    save_pte_program(exec_prog, model_name)