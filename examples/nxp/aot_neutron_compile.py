# Copyright 2024-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script to compile the model for the NXP Neutron NPU

import argparse
import io
import logging
from collections import defaultdict
from typing import Iterator

import torch
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e

from executorch.backends.nxp.neutron_partitioner import NeutronPartitioner
from executorch.backends.nxp.nxp_backend import generate_neutron_compile_spec
from executorch.backends.nxp.pytorch_passes.nxp_pytorch_pass_manager import NXPPyTorchPassManager
from executorch.backends.nxp.quantizer.neutron_quantizer import NeutronQuantizer
from executorch.examples.models import MODEL_NAME_TO_MODEL
from executorch.examples.models.model_factory import EagerModelFactory
from executorch.examples.nxp.cifar_net.cifar_net import CifarNet
from executorch.examples.nxp.cifar_net.cifar_net import test_cifarnet_model
from executorch.examples.nxp.models.mlperf_tiny import (AnomalyDetection, KeywordSpotting, ImageClassification,
                                                        VisualWakeWords)
from executorch.exir import ExecutorchBackendConfig
from executorch.extension.export_util import export_to_edge, save_pte_program

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def print_ops_in_edge_program(edge_program):
    """ Find all ops used in the `edge_program` and print them out along with their occurrence counts. """

    ops_and_counts = defaultdict(lambda: 0)  # Mapping ops to the numer of times they are used.
    for node in edge_program.graph.nodes:
        if 'call' not in node.op:
            continue  # `placeholder` or `output`. (not an operator)

        if hasattr(node.target, '_schema'):
            # Regular op.
            # noinspection PyProtectedMember
            op = node.target._schema.schema.name
        else:
            # Builtin function.
            op = str(node.target)

        ops_and_counts[op] += 1

    # Sort the ops based on how many times they are used in the model.
    ops_and_counts = sorted(ops_and_counts.items(), key=lambda x: x[1], reverse=True)

    # Print the ops and use counts.
    for op, count in ops_and_counts:
        print(f'{op: <50} {count}x')


def get_model_and_inputs_from_name(model_name: str):
    """Given the name of an example pytorch model, return it, example inputs and calibration inputs (can be None)

    Raises RuntimeError if there is no example model corresponding to the given name.
    """
    calibration_inputs = None  # TBD
    # Case 1: Model is defined in this file
    if model_name in models.keys():
        m = models[model_name]()
        model = m.get_eager_model()
        example_inputs = m.get_example_inputs()
        calibration_inputs = m.get_calibration_inputs(64)
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
    "visual_wake_words": VisualWakeWords,
    "keyword_spotting": KeywordSpotting,
    "image_classification": ImageClassification,
    "anomaly_detection": AnomalyDetection,
}


def post_training_quantize(model, calibration_inputs: tuple[torch.Tensor] | Iterator[tuple[torch.Tensor]]):
    """ Quantize the provided model.

    :param model: Aten model to quantize.
    :param calibration_inputs: Either a tuple of calibration input tensors where each element corresponds to a model
                                input. Or an iterator over such tuples.
    """
    # Based on executorch.examples.arm.aot_amr_compiler.quantize
    logging.info("Quantizing model")
    logging.debug(f"Original model: {model}")
    quantizer = NeutronQuantizer()

    m = prepare_pt2e(model, quantizer)
    # Calibration:
    logging.debug(f"Calibrating model")
    if not isinstance(calibration_inputs, tuple):  # TODO(Robert): Assumption that calibration_inputs is finite.
        get_batch_size = lambda data: data[0].shape[0]
        for i, data in enumerate(calibration_inputs):
            if i % (1000 // get_batch_size(data)) == 0:
                logging.debug(f"{i * get_batch_size(data)} inputs done")
            m(*data)
    else:
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
        help=f"Provide model name. Valid ones: {set(list(models.keys()) + list(MODEL_NAME_TO_MODEL.keys()))}",
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
        "--target",
        required=False,
        default="imxrt700",
        help="Platform for running the delegated model",
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
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        required=False,
        default=False,
        help="Test the selected model and print the accuracy between 0 and 1.",
    )
    parser.add_argument(
        "--operators_not_to_delegate",
        required=False,
        default=[],
        type=str,
        nargs='*',
        help="List of operators not to delegate. E.g., --operators_not_to_delegate aten::convolution aten::mm"
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

    # 3. Run pre-processing passes of the float32 aten dialect program.
    pass_manager = NXPPyTorchPassManager(exir_program_aten)
    pass_manager.run()  # All passes by default.

    # 4. Quantize if required
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

    if args.test:
        match args.model_name:
            case 'cifar10':
                accuracy = test_cifarnet_model(exir_program_aten)

            case _:
                raise NotImplementedError(f'Testing of model `{args.model_name}` is not yet supported.')

        cyan, end_format = '\033[96m', '\033[0m'
        quantized_str = 'quantized ' if args.quantize else ''
        print(f'\n{cyan}Accuracy of the {quantized_str}`{args.model_name}`: {accuracy}{end_format}\n')

    # 5. Export to edge program
    # edge_program = exir.to_edge(exir_program_aten)
    edge_program = export_to_edge(exir_program_aten, example_inputs,
                                  verbose=args.debug)  # TODO for now reusing the default for edge_compile_config
    # (compared to Arm)
    logging.debug(f"Exported graph:\n{edge_program.exported_program().graph}")

    # 6. Delegate to Neutron
    if args.delegate is True:
        logging.info("Executing Neutron Partitioner and Delegate")
        edge_program = edge_program.to_backend(
            NeutronPartitioner(
                generate_neutron_compile_spec(
                    args.target,
                    operators_not_to_delegate=args.operators_not_to_delegate
>>>>>>> 295ca4dfb28cf93049d03defcd2d84058772e3d4
                )
            )
        )
        logging.debug(f"Lowered graph:\n{edge_program.exported_program().graph}")

    # 7. Export to ExecuTorch program
    try:
        exec_prog = edge_program.to_executorch(
            config=ExecutorchBackendConfig(extract_delegate_segments=False)
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
        "_nxp_delegate" if args.delegate is True else ""
    )
    save_pte_program(exec_prog, model_name)
