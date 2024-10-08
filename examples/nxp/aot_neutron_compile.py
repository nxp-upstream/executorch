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

from executorch.examples.models import MODEL_NAME_TO_MODEL
from executorch.examples.models.model_factory import EagerModelFactory
from executorch.examples.portable.utils import save_pte_program
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig
import executorch.exir as exir

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

def get_model_and_inputs_from_name(model_name: str):
    """Given the name of an example pytorch model, return it and example inputs.

    Raises RuntimeError if there is no example model corresponding to the given name.
    """
    # Case 1: Model is defined in this file
    if model_name in models.keys():
        model = models[model_name]()
        example_inputs = models[model_name].example_input
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

    return model, example_inputs

models = {
    "cifar10": None, #TBD
}

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
        "--debug", action="store_true", help="Set the logging level to debug."
    )

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format=FORMAT, force=True)


    # 1. pick model from one of the supported lists
    model, example_inputs = get_model_and_inputs_from_name(args.model_name)
    model = model.eval()

    # 2. Export the model
    exir_program_aten = torch.export.export(model, example_inputs)
    # pre-autograd export is used in the quantization example here
    # https://pytorch.org/docs/stable/quantization.html#prototype-pytorch-2-export-quantization
    # The torch.export suitability for quantization is not clear, hence if problems use the bellow implementation:
    # model = torch._export.capture_pre_autograd_graph(model, example_inputs)

    # 3. Quantize if required
    if args.quantize:
        logging.warning(f"Neutron quantizer is not yet implemented")

    # 4. Export to edge program
    edge = exir.to_edge(exir_program_aten)
    logging.debug(f"Exported graph:\n{edge.exported_program().graph}")

    # 5. Delegate to Neutron
    if args.delegate is True:
        logging.warning("Neutron Delegate is not available yet")

    # 6. Export to ExecuTorch program
    try:
        exec_prog = edge.to_executorch(
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