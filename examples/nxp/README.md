# ExecuTorch NXP Backend examples

This directory contains models and scripts to run a Neutron delegated PyTorch model with ExecuTorch.

## Directory structure

```bash
nxp
├── cifar_net                       # CifarNet10 model.
├── models                          # Models directory.
│   ├── mlperf_tiny                 # MLPerf Tiny model directory.
│   ├── model_manager.py            # Model manager for model and dataset handling.
│   └── utils.py                    # Model manager utilities.
├── aot_neutron_compile.py          # Script shows end-to-end workflow for compiling and running on Neutron.
└── README.md                       # This file.
```

## Examples

Here is described a workflow how to generate a Neutron-delegated `.pte` ExecuTorch model from an example model 
in `torch.nn.Module` format. Resulting ExecuTorch model is ready to deploy on NXP Neutron enabled devices.
We support several example models. To run these examples, we will use `aot_neutron_compile.py`.

#### Steps to run:

1. Build `quantized_ops_aot_lib` package

    To run quantization we need to pass shared library `libquantized_ops_aot_lib.so`. First, we need to build
    `quantized_ops_aot_lib` package. To build it, we need to set the CMake arguments:
    `-DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON` and `-DEXECUTORCH_BUILD_KERNELS_QUANTIZED_AOT=ON`.

2. Prepare calibration data (for MLPerf Tiny models)

    Clone [MLPerf Tiny](https://github.com/mlcommons/tiny) repository and follow 
    `benchmark/experimental/training_torch/README.md` to prepare calibration data for selected model. Copy created 
    calibration data file for the model to `executorch/data/calibration_data/<model_name_dir>`.

3. Prepare `PYTHONPATH` environment variable

    You need to export the `PYTHONPATH` to point to the root of the "executorch" directory:
    ```bash
    $ cd executorch
    $ export PYTHONPATH=` cd ..; pwd`
    $ echo $PYTHONPATH
    ```

4. Run `aot_neutron_compile.py` script

    Run the script with following arguments: `--model_name` - specify model from a list of available models, `--so_library` 
    - specify path to `libquantized_ops_aot_lib.so` library from step 1. Quantization and delegation are controlled by
    `--quantize` and `--delegate` flags, turned off by default.
    
    Supported models:
    - cifar10
    - visual_wake_words
    - keyword_spotting
    - image_classification 
    - anomaly_detection
   
    ```bash
    $ python -m examples.nxp.aot_neutron_compile --model_name <model_name>  path/to/libquantized_ops_aot_lib.so --quantize --quantize
    ```

5. Run the model

    ```bash
    $ <path-to-executorch-build>/executor_runner -model_path examples/nxp/<model_name>.pte
    ```
