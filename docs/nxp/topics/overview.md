# Overview

ExecuTorch is an end-to-end solution for enabling on-device inference capabilities across mobile and edge devices including wearables, embedded devices and microcontrollers. It is part of the PyTorch Edge ecosystem and enables efficient deployment of PyTorch models to edge devices. For more information, see https://pytorch.org/executorch-overview.

The MCUXpresso Software Development Kit \(MCUXpresso SDK\) provides a comprehensive software package with a pre-integrated ExecuTorch based on version v0.5.0 with initial support for Neutron Backend. Neutron Backend enables acceleration of ML models on the [eIQ® Neutron Neural Processing Unit (NPU)](https://www.nxp.com/applications/technologies/ai-and-machine-learning/eiq-neutron-npu:EIQ-NEUTRON-NPU).

This document describes the steps required to download and start using the ExecuTorch. Additionally, the document describes the steps required to create an application for running pre-trained models.

**Note:** The document also assumes knowledge of machine learning frameworks for model training.

## Supported platforms: 
* [i.MX RT700](https://www.nxp.com/products/i.MX-RT700)

## Installation
The ExecuTorch, with the Neutron Backend consists of:
* ExecuTorch with Neutron Backend for Ahead of Time ML Model Compilation
* Neutron Converter
* MCUXpresso SDK

Here we briefly describe each components purpose and steps to install them.

The **ExecuTorch AoT** and **Neutron Converter** are needed to convert a PyTorch model to ExecuTorch and Delegate it to eIQ Neutron NPU using the Neutron Backend. 
The **MCUXpresso SDK** provides project to build the ExecuTorch Runtime Library, the example application with simple CNN, toolchains and other middleware libraries to build and deploy the application on the target platform.

If you want run to prepared example application on the i.MX RT700 platform, and skip the model preparation phase continue with the [MCUXpresso SDK Part](#mcuxpresso-sdk-part). 

### ExecuTorch for Ahead of Time model preparation
The ExecuTorch enables to deploy PyTorch models on edge devices. For this purpose the PyTorch model must be processed and converter by the ExecuTorch Ahead of Time (AoT) part. You can obtain the full ExecuTorch including the AoT part aligned with this version of MCUX SDK from the [mcuxsdk-middleware-executorch](https://github.com/nxp-mcuxpresso/mcuxsdk-middleware-executorch/tree/release/mcux-full) release/mcux-full branch.

#### Installation
Prerequisities:
* x86 Linux Machine with GLIBC-2.29 or higher (e.g. Ubuntu 20.04 or higher)
* Python 3.10 or 3.11

To build and install the ExecuTorch follow these steps:

1. (Optional) Setup python virtual environment on desired location and activate it.
```commandline
$ python3 -m venv venv
$ source venv/bin/activate
``` 

2. Clone the ExecuTorch from [mcuxsdk-middleware-executorch](https://github.com/nxp-mcuxpresso/mcuxsdk-middleware-executorch/tree/release/mcux-full)
```commandline
$ git clone --branch release/mcux-full https://github.com/nxp-mcuxpresso/mcuxsdk-middleware-executorch.git
$ cd mcuxsdk-middleware-executorch
$ git submodule update --init --recursive
```

3. Build and install the ExecuTorch and its dependencies:
```commandline
$ ./install_requirements.sh
```
> [!WARNING]
> The `install_requirements.sh` installs the CPU version of `torch` from `https://download.pytorch.org/whl/cpu`. If you are behind corporate proxy, it might have issues accessing it and you will see warnings like: 
> ```commandline
>  WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1006)'))': /whl/test/cpu/torch/
>  ```
> In this case the CUDA version of torch is installed and the `install_requirements.sh` script fails with: 
> ```commandline
> PyTorch: CUDA cannot be found.  Depending on whether you are building
> ```
> Make sure the pip can access the `https://download.pytorch.org/whl/cpu` PyPI.

Next continue with installation of the [Neutron Converter](#neutron-converter)

### Neutron Converter
The eIQ Neutron Backend uses the Neutron Converter to convert the ExecuTorch program to the eIQ Neutron NPU microcode.

#### Installation
The Neutron Converter is available as a Python package and can be installed by the `pip` command from eiq.nxp.com/repository:
```commandline
pip install --extra-index-url eiq.nxp.com/repository neutron_converter_sdk_25_03
```
The Neutron Converter is used internally by the ExecuTorch, and it is tight with the particular BSP you are using - the suffix of the python package name. In the code snippet above the flavor is the `SDK_25_03`.
In the [aot_neutron_convert.py](../../../examples/nxp/aot_neutron_compile.py) example script by the `--neutron_converter_flavor` parameter. 

### MCUXpresso SDK
The MCUXpresso SDK is used to build, debug and deploy the application using the ExecuTorch on the target platform.

You can obtain the MCUXpresso SDK from [MCUXpresso SDK Builder](https://mcuxpresso.nxp.com/en) including the IDE. 
See the [getting_mcuxpress](getting_mcuxpresso.md) for details. 

In the MCUXpresso SDK, there are 2 project available related to ExecuTorch: 
* executorch_lib
* executorch_cifarnet

For more details see [example_applications](example_applications.md). Here you will find the details to run build and run the demo applications.







