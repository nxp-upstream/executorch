# PyTorch Model Conversion to ExecuTorch Format

In this guideline we will show how to use the ExecuTorch AoT part to convert a PyTorch model to ExecuTorch format and delegate the model computation to eIQ Neutron NPU using the eIQ Neutron Backend.

First we will start with an example script converting the model. This example show the CifarNet model preparation. It is the same model which is part of the `example_cifarnet`

The steps are expected to be executed from the executorch root folder, in our case the `mcuxsdk-middleware-executorch`
1. After building the ExecuTorch you shall have the `libquantized_ops_aot_lib.so` located in the `pip-out` folder. We will need this library when generating the quantized cifarnet ExecuTorch model. So as first step we will find it:
```commandline
$ find ./pip-out -name `libquantized_ops_aot_lib.so`
./pip-out/temp.linux-x86_64-cpython-310/cmake-out/kernels/quantized/libquantized_ops_aot_lib.so
./pip-out/lib.linux-x86_64-cpython-310/executorch/kernels/quantized/libquantized_ops_aot_lib.so
```

2. Configure python to add the current directory to PYTHONPATH and copy the generated `program.fbs` and `scalar_type.fbs` to 
run the example. The code snipet bellow assumes you are using a virtual environment. Adjust according to your true setup. 
```commandline
$ export PYTHONPATH=`cd ..; pwd` 
$ cp  venv/lib/python3.10/site-packages/executorch/exir/_serialize/program.fbs exir/_serialize/program.fbs
$ cp  venv/lib/python3.10/site-packages/executorch/exir/_serialize/scalar_type.fbs  exir/_serialize/scalar_type.fbs
```

3. Now run the `aot_neutron_compile.py` example with the `cifar10` model 
```commandline
$ python examples/nxp/aot_neutron_compile.py \
    --quantize --so_library ./pip-out/lib.linux-x86_64-cpython-310/executorch/kernels/quantized/libquantized_ops_aot_lib.so \
    --delegate --neutron_converter_flavor SDK_25_03 -m cifar10
```

3. It will generate you `cifar10_nxp_delegate.pte` file which can be used with the MXUXpresso SDK `cifarnet_example` project. 

The generated PTE file is used in the executorch_cifarnet example application, see [example_application](example_applications.md).
 
