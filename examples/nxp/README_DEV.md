Disclaimer: 
- this file contains temporal information only for development purpose. Shall not be upstreamed.

# Running the Example

## Create link to generated *fbs files
To run the aot_neutron_compile.py example e.g. from CLion you need to link the generated *.fbs file from the executorch
installation into the source code repository: 
```bash
# Check where the fbs files are, we need the executorch/exir/_serialize/program.fbs and scalar_type.bfs
# E.g. ~/REPOS_2/executorch/venv3.10/lib/python3.10/site-packages/executorch/exir/_serialize/program.fbs
$ cd <executorch source dir> 
$ find . -name "program.fbs"
$ ln -s <path_to_venv>/lib/python3.10/site-packages/executorch/exir/_serialize/program.fbs exir/_serialize/program.fbs
$ ln -s <path_to_venv>/lib/python3.10/site-packages/executorch/exir/_serialize/scalar_type.fbs exir/_serialize/scalar_type.fbs
```

## Run the aot_neutron_copiler.py
### From Command line
Not all the python files are available in the installed python package.
You need to export the PYTHONPATH to point to the root of the "executorch" directory:
```
$ export PYTHONPATH=` cd ..; pwd`
$ echo $PYTHONPATH
$ python -m examples.nxp.aot_neutron_compile
```
### CLion
Make sure the configuration includes the "Add Content roots to PYTHONPATH". 

## Run the model
To run the model use the executorch_runner binary, e.g. :
```bash
(venv3.10) nxf39574@SMW009643:~/REPOS_2/executorch$ cmake-build-debug/executor_runner -model_path examples/nxp/add.pte 
I 00:00:00.001961 executorch:executor_runner.cpp:73] Model file examples/nxp/add.pte is loaded.
I 00:00:00.002019 executorch:executor_runner.cpp:82] Using method forward
I 00:00:00.002026 executorch:executor_runner.cpp:129] Setting up planned buffer 0, size 48.
I 00:00:00.002060 executorch:executor_runner.cpp:152] Method loaded.
I 00:00:00.002069 executorch:executor_runner.cpp:162] Inputs prepared.
I 00:00:00.003499 executorch:executor_runner.cpp:171] Model executed successfully.
I 00:00:00.003550 executorch:executor_runner.cpp:175] 1 outputs: 
Output 0: tensor(sizes=[1], [2.])
```