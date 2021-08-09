<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

This is a fork of TVM for adding BYOC integrations for the 3LA project.

Right now we have a VTA integration in `src/relay/backend/contrib/vta_matmul`. Note that you have to include the line `SET(USE_VTA_MATMUL ON)` in `build/config.cmake` before building TVM to support this (other flags that should be on: `USE_LLVM`, `USE_VTA_FSIM`). We have a test of this backend in `tests/python/relay/test_external_codegen.py` (see `test_extern_vta()`).

This version also uses a fork of the VTA repo meant to dump logs.
Try `vta/python/integration/matmul_tutorial.py` to use the dumping facility.
VTA can be set into dumping mode by calling `vta.testing.simulator.dump_mode(True)`.
You can specify the location at which the dump will be deposited using `vta.testing.simulator.dump_target(path)`; the default is `./vta_sim_dump.json`.
See the readme at [the VTA fork](https://github.com/uwsampl/3la-vta) to see a description of the dumping mode and the dumping format.

You can use `vta.testing.ila_converter.convert(dump_file, dest_file)` to convert a VTA simulator dump into an ILA program fragment.

# 3LA environment setup

## Docker setup
Please follow the instruction [here](https://github.com/PrincetonUniversity/3la-integrate) to set up the 3LA integrated docker container.

To attach to the container, run `sudo docker exec -it <name of the container> /bin/bash`

Before running any 3LA related test, `source init.sh` under `/root` first.

## 3LA tvm setup
Please follow the steps [here](https://tvm.apache.org/docs/install/from_source.html#developers-get-source-from-github). Note to replace the github repo link to this repo. Then switch to `conv1d-codegen` or `3la-rebase-complete` branch.

Before running `cmake`, please add the following lines to `config.cmake` 
```cmake
set(USE_ILAVTA_CODEGEN ON)
set(USE_ILACNN_CODEGEN ON)
set(USE_ILAFLEX_CODEGEN ON)
```
and then set `USE_LLVM` to `ON`.

Before installing the python interface of this variant of tvm, you probably need to uninstall the tvm  that was installed when building the docker image (to do so, run `pip uninstall <package>`).

<img src=https://raw.githubusercontent.com/apache/incubator-tvm-site/main/images/logo/tvm-logo-small.png width=128/> Open Deep Learning Compiler Stack
[Documentation](https://tvm.apache.org/docs) |
[Contributors](CONTRIBUTORS.md) |
[Community](https://tvm.apache.org/community) |
[Release Notes](NEWS.md)

[![Build Status](https://ci.tlcpack.ai/buildStatus/icon?job=tvm/main)](https://ci.tlcpack.ai/job/tvm/job/main/)
[![WinMacBuild](https://github.com/apache/tvm/workflows/WinMacBuild/badge.svg)](https://github.com/apache/tvm/actions?query=workflow%3AWinMacBuild)

Apache TVM is a compiler stack for deep learning systems. It is designed to close the gap between the
productivity-focused deep learning frameworks, and the performance- and efficiency-focused hardware backends.
TVM works with deep learning frameworks to provide end to end compilation to different backends.

License
-------
© Contributors Licensed under an [Apache-2.0](LICENSE) license.

Contribute to TVM
-----------------
TVM adopts apache committer model, we aim to create an open source project that is maintained and owned by the community.
Check out the [Contributor Guide](https://tvm.apache.org/docs/contribute/).

Acknowledgement
---------------
We learned a lot from the following projects when building TVM.
- [Halide](https://github.com/halide/Halide): Part of TVM's TIR and arithmetic simplification module
  originates from Halide. We also learned and adapted some part of lowering pipeline from Halide.
- [Loopy](https://github.com/inducer/loopy): use of integer set analysis and its loop transformation primitives.
- [Theano](https://github.com/Theano/Theano): the design inspiration of symbolic scan operator for recurrence.
