"""
Clones in an MxNet EfficientNet implementation, imports to TVM,
and runs via ILACNN codegen
"""
import os
import subprocess

import numpy as np

import tvm
from tvm import relay
from tvm import runtime
from tvm.relay.op.contrib import ilacnn

from EfficientNet.efficientnet_model import get_efficientnet

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ENET_DIR = os.path.join(TEST_DIR, "EfficientNet")
PARAMS_FILE = os.path.join(ENET_DIR, "0.3358-imagenet-efficientnet-b0-47-best.params")

def data_present():
    return os.path.exists(PARAMS_FILE)


def get_data():
    subprocess.run(["wget", "https://www.dropbox.com/s/l2ehu85vmmj3w5w/0.3358-imagenet-efficientnet-b0-47-best.params"], cwd=ENET_DIR)


def main():
    if not data_present():
        get_data()

    enet, _ = get_efficientnet("efficientnet-b0")
    enet.load_parameters(PARAMS_FILE)
    mod, params = relay.frontend.from_mxnet(enet, {"data": (1, 3, 224, 224)})

    params["data"] = tvm.nd.array(np.random.rand(1, 3, 224, 224).astype("float32"))
    args = [params[var.name_hint] for var in mod["main"].params]
    mod["main"] = ilacnn.remove_padding(mod["main"])

    pattern_table = ilacnn.pattern_table()
    mod = tvm.relay.transform.MergeComposite(pattern_table)(mod)
    mod = tvm.relay.transform.AnnotateTarget(["ilacnn"])(mod)
    mod = tvm.relay.transform.PartitionGraph()(mod)

    with tvm.transform.PassContext(opt_level=3):
        device = tvm.cpu()
        target = "llvm"
        exe = relay.vm.compile(mod, target)
        vm = runtime.vm.VirtualMachine(exe, device)

    ret = vm.invoke("main", *args)


if __name__ == "__main__":
    main()
