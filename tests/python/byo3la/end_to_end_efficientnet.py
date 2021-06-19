"""
Clones in an MxNet EfficientNet implementation, imports to TVM,
and runs via ILACNN codegen
"""
import os
import subprocess

import timeit

import numpy as np

import tvm
from tvm import relay
from tvm import runtime
from tvm.relay.op.contrib import ilacnn

from EfficientNet.efficientnet_model import get_efficientnet
import mxnet

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ENET_DIR = os.path.join(TEST_DIR, "EfficientNet")
PARAMS_FILE = os.path.join(ENET_DIR, "0.3358-imagenet-efficientnet-b0-47-best.params")

def data_present():
    return os.path.exists(PARAMS_FILE)


def get_data():
    subprocess.run(["wget", "https://www.dropbox.com/s/l2ehu85vmmj3w5w/0.3358-imagenet-efficientnet-b0-47-best.params"], cwd=ENET_DIR)

def cal_error(result, ref):
    diff = result - ref
    abs_diff = np.abs(diff)
    mean_diff = np.sum(abs_diff) / (diff.size)
    # print(result.size, ref.size)
    return mean_diff/np.mean(np.abs(result)), mean_diff/np.mean(np.abs(ref))

def main():
    if not data_present():
        get_data()

    # enet, _ = get_efficientnet("efficientnet-b0")
    # enet.load_parameters(PARAMS_FILE)
    # mod, params = relay.frontend.from_mxnet(enet, {"data": (1, 3, 32, 32)})
    # net = mxnet.gluon.model_zoo.vision.squeezenet1_1(pretrained=True)
    net = mxnet.gluon.model_zoo.vision.get_resnet(2, 34, pretrained=True)
    mod, params = relay.frontend.from_mxnet(net, {"data": (1, 34, 34, 34)})
    
    mod_wo_acc = mod

    params["data"] = tvm.nd.array(np.random.rand(1, 3, 32, 32).astype("float32"))
    args = [params[var.name_hint] for var in mod["main"].params]
    mod["main"] = ilacnn.remove_padding(mod["main"])

    args_copy = args

    pattern_table = ilacnn.pattern_table()
    mod = tvm.relay.transform.MergeComposite(pattern_table)(mod)
    mod = tvm.relay.transform.AnnotateTarget(["ilacnn"])(mod)
    mod = tvm.relay.transform.PartitionGraph()(mod)

    with tvm.transform.PassContext(opt_level=3):
        device = tvm.cpu()
        target = "llvm"
        exe = relay.vm.compile(mod_wo_acc, target)
        vm = runtime.vm.VirtualMachine(exe, device)
        start = timeit.default_timer()
        ret_ref = vm.invoke("main", *args_copy)
        end = timeit.default_timer()
        print("host_wo_acc runtime runs {:05f} seconds".format(end - start))
        ref_out = ret_ref.asnumpy()

    with tvm.transform.PassContext(opt_level=3):
        device = tvm.cpu()
        target = "llvm"
        exe = relay.vm.compile(mod, target)
        vm = runtime.vm.VirtualMachine(exe, device)

        start = timeit.default_timer()
        ret = vm.invoke("main", *args)
        end = timeit.default_timer()

        ila_out = ret.asnumpy()

        print("ila runtime runs {:05f} seconds".format(end - start))

    err_out, err_ref = cal_error(ila_out, ref_out)
    print("result analysis --- relative error (vs. sim_out): {:5.5%}\
            relative error (vs. ref): {:5.5%}\n".format(err_out, err_ref))

if __name__ == "__main__":
    main()
