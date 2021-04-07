import numpy as np
import tvm
from tvm import relay
from tvm import runtime
from tvm.relay.op.contrib import ilacnn

import sys

def cal_error(result, ref):
    diff = result - ref
    abs_diff = np.abs(diff)
    mean_diff = np.sum(abs_diff) / (diff.size)
    # print(result.size, ref.size)
    return mean_diff/np.mean(np.abs(result)), mean_diff/np.mean(np.abs(ref))

# just some simple smoke tests
def test_conv2d_unpadded():
    in_chan = 3
    in_row = 30
    in_col = 30
    k_num = 3
    k_chan = in_chan
    k_row = 3
    k_col = 3
    x = relay.Var("x", type_annotation=relay.TensorType((1, in_chan, in_row, in_col)))
    y = relay.Var("y", type_annotation=relay.TensorType((k_num, k_chan, k_row, k_col)))
    conv_func = relay.Function([x, y], relay.nn.conv2d(x, y))
    mod = tvm.IRModule()
    mod["main"] = conv_func

    mod_wo_acc = tvm.IRModule()
    mod_wo_acc["main"] = conv_func

    pattern_table = ilacnn.pattern_table()
    mod = tvm.relay.transform.MergeComposite(pattern_table)(mod)
    mod = tvm.relay.transform.AnnotateTarget(["ilacnn"])(mod)
    mod = tvm.relay.transform.PartitionGraph()(mod)
    print(mod)

    inp = np.random.uniform(-1,1, (1, in_chan, in_row, in_col)).astype("float32")
    wgt = np.random.uniform(-1,1, (k_num, k_chan, k_row, k_col)).astype("float32")

    wgt = wgt.reshape((k_num, k_chan, k_row, k_col)).astype("float32")
    
    with open("./test/inputs.log", 'w') as fout:
        print('input array:\n{}\n'.format(inp), file=fout)
        print('wgt_array:\n{}\n'.format(wgt),file=fout)
    
    # without ilacnn backend
    with tvm.transform.PassContext():
        device = tvm.cpu()
        target = "llvm"
        exe = relay.vm.compile(mod_wo_acc, target)
        vm = runtime.vm.VirtualMachine(exe, device)

        args = [inp,wgt]
        ret = vm.invoke("main", *args)
        ref_out = ret.asnumpy()

    # use ilacnn backend
    with tvm.transform.PassContext(opt_level=3):
        device = tvm.cpu()
        target = "llvm"
        exe = relay.vm.compile(mod, target)
        vm = runtime.vm.VirtualMachine(exe, device)

        args = [inp,wgt]
        ret = vm.invoke("main", *args)
        ila_out = ret.asnumpy()

    err_out, err_ref = cal_error(ila_out, ref_out)
    print("result analysis --- relative error (vs. sim_out): {:5.5%}\
            relative error (vs. ref): {:5.5%}\n".format(err_out, err_ref))
    # print("reference output: \n{}".format(ref_out))
    # print("ila output: \n{}".format(ila_out))

def test_conv2d_padded():
    x = relay.Var("x", type_annotation=relay.TensorType((1, 3, 20, 18)))
    y = relay.Var("y", type_annotation=relay.TensorType((3, 3, 3, 3)))
    conv_func = relay.Function([x, y], relay.nn.conv2d(x, y, padding=(2, 3)))
    mod = tvm.IRModule()
    mod["main"] = ilacnn.remove_padding(conv_func)

    mod_wo_acc = tvm.IRModule()
    mod_wo_acc["main"] = conv_func

    pattern_table = ilacnn.pattern_table()
    mod = tvm.relay.transform.MergeComposite(pattern_table)(mod)
    mod = tvm.relay.transform.AnnotateTarget(["ilacnn"])(mod)
    mod = tvm.relay.transform.PartitionGraph()(mod)
    print(mod)

    inp = np.random.rand(1, 3, 20, 18).astype("float32")
    wgt = np.random.rand(3, 3, 3, 3).astype("float32")

    # without ilacnn backend
    with tvm.transform.PassContext():
        device = tvm.cpu()
        target = "llvm"
        exe = relay.vm.compile(mod_wo_acc, target)
        vm = runtime.vm.VirtualMachine(exe, device)

        args = [inp,wgt]
        ret = vm.invoke("main", *args)
        ref_out = ret.asnumpy()

    with tvm.transform.PassContext(opt_level=3):
        device = tvm.cpu()
        target = "llvm"
        exe = relay.vm.compile(mod, target)
        vm = runtime.vm.VirtualMachine(exe, device)

        args = [inp, wgt]
        ret = vm.invoke("main", *args)
        ila_out = ret.asnumpy()

    err_out, err_ref = cal_error(ila_out, ref_out)
    print("result analysis --- relative error (vs. sim_out): {:5.5%}\
            relative error (vs. ref): {:5.5%}\n".format(err_out, err_ref))

if __name__ == "__main__":
    test_conv2d_unpadded()
    test_conv2d_padded()
