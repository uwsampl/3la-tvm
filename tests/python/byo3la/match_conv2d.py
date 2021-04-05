import numpy as np
import tvm
from tvm import relay
from tvm import runtime
from tvm.relay.op.contrib import ilacnn

# just some simple smoke tests
def test_conv2d_unpadded():
    x = relay.Var("x", type_annotation=relay.TensorType((1, 3, 224, 224)))
    y = relay.Var("y", type_annotation=relay.TensorType((3, 3, 3, 3)))
    conv_func = relay.Function([x, y], relay.nn.conv2d(x, y))
    mod = tvm.IRModule()
    mod["main"] = conv_func

    pattern_table = ilacnn.pattern_table()
    mod = tvm.relay.transform.MergeComposite(pattern_table)(mod)
    mod = tvm.relay.transform.AnnotateTarget(["ilacnn"])(mod)
    mod = tvm.relay.transform.PartitionGraph()(mod)
    print(mod)

    with tvm.transform.PassContext(opt_level=3):
        device = tvm.cpu()
        target = "llvm"
        exe = relay.vm.compile(mod, target)
        vm = runtime.vm.VirtualMachine(exe, device)

        args = [np.random.rand(1, 3, 224, 224).astype("float32"),
                np.random.rand(3, 3, 3, 3).astype("float32")]
        ret = vm.invoke("main", *args)


def test_conv2d_padded():
    x = relay.Var("x", type_annotation=relay.TensorType((1, 3, 220, 218)))
    y = relay.Var("y", type_annotation=relay.TensorType((3, 3, 3, 3)))
    conv_func = relay.Function([x, y], relay.nn.conv2d(x, y, padding=(2, 3)))
    mod = tvm.IRModule()
    mod["main"] = ilacnn.remove_padding(conv_func)

    pattern_table = ilacnn.pattern_table()
    mod = tvm.relay.transform.MergeComposite(pattern_table)(mod)
    mod = tvm.relay.transform.AnnotateTarget(["ilacnn"])(mod)
    mod = tvm.relay.transform.PartitionGraph()(mod)
    print(mod)

    with tvm.transform.PassContext(opt_level=3):
        device = tvm.cpu()
        target = "llvm"
        exe = relay.vm.compile(mod, target)
        vm = runtime.vm.VirtualMachine(exe, device)

        args = [np.random.rand(1, 3, 220, 218).astype("float32"),
                np.random.rand(3, 3, 3, 3).astype("float32")]
        ret = vm.invoke("main", *args)


if __name__ == "__main__":
    test_conv2d_unpadded()
    test_conv2d_padded()
