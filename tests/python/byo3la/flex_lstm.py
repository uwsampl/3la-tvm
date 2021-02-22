from __future__ import absolute_import, print_function

import os
import tvm
from tvm import relay
from tvm import runtime
from tvm.contrib import graph_runtime
import numpy as np
from tvm.relay.op.contrib import ilaflex

def test_lstm_layer():
    batch_size = 1
    input_size = 32
    hidden_size = 32
    time_steps = 5

    input_shape = (batch_size, time_steps, input_size)
    state_shape = (batch_size, hidden_size)
    i2h_weight_shape = (4*hidden_size, input_size)
    h2h_weight_shape = (4*hidden_size, hidden_size)
    bias_shape = (4*hidden_size,)

    input_tensor = relay.const(np.random.rand(*input_shape))
    init_state = relay.Tuple([
        relay.const(np.random.rand(*state_shape)),
        relay.const(np.random.rand(*state_shape))
    ])
    i2h_weight = relay.const(np.random.rand(*i2h_weight_shape))
    h2h_weight = relay.const(np.random.rand(*h2h_weight_shape))
    bias = relay.const(np.random.rand(*bias_shape))

    mod = tvm.IRModule()
    # we can probably get rid of some of the type annotations
    # and not need all these arguments
    lstm_call = ilaflex.create_lstm_call(
        mod, input_tensor, init_state,
        i2h_weight, h2h_weight, bias,
        batch_size, input_size, hidden_size, time_steps)

    mod["main"] = relay.Function([], lstm_call)

    # Note: we are not using any BYOC pattern-matching!
    # May have to use the same approach for the linear layer to avoid bad interactions

    target = 'llvm'
    ctx = tvm.cpu()
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        exe = relay.vm.compile(mod, target)
        vm = runtime.vm.VirtualMachine(exe, ctx)
        ret = vm.invoke("main")

    # smoke test: just checking that it runs at all


if __name__ == "__main__":
    test_lstm_layer()
