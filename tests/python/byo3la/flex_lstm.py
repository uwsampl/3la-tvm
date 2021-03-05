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
    input_size = 64
    hidden_size = 64
    time_steps = 3

    input_shape = (batch_size, time_steps, input_size)
    state_shape = (batch_size, hidden_size)
    i2h_weight_shape = (4*hidden_size, input_size)
    h2h_weight_shape = (4*hidden_size, hidden_size)
    bias_shape = (4*hidden_size,)

    coef = 1
    input_tensor = relay.const(coef*np.random.uniform(-1,1,input_shape))
    init_state = relay.Tuple([
        relay.const(coef*np.random.uniform(-1, 1, state_shape)),
        relay.const(coef*np.random.uniform(-1, 1, state_shape))
    ])
    i2h_weight = relay.const(coef*np.random.uniform(-1, 1, i2h_weight_shape))
    h2h_weight = relay.const(coef*np.random.uniform(-1, 1, h2h_weight_shape))
    # bias = relay.const(coef*np.random.uniform(-1, 1, bias_shape))
    bias = relay.const(np.zeros(*bias_shape))

    # print(input_tensor)
    # print(i2h_weight)
    # print(h2h_weight)
    # print(bias)

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
        out = ret.asnumpy()

    print('lstm result is \n {}'.format(out))

    # smoke test: just checking that it runs at all


if __name__ == "__main__":
    test_lstm_layer()
