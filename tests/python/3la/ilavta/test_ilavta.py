# Required directories / files for running this test:
# - Make sure `produce_ila_fragment.py` (https://github.com/uwsampl/3la-vta-testbench/blob/main/test/produce_ila_fragment.py)
#   is under the directory where this script is being called
# - Make sure `vta.testing.ila_converter` presents in the vta python library
# - Create two directories named `prog_frag` and `result` under the directory where this script is being called
#
# This script can be called after creating the two directories under `3la-vta-testbench/test` (https://github.com/uwsampl/3la-vta-testbench/tree/main/test)
import tvm
import numpy as np
from tvm.relay.op.contrib import ilavta
from tvm.contrib import graph_runtime
import tvm.topi as topi

def check_global_func():
    assert tvm.get_global_func('relay.ext.ilavta')

def run_passes(mod):
    patterns = ilavta.pattern_table()
    mod = tvm.relay.transform.MergeComposite(patterns)(mod)
    mod = tvm.relay.transform.AnnotateTarget('ilavta')(mod)
    mod = tvm.relay.transform.PartitionGraph()(mod)
    print('[Python] Transformation complete')
    return mod

def run_module(mod, *inputs):
    target = tvm.target.create('llvm')
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = tvm.relay.build(mod, target)
    ctx = tvm.cpu()
    runtime_exec = graph_runtime.create(graph, lib, ctx)
    
    input_tensors = list(map(lambda x: tvm.nd.array(x, ctx=ctx), inputs))
    
    print('[Python] Execute Graph')
    for (i, inp) in enumerate(input_tensors):
        runtime_exec.set_input(i, inp)
    runtime_exec.set_input(**params)
    runtime_exec.run()

    output = runtime_exec.get_output(0).asnumpy()
    print('[Python] Done')
    return output

def test_dense_impl(in_dim, w_dim, func_ref = lambda inp, wgt: np.matmul(inp, wgt.transpose())):
    check_global_func()
    dtype = 'int8'
    print('[Python] Running on {} mult {}'.format(in_dim, w_dim))
    
    ishape = tvm.relay.TensorType(in_dim, dtype=dtype)
    wshape = tvm.relay.TensorType(w_dim, dtype=dtype)
    
    inputs = tvm.relay.var('inputs', ishape)
    weight = tvm.relay.var('weight', wshape)

    output = tvm.relay.nn.dense(inputs, weight)

    mod = tvm.ir.IRModule.from_expr(output)
    mod = run_passes(mod)
    
    inp = np.array([np.random.randint(1, 3) for _ in range(in_dim[0] * in_dim[1])]).reshape(in_dim).astype(np.int8)
    wgt = np.array([np.random.randint(1, 3) for _ in range(w_dim[0] * w_dim[1])]).reshape(w_dim).astype(np.int8)

    ref = func_ref(inp, wgt) 
    output = run_module(mod, inp, wgt).astype(dtype)
    np.allclose(output, ref)


def test_nested_dense(*shapes):
    dtype = 'int8'
    
    ashape, bshape, cshape, dshape = [tvm.relay.TensorType(dim, dtype=dtype) for dim in shapes]
    a = tvm.relay.var('a', ashape)
    b = tvm.relay.var('b', bshape)
    c = tvm.relay.var('c', cshape)
    d = tvm.relay.var('d', dshape)

    output = tvm.relay.nn.dense(
        tvm.relay.nn.dense(a, b),
        tvm.relay.nn.dense(c, d)
    )

    mod = tvm.ir.IRModule.from_expr(output)
    mod = run_passes(mod)

    inputs = [np.random.random_integers(0, 5, shape).astype(np.int8) for shape in shapes]
    p = np.matmul(inputs[0], inputs[1].transpose())
    q = np.matmul(inputs[2], inputs[3].transpose())
    ref = np.matmul(p, q.transpose())
    
    output = run_module(mod, *inputs).astype(np.int8)
    np.allclose(output, ref)

def test_dense_subgraph(in_dim, wgt_dim):
    dtype = 'int8'

    ishape = tvm.relay.TensorType(in_dim, dtype=dtype)
    wshape = tvm.relay.TensorType(wgt_dim, dtype=dtype)

    inp = tvm.relay.var('input', ishape)
    wgt = tvm.relay.var('weight', wshape)

    o1 = inp + inp 
    o2 = tvm.relay.nn.dense(o1, wgt)
    o3 = tvm.relay.nn.relu(o2)
    output = tvm.relay.nn.dense(o3, o3)

    mod = tvm.ir.IRModule.from_expr(output)
    mod = run_passes(mod)

    v_inp = np.random.random_integers(0, 5, in_dim).astype(np.int8)
    v_wgt = np.random.random_integers(0, 5, wgt_dim).astype(np.int8)

    def func_ref(inp, wgt):
        def relu_(x):
            return x * (x > 0)
        o1 = inp + inp
        o2 = np.matmul(o1, wgt.transpose())
        o3 = relu_(o2)
        return np.matmul(o3, o3.transpose())
    
    output = run_module(mod, v_inp, v_wgt).astype(np.int8)
    ref = func_ref(v_inp, v_wgt).astype(np.int8)

    np.allclose(output, ref)

def test_dense():
    for batch in [8, 16, 32, 64]:
        for n_inp_cols in [16, 32, 64]:
            for n_wgt_rows in [16, 32, 64]:
                in_dim = (batch, n_inp_cols)
                wgt_dim = (n_wgt_rows, n_inp_cols)
                print('[Python] Testing on {} . {}'.format(in_dim, wgt_dim))
                
                print('[Case] Single Dense')
                test_dense_impl(in_dim ,wgt_dim)
                
                print('[Case] Nested Dense')
                test_nested_dense(in_dim, wgt_dim, in_dim, wgt_dim)

                print('[Case] Dense in Subgraph')
                test_dense_subgraph(in_dim, wgt_dim)

def test_bias_add_impl(in_dim, bias_dim):
    input_dtype = 'int32'
    output_dtype = 'int8'

    data_shape = tvm.relay.TensorType(in_dim, dtype=input_dtype)
    bias_shape = tvm.relay.TensorType(bias_dim, dtype=input_dtype)

    inputs = tvm.relay.var('data', data_shape)
    bias   = tvm.relay.var('bias', bias_shape)

    output = tvm.relay.nn.bias_add(inputs, bias)

    mod = tvm.ir.IRModule.from_expr(output)
    mod = run_passes(mod)
    
    inputs_data = np.random.random_integers(0, 50, in_dim).astype(input_dtype)
    bias_data   = np.random.random_integers(0, 50, bias_dim).astype(input_dtype)
    def func_ref(x, y):
        z = x.copy()
        for i in range(x.shape[0]):
            z[i] = x[i] + y
        return z
    
    out_data = run_module(mod, inputs_data, bias_data).astype(np.int8)
    ref = func_ref(inputs_data, bias_data).astype(np.int8)
    
    np.allclose(out_data, ref)

def test_bias_add():
    for batch in [8, 16, 32, 64]:
        for n_inp_cols in [16, 32, 64]:
            data_dim = (batch, n_inp_cols)
            bias_dim = (n_inp_cols, )
            print('[Python] Bias Add on {} + {}'.format(data_dim, bias_dim))
            test_bias_add_impl(data_dim, bias_dim)

if __name__ == '__main__':
    check_global_func()
    test_dense()
    test_bias_add()
