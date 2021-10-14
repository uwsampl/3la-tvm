from __future__ import absolute_import, print_function

import os
import tvm
from tvm import te
import numpy as np
from tvm import rpc
# from tvm.contrib import util
from tvm.relay.op.contrib import ilaflex

from flexnlp.src.utils import tool

#from utils import tool

# define the graph
# dtype="int8"
dtype="float32"
p = 1
m = 16
n = 10

x_shape = (p, m)
w_shape = (n, m)
b_shape = (n, )

shape1 = tvm.relay.TensorType(x_shape, dtype=dtype)
shape2 = tvm.relay.TensorType(w_shape, dtype=dtype)
shape3 = tvm.relay.TensorType(b_shape, dtype=dtype)
x1 = tvm.relay.var("x1", shape1)
w1 = tvm.relay.var("w1", shape2)
b1 = tvm.relay.var("b1", shape3)
x1 = tvm.relay.var("x2", shape1)
w2 = tvm.relay.var("w2", shape2)
b2 = tvm.relay.var("b2", shape3)
y1 = tvm.relay.nn.bias_add(tvm.relay.nn.dense(x1, w1), b1)
y2 = tvm.relay.nn.bias_add(tvm.relay.nn.dense(y1, w2), b2)

mod = tvm.ir.IRModule.from_expr(y1)
print(mod)


# pattern matching
pattern_table = ilaflex.pattern_table()
mod = tvm.relay.transform.MergeComposite(pattern_table)(mod)
mod = tvm.relay.transform.AnnotateTarget(["ilaflex"])(mod) 
mod = tvm.relay.transform.PartitionGraph()(mod) 
print("mod after annotation\n")
print(mod)

print("[Python] Compile with the 3LA extension")
target = tvm.target.create('llvm')
with tvm.transform.PassContext(opt_level=3):
    graph, lib, params = tvm.relay.build(mod, target)

##
## execute
##
from tvm.contrib import graph_runtime
ctx = tvm.cpu()
runtime_exec = graph_runtime.create(graph, lib, ctx)

coef = 1

if dtype == "int8":
    x_np = np.random.uniform(-128, 128, size=x_shape).astype(dtype)
    y_np = np.random.uniform(-128, 128, size=w_shape).astype(dtype)
    z_np = np.random.uniform(-128, 128, size=b_shape).astype(dtype)
else:
    x_np = coef * np.random.uniform(0, 1, size=x_shape).astype(dtype)
    y_np = coef * np.random.uniform(0, 1, size=w_shape).astype(dtype)
    z_np = coef * np.random.uniform(0, 1, size=b_shape).astype(dtype)


ref = np.add(np.matmul(x_np, np.transpose(y_np)), z_np)
# print(ref)
x_tvm = tvm.nd.array(x_np)
y_tvm = tvm.nd.array(y_np)
z_tvm = tvm.nd.array(z_np)

print("[Python] Execute the compiled model")
runtime_exec.set_input(0, x_tvm)
runtime_exec.set_input(1, y_tvm)
runtime_exec.set_input(2, z_tvm)
runtime_exec.set_input(**params)
runtime_exec.run()

output = runtime_exec.get_output(0).asnumpy()
output = output.astype(np.float32)
print("[Python] Done")

tl = tool()
err_out, err_ref = tl.cal_error(output, ref)
print("relative error: {:5.5%} vs. output, {:5.5%} vs. ref".format(err_out, err_ref))
# print('output: {}'.format(output.shape))
# print(output)
print('===============')
# print('ref: {}'.format(ref.shape))
# print(ref)
