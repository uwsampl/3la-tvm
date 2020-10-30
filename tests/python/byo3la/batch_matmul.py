from __future__ import absolute_import, print_function

import os
import tvm
from tvm import te
import numpy as np
from tvm import rpc
from tvm.contrib import util
from tvm.relay.op.contrib import ilavta

print("[Python] Import/define the application Relay model")

# define the graph
dtype="int8"
m = 2
n = 4
b = 1

shape1 = tvm.relay.TensorType((b, m, n), dtype=dtype)
shape2 = tvm.relay.TensorType((b, m, m), dtype=dtype)
x = tvm.relay.var("x", shape1)
y = tvm.relay.var("y", shape1)
z = tvm.relay.var("z", shape2)
inter = tvm.relay.nn.batch_matmul(x, y)
final = tvm.relay.nn.batch_matmul(z, inter)

mod = tvm.ir.IRModule.from_expr(final)


# pattern matching
pattern_table = ilavta.pattern_table()
mod = tvm.relay.transform.MergeComposite(pattern_table)(mod)
mod = tvm.relay.transform.AnnotateTarget(["ilavta"])(mod) 
#mod = tvm.relay.transform.MergeCompilerRegions()(mod) 
mod = tvm.relay.transform.PartitionGraph()(mod) 

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

x_np = np.random.uniform(1, 10, size=(b, m, n)).astype(np.int8)
y_np = np.random.uniform(1, 10, size=(b, m, n)).astype(np.int8)
z_np = np.random.uniform(1, 10, size=(b, m, m)).astype(np.int8)
x_tvm = tvm.nd.array(x_np, ctx=ctx)
y_tvm = tvm.nd.array(y_np, ctx=ctx)
z_tvm = tvm.nd.array(z_np, ctx=ctx)

print("[Python] Execute the compiled model")
runtime_exec.set_input(0, z_tvm)
runtime_exec.set_input(1, x_tvm)
runtime_exec.set_input(2, y_tvm)
runtime_exec.set_input(**params)
runtime_exec.run()

output = runtime_exec.get_output(0).asnumpy()
output = output.astype(np.uint8)
print(output)
print("[Python] Done")
