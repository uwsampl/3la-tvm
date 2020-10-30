from __future__ import absolute_import, print_function

import os
import tvm
from tvm import te
import numpy as np
from tvm import rpc
from tvm.contrib import util
from tvm.relay.op.contrib import ilavta


# define the graph
dtype="int8"
m = 2
n = 4
b = 1

batch = tvm.relay.TensorType((b, m, n), dtype=dtype)
x = tvm.relay.var("x", batch)
y = tvm.relay.var("y", batch)
z = tvm.relay.nn.batch_matmul(x, y)
w = tvm.relay.nn.batch_matmul(z, z)

mod = tvm.ir.IRModule.from_expr(w)

# pattern matching
pattern_table = ilavta.pattern_table()
mod = tvm.relay.transform.MergeComposite(pattern_table)(mod)
mod = tvm.relay.transform.AnnotateTarget(["ilavta"])(mod) 
#mod = tvm.relay.transform.MergeCompilerRegions()(mod) 
mod = tvm.relay.transform.PartitionGraph()(mod) 
# print(mod)

target = tvm.target.create('llvm')
with tvm.transform.PassContext(opt_level=3):
    graph, lib, params = tvm.relay.build(mod, target)
# print(graph)

##
## execute
##

from tvm.contrib import graph_runtime
# ctx = tvm.context('llvm', 0)
ctx = tvm.cpu()
runtime_exec = graph_runtime.create(graph, lib, ctx)

x_np = np.random.uniform(1, 10, size=(b, m, n)).astype(np.int8)
y_np = np.random.uniform(1, 10, size=(b, m, n)).astype(np.int8)
x_tvm = tvm.nd.array(x_np, ctx=ctx)
y_tvm = tvm.nd.array(y_np, ctx=ctx)
print(x_tvm)
print(y_tvm)

runtime_exec.set_input(0, x_tvm)
runtime_exec.set_input(1, y_tvm)
runtime_exec.set_input(**params)
runtime_exec.run()
