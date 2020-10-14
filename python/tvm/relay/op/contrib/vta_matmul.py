import tvm
from ...dataflow_pattern import wildcard, is_op
from .register import register_pattern_table

@tvm.ir.register_op_attr('nn.dense', 'target.vta_matmul')
def _wrap_nn_dense(attrs, args):
    # print('================ registered vta_matmul ==================')
    return True