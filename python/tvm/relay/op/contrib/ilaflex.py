import tvm.ir
from ...dataflow_pattern import wildcard, is_op
from .register import register_pattern_table


def make_pattern_linear():
    a = wildcard()
    b = wildcard()
    c = wildcard()
    matmul = is_op('nn.batch_matmul')(a, b)
    linear = is_op('nn.bias_add')(matmul, c)
    return linear

@register_pattern_table("ilaflex")
def pattern_table():
    linear_pat = ("ilaflex.linear", make_pattern_linear())
    ilaflex_patterns = [linear_pat]
    return ilaflex_patterns
