import tvm.ir
from ...dataflow_pattern import wildcard, is_op
from .register import register_pattern_table


def make_pattern_linear():
    a = wildcard()
    b = wildcard()
    c = wildcard()
    dense = is_op('nn.dense')(a, b)
    linear = is_op('nn.bias_add')(dense, c)
    return linear

@register_pattern_table("ilaflex")
def pattern_table():
    linear_pat = ("ilaflex.linear", make_pattern_linear())
    ilaflex_patterns = [linear_pat]
    return ilaflex_patterns
