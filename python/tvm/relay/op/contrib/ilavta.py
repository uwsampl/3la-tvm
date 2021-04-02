import tvm.ir
from ...dataflow_pattern import wildcard, is_op
from .register import register_pattern_table


def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator is translated
    to 3LA VTA.
    Paramters
    ---------
    op_name : Str
        The name of operator that will be registered.
    Returns
    -------
    f : callable
        A function that returns if the operator is translated to ILA.
    """
    @tvm.ir.register_op_attr(op_name, "target.ilavta")
    def _func_wrapper(attrs, *args):
        return supported

    return _func_wrapper


# _register_external_op_helper("nn.conv2d")
# _register_external_op_helper("nn.batch_matmul")
_register_external_op_helper("nn.bias_add")
_register_external_op_helper("nn.dense")
# _register_external_op_helper("nn.relu")


def make_pattern_conv2d():
    data = wildcard()
    weight = wildcard()
    conv = is_op('nn.conv2d')(data, weight)
    return conv

def make_pattern_batch_matmul():
    a = wildcard()
    b = wildcard()
    matmul = is_op('nn.batch_matmul')(a, b)
    return matmul

def make_pattern_dense():
    a = wildcard()
    b = wildcard()
    return is_op('nn.dense')(a, b)

def make_pattern_bias_add():
    data = wildcard()
    bias = wildcard()
    return is_op('nn.bias_add')(data, bias)

def make_pattern_relu():
    data = wildcard()
    return is_op('nn.relu')(data)

@register_pattern_table("ilavta")
def pattern_table():
    # conv2d_pat = ("ilavta.conv2d", make_pattern_conv2d())
    matmul_pat = ("ilavta.batch_matmul", make_pattern_batch_matmul())
    dense_pat  = ("ilavta.dense", make_pattern_dense())  
    bias_add_pat = ("ilavta.bias_add", make_pattern_bias_add())
    relu_pat = ("ilavta.relu", make_pattern_relu())
    ilavta_patterns = [matmul_pat, dense_pat, bias_add_pat, relu_pat]
    return ilavta_patterns
