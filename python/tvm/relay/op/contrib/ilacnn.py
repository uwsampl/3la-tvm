"""
Python bindings and helpers for ILACNN codegen,
note that the accelerator does not do padding for Conv2D's,
so you should use remove_padding on the main function before pattern matching
(this converts conv2d's with padding to conv2d(pad(data)))
"""
import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprMutator
import tvm.ir
from ...dataflow_pattern import wildcard, is_op
from .register import register_pattern_table

def remove_padding(func):
    """
    The CNN accelerator cannot handle padding in conv2d,
    so this will rewrite all conv2d's with padding into
    conv2d on a separately padded tensor (i.e., handle padding in the host)
    """
    class PaddingRemover(ExprMutator):
        def visit_call(self, call):
            if call.attrs is None:
                return super().visit_call(call)
            attrs = call.attrs
            if not isinstance(attrs, relay.op.op_attrs.Conv2DAttrs):
                return super().visit_call(call)
            padding = attrs.padding
            # nothing to do if no padding
            if all(map(lambda d: d == 0, padding)):
                return super().visit_call(call)

            # otherwise rewrite as a padded call
            data = self.visit(call.args[0])
            weight = self.visit(call.args[1])

            # relay.nn.pad expects padding in the format of (x_left, x_right), (y_top, y_bottom)
            data_layout = attrs.data_layout
            # we are only padding the H and W dimensions
            pad_dims = [(0, 0), (0, 0), (padding[0], padding[2]), (padding[1], padding[3])]
            if data_layout == "NHWC":
                pad_dims = [(0, 0), (padding[0], padding[2]), (padding[1], padding[3]), (0, 0)]

            padded_data = relay.nn.pad(data, pad_dims)
            return relay.nn.conv2d(padded_data, weight,
                                   strides=attrs.strides,
                                   padding=0,
                                   dilation=attrs.dilation,
                                   groups=attrs.groups,
                                   channels=attrs.channels,
                                   kernel_size=attrs.kernel_size,
                                   data_layout=attrs.data_layout,
                                   kernel_layout=attrs.kernel_layout,
                                   out_layout=attrs.out_layout,
                                   out_dtype=attrs.out_dtype)

    remover = PaddingRemover()
    return remover.visit(func)


@register_pattern_table("ilacnn")
def pattern_table():
    conv2d_pattern = ("ilacnn.conv2d", is_op('nn.conv2d')(wildcard(), wildcard()))
    return [conv2d_pattern]
