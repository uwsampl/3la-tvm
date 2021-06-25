"""
This implements a rewrite of the ResMLP linear layer
as imported by TVM's PT importer into the format
expected by the FlexNLP codegen.

It sure would be nice to do this with some simple
rewrite rules instead!
"""
import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprMutator

class LinearLayerRewriter(ExprMutator):
    """
    Rewrites the linear layers implemented
    in the automatically imported ResMLP
    (which look like reshape(dense(*, *), (1, 256, 512)) + *)
    into those matched in FlexNLP
    (reshape(bias_add(dense(*, *), *)), (1, 256, 512))

    It would be preferable to find simple rewrite rules
    to do this via a rewrite system!
    """
    def visit_call(self, call):
        if not isinstance(call.op, tvm.ir.op.Op):
            return super().visit_call(call)
        op = call.op
        if op.name != "add":
            return super().visit_call(call)
        if len(call.args) != 2:
            return super().visit_call(call)

        left_arg = call.args[0]
        bias_arg = call.args[1]
        if not isinstance(left_arg, relay.Call):
            return super().visit_call(call)
        left_op = left_arg.op
        if left_op.name != "reshape":
            return super().visit_call(call)
        reshape_attrs = left_arg.attrs
        if tuple(reshape_attrs.newshape) != (1, 256, 512):
            return super().visit_call(call)

        reshape_arg = left_arg.args[0]
        if reshape_arg.op.name != "nn.dense":
            return super().visit_call(call)

        # otherwise we have a match
        dense_term = self.visit(reshape_arg)
        bias_term = self.visit(bias_arg)
        new_term = relay.reshape(
            relay.nn.bias_add(dense_term, bias_term),
            (1, 256, 512))
        return new_term
