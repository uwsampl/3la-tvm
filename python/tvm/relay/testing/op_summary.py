"""
Utility functions for counting the number of operators
and BYOC overloads in modules.
"""
import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprVisitor

def is_overload(func):
    if func.attrs is None:
        return False
    return "Compiler" in func.attrs


def get_count_expr(counter_class, expr):
    counter = counter_class()
    counter.visit(expr)
    return counter.count


def get_count_mod(counter_class, mod):
    total_count = 0
    for gv in mod.get_global_vars():
        total_count += get_count_expr(counter_class, mod[gv])
    return total_count


class Counter(ExprVisitor):
    def __init__(self):
        super().__init__()
        self.count = 0

    def eligible(self, expr):
        raise NotImplementedError()

    def increment(self, expr):
        return 1

    def visit(self, expr):
        if self.eligible(expr):
            self.count += self.increment(expr)
        super().visit(expr)


class OpCounter(Counter):
    def eligible(self, expr):
        return isinstance(expr, tvm.ir.op.Op)


class OverloadCounter(Counter):
    def eligible(self, expr):
        return isinstance(expr, relay.Function) and is_overload(expr)


class OpInOverloadCounter(Counter):
    def eligible(self, expr):
        return isinstance(expr, relay.Function) and is_overload(expr)

    def increment(self, expr):
        return get_count_expr(OpCounter, expr)


def count_all_ops(mod):
    return get_count_mod(OpCounter, mod)


def count_all_overloads(mod):
    return get_count_mod(OverloadCounter, mod)


def count_all_ops_in_overloads(mod):
    return get_count_mod(OpInOverloadCounter, mod)
