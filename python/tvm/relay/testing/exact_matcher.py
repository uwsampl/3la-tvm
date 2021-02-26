"""
A very simple substitute for the BYOC pattern matcher that checks for
exact AST matches and replaces them with specially annotated custom
codegen calls
"""
import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprFunctor, ExprMutator
from tvm.relay.analysis import free_vars, bound_vars

# dumb copy of what src/relay/transforms/de_duplicate.cc is doing
def deduplicate_vars(expr):
    """
    Given the expr, replace all vars in the expression with fresh ones.
    This is done to preserve well-formedness in Relay (all var definitions must be unique)
    """
    class Deduplicator(ExprMutator):
        def __init__(self):
            super().__init__()
            self.var_map = {}

        def visit_var(self, var):
            if var in self.var_map:
                return self.var_map[var]
            fresh_var = relay.Var(var.name_hint, type_annotation=var.type_annotation)
            self.var_map[var] = fresh_var
            return fresh_var

        def visit_pattern(self, pattern):
            if isinstance(pattern, relay.PatternWildcard):
                return pattern
            if isinstance(pattern, relay.PatternVar):
                return relay.PatternVar(self.visit(pattern.var))
            if isinstance(pattern, relay.PatternTuple):
                return relay.PatternTuple([self.visit(subpattern)
                                           for subpattern in pattern.patterns])
            if isinstance(pattern, relay.PatternConstructor):
                return relay.PatternConstructor(pattern.constructor,
                                                [self.visit(subpattern)
                                                 for subpattern in pattern.patterns])
            raise ValueError(f"Invalid pattern {pattern}")

        def visit_match(self, match):
            new_val = self.visit(match.data)
            clauses = [relay.Clause(self.visit_pattern(c.lhs), self.visit(c.rhs))
                       for c in match.clauses]
            return relay.Match(new_val, clauses)

    dedup = Deduplicator()
    return dedup.visit(expr)

def check_match(template, target):
    """
    Given a template expression and a target expression,
    return (bool, optional dict):
    the bool is if the expressions match structurally;
    the dict is None if they don't match
    and gives a mapping of template vars -> target expressions if they do.
    (Free vars in the template are treated as match vars.)
    """
    class Matcher(ExprFunctor):
        def __init__(self, match_template):
            super().__init__()
            self.template = match_template
            self.template_vars = set(free_vars(match_template))
            self.matched_exprs = {}
            # for *bound* vars, we need to be sure they correspond
            self.var_mapping = {}

        def check_assigned_matches(self, expr):
            if not isinstance(self.template, relay.Var):
                return False
            if self.template not in self.template_vars:
                return False
            if self.template not in self.matched_exprs:
                self.matched_exprs[self.template] = expr
                return True
            # if it's something we've already matched, has to be an exact match
            return tvm.ir.structural_equal(self.matched_exprs[self.template], expr)

        def check_nested_match(self, template_subexpr, expr):
            old_template = self.template
            self.template = template_subexpr
            ret = self.visit(expr)
            self.template = old_template
            return ret

        def visit_var(self, var):
            if self.check_assigned_matches(var):
                return True
            if var not in self.var_mapping:
                return False
            return self.template == self.var_mapping[var]

        def trivial_equality(self, other):
            if self.check_assigned_matches(other):
                return True
            return self.template == other

        def visit_constant(self, const):
            return self.trivial_equality(const)

        def visit_constructor(self, ctor):
            return self.trivial_equality(ctor)

        def visit_global_var(self, gv):
            return self.trivial_equality(gv)

        def visit_op(self, op):
            return self.trivial_equality(op)

        def visit_let(self, let):
            if self.check_assigned_matches(let):
                return True
            if not isinstance(self.template, relay.Let):
                return False
            self.var_mapping[let.var] = self.template.var
            value_match = self.check_nested_match(self.template.value, let.value)
            if not value_match:
                return False
            return self.check_nested_match(self.template.body, let.body)

        def visit_function(self, func):
            if self.check_assigned_matches(func):
                return True
            if not isinstance(self.template, relay.Function):
                return False
            if len(func.params) != len(self.template.params):
                return False
            for i in range(len(func.params)):
                self.var_mapping[func.params[i]] = self.template.params[i]
            return self.check_nested_match(self.template.body, func.body)

        def visit_call(self, call):
            if self.check_assigned_matches(call):
                return True
            if not isinstance(self.template, relay.Call):
                return False
            if len(self.template.args) != len(call.args):
                return False
            if not self.check_nested_match(self.template.op, call.op):
                return False
            for i in range(len(call.args)):
                if not self.check_nested_match(self.template.args[i], call.args[i]):
                    return False
            return True

        def visit_tuple(self, tup):
            if self.check_assigned_matches(tup):
                return True
            if not isinstance(self.template, relay.Tuple):
                return False
            if len(self.template.fields) != len(tup.fields):
                return False
            for i in range(len(tup.fields)):
                if not self.check_nested_match(self.template.fields[i], call.fields[i]):
                    return False
            return True

        def visit_if(self, if_expr):
            if self.check_assigned_matches(if_expr):
                return True
            if not isinstance(self.template, relay.If):
                return False
            if not self.check_nested_match(self.template.cond, if_expr.cond):
                return False
            if not self.check_nested_match(self.template.true_branch, if_expr.true_branch):
                return False
            return self.check_nested_match(self.template.false_branch, if_expr.false_branch)

        def visit_tuple_getitem(self, tgi):
            if self.check_assigned_matches(tgi):
                return True
            if not isinstance(self.template, relay.TupleGetItem):
                return False
            if self.template.index != tgi.index:
                return False
            return self.check_nested_match(self.template.tuple_value, tgi.tuple_value)

        def check_nested_pattern(self, template_pattern, pattern):
            if isinstance(pattern, relay.PatternWildcard):
                return template_pattern == pattern
            if isinstance(pattern, relay.PatternVar):
                if not isinstance(template_pattern, relay.PatternVar):
                    return False
                return self.check_nested_match(template_pattern.var, pattern.var)
            if isinstance(pattern, relay.PatternTuple):
                if not isinstance(template_pattern, relay.PatternTuple):
                    return False
                if len(template_pattern.patterns) != len(pattern.patterns):
                    return False
                for i in range(len(pattern.patterns)):
                    if not self.check_nested_pattern(template_pattern.patterns[i],
                                                     pattern.patterns[i]):
                        return False
                return True
            if isinstance(pattern, relay.PatternConstructor):
                if not isinstance(template_pattern, relay.PatternConstructor):
                    return False
                if len(template_pattern.patterns) != len(pattern.patterns):
                    return False
                if template_pattern.constructor != pattern.constructor:
                    return False
                for i in range(len(pattern.patterns)):
                    if not self.check_nested_pattern(template_pattern.patterns[i],
                                                     pattern.patterns[i]):
                        return False
                return True
            raise ValueError(f"Invalid pattern: {pattern}")

        def visit_match(self, match):
            if self.check_assigned_matches(match):
                return True
            if not isinstance(self.template, relay.Match):
                return False
            if not self.check_nested_match(self.template.data, match.data):
                return False
            if len(self.template.clauses) != len(match.clauses):
                return False
            for i in range(len(match.clauses)):
                template_clause = self.template.clauses[i]
                clause = match.clauses[i]
                if not self.check_nested_pattern(template_clause.lhs, clause.lhs):
                    return False
                if not self.check_nested_match(template.clause.rhs, clause.rhs):
                    return False
            return True

        # punting on refs for now

    matcher = Matcher(template)
    res = matcher.visit(target)
    if not res:
        return False, None

    mapping = matcher.matched_exprs
    # check to make sure none of the mappings breaks the variable scoping
    tgt_bound = set(bound_vars(target))
    for matched in mapping.values():
        # if a bound var in the target is free in a matched fragment,
        # that means we are taking a bound var out of its scope
        matched_free = set(free_vars(matched))
        if len(matched_free.intersection(tgt_bound)) != 0:
            return False, None

    return res, mapping

class MatchMutator(ExprMutator):
    def __init__(self, target, compiler_name, composite_name, composite_counter=0):
        """
        Target: Expression that the matcher is seeking
        Compiler name: Name for the custom codegen
        Composite name: Name for the *construct produced* in the custom codegen
        Composite counter: Id number used for generating compiler IDs
                           (they must be globally unique)

        Free vars in the target expression will be arguments to the extracted function
        """
        super().__init__()
        self.target = target
        self.target_vars = free_vars(target)
        self.compiler_name = compiler_name
        self.composite_name = composite_name
        self.composite_counter = composite_counter

    def extract_target(self, match_args):
        match_ordering = [match_args[v] if v in match_args else relay.Tuple([]) for v in self.target_vars]
        inner_body = deduplicate_vars(self.target)
        inner_args = free_vars(inner_body)
        inner_func = relay.Function(inner_args, inner_body)
        inner_func = inner_func.with_attr("Composite", self.composite_name)

        outer_args = [relay.Var(f"outer_arg_{i}") for i in range(len(inner_args))]
        outer_func = relay.Function(outer_args, inner_func(*outer_args))
        outer_func = outer_func.with_attr("Compiler", self.compiler_name)
        outer_func = outer_func.with_attr(
            "global_name",
            f"{self.composite_name}_{self.composite_counter}")
        self.composite_counter += 1
        return outer_func(*match_ordering)

    # could probably do this via reflection, but let's not...
    def visit_var(self, var):
        found_match, match_args = check_match(self.target, var)
        if found_match:
            return self.extract_target(match_args)
        return var

    def visit_constant(self, constant):
        found_match, match_args = check_match(self.target, constant)
        if found_match:
            return self.extract_target(match_args)
        return constant

    def visit_call(self, call):
        found_match, match_args = check_match(self.target, call)
        if found_match:
            return self.extract_target(match_args)
        return relay.Call(self.visit(call.op),
                          [self.visit(arg) for arg in call.args],
                          call.attrs)

    def visit_tuple(self, tup):
        found_match, match_args = check_match(self.target, tup)
        if found_match:
            return self.extract_target(match_args)
        return relay.Tuple([self.visit(field) for field in tup.fields])

    def visit_function(self, func):
        # headache-saver: if it's a codegen function, we will ignore it
        if func.attrs is not None and ("Compiler" in func.attrs or "Composite" in func.attrs):
            return func
        found_match, match_args = check_match(self.target, func)
        if found_match:
            return self.extract_target(match_args)
        return relay.Function(func.params,
                              self.visit(func.body),
                              func.ret_type,
                              func.type_params,
                              func.attrs)

    def visit_if(self, if_expr):
        found_match, match_args = check_match(self.target, if_expr)
        if found_match:
            return self.extract_target(match_args)
        return relay.If(self.visit(if_expr.cond),
                        self.visit(if_expr.true_branch),
                        self.visit(if_expr.false_branch))

    def visit_tuple_getitem(self, tgi):
        found_match, match_args = check_match(self.target, tgi)
        if found_match:
            return self.extract_target(match_args)
        return relay.TupleGetItem(self.visit(tgi.tuple_value), tgi.index)

    def visit_op(self, op):
        found_match, match_args = check_match(self.target, op)
        if found_match:
            return self.extract_target(match_args)
        return op

    def visit_global_var(self, gv):
        found_match, match_args = check_match(self.target, gv)
        if found_match:
            return self.extract_target(match_args)
        return gv

    def visit_constructor(self, ctor):
        found_match, match_args = check_match(self.target, ctor)
        if found_match:
            return self.extract_target(match_args)
        return ctor

    def visit_let(self, let):
        found_match, match_args = check_match(self.target, let)
        if found_match:
            return self.extract_target(match_args)
        return relay.Let(let.var, self.visit(let.value), self.visit(let.body))

    def visit_match(self, match):
        found_match, match_args = check_match(self.target, match)
        if found_match:
            return self.extract_target(match_args)
        return Match(
            self.visit(match.data),
            [Clause(c.lhs, self.visit(c.rhs)) for c in match.clauses],
            complete=match.complete,
        )

    # again, punting on refs

def annotate_exact_matches(expr, target, compiler_name, composite_name):
    """
    Given an expression and a target pattern,
    this will replace all instances of the target pattern
    in the expression with an annotated compiler call.

    Free variables (not bound in a let block or function definition) are treated as pattern variables.

    That means if your pattern is relay.nn.dense(x, y),
    x and y are pattern vars and you will match
    any occurrences of relay.nn.dense() in the expression
    and map the arguments to x and y

    Here is what the resulting inserted calls look like:

    (fn*(pattern_vars) {
       (fn**(fresh vars) {
          pattern expr (with fresh vars substituting the pattern vars)
        })(pattern vars)
    })(expression matching pattern var 1, expr matching pattern var 2, ...)
    * where "Compiler" attribute = compiler_name
    ** where "Composite" attribute = composite_name

    This nested function structure is designed to make it easier for BYOC codegens to match those definitions.
    """
    mut = MatchMutator(target, compiler_name, composite_name)
    return mut.visit(expr)
