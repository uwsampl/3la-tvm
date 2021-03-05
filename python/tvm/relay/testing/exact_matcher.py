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
                return relay.PatternTuple([self.visit_pattern(subpattern)
                                           for subpattern in pattern.patterns])
            if isinstance(pattern, relay.PatternConstructor):
                return relay.PatternConstructor(pattern.constructor,
                                                [self.visit_pattern(subpattern)
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
            # template: What we are matching against at any given point
            # (note: since we cannot add extra args to overloaded functions,
            #  we have to mutate this field when we go down subtrees
            #  and set it back once we return)
            self.template = match_template
            # template vars: Holes we will fill with matching subexpressions.
            # The matcher will check that the matches are consistent
            # (that the same hole matches the same expression everywhere
            #  -- must be an exact syntactic match)
            self.template_vars = set(free_vars(match_template))
            # map of template vars to matched expressions
            self.matched_exprs = {}
            # map of *bound* variables in the template to *bound* variables
            # in the input; this mapping must correspond to have a match
            self.var_mapping = {}

        def at_template_var(self):
            """
            Returns True iff we are at a template var
            """
            if not isinstance(self.template, relay.Var):
                return False
            return self.template in self.template_vars

        def check_assigned_matches(self, expr):
            """
            Checks if we are comparing the expression to a template variable
            and updates the match dictionary if we are.

            Returns True for a new match assignment or a consistent match;
            False for a match inconsistent with previous assignments
            """
            assert self.at_template_var()
            if self.template not in self.matched_exprs:
                self.matched_exprs[self.template] = expr
                return True
            # if it's something we've already matched, has to be an exact match
            return tvm.ir.structural_equal(self.matched_exprs[self.template], expr)

        def check_direct_match(self, expr):
            # wrapper over direct visitor that first
            # checks if we have a template var match
            # and also checks if the types match
            if self.at_template_var():
                return self.check_assigned_matches(expr)
            # if they're not the same type, they can't possibly match
            if not isinstance(self.template, type(expr)):
                return False
            return self.visit(expr)

        def check_nested_match(self, template_subexpr, expr):
            """
            Recurses by reassigning the template variable to the subexpression
            and assigning back after we're done (ugly, yes)
            """
            old_template = self.template
            self.template = template_subexpr
            ret = self.check_direct_match(expr)
            self.template = old_template
            return ret

        def visit_var(self, var):
            if var not in self.var_mapping:
                return False
            return self.template == self.var_mapping[var]

        def trivial_equality(self, other):
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
            if (let.var in self.var_mapping
                and self.var_mapping[let.var] != self.template.var):
                return False
            self.var_mapping[let.var] = self.template.var
            value_match = self.check_nested_match(self.template.value, let.value)
            if not value_match:
                return False
            return self.check_nested_match(self.template.body, let.body)

        def visit_function(self, func):
            if len(func.params) != len(self.template.params):
                return False
            for i in range(len(func.params)):
                self.var_mapping[func.params[i]] = self.template.params[i]
            return self.check_nested_match(self.template.body, func.body)

        def visit_call(self, call):
            if len(self.template.args) != len(call.args):
                return False
            if not self.check_nested_match(self.template.op, call.op):
                return False
            for i in range(len(call.args)):
                if not self.check_nested_match(self.template.args[i], call.args[i]):
                    return False
            return True

        def visit_tuple(self, tup):
            if len(self.template.fields) != len(tup.fields):
                return False
            for i in range(len(tup.fields)):
                if not self.check_nested_match(self.template.fields[i], tup.fields[i]):
                    return False
            return True

        def visit_if(self, if_expr):
            if not self.check_nested_match(self.template.cond, if_expr.cond):
                return False
            if not self.check_nested_match(self.template.true_branch, if_expr.true_branch):
                return False
            return self.check_nested_match(self.template.false_branch, if_expr.false_branch)

        def visit_tuple_getitem(self, tgi):
            if self.template.index != tgi.index:
                return False
            return self.check_nested_match(self.template.tuple_value, tgi.tuple_value)

        def check_nested_pattern(self, template_pattern, pattern):
            if isinstance(pattern, relay.PatternWildcard):
                return isinstance(template_pattern, relay.PatternWildcard)
            if isinstance(pattern, relay.PatternVar):
                if not isinstance(template_pattern, relay.PatternVar):
                    return False
                if (pattern.var in self.var_mapping
                    and self.var_mapping[pattern.var] != template_pattern.var):
                    return False
                self.var_mapping[pattern.var] = template_pattern.var
                return True
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
            if not self.check_nested_match(self.template.data, match.data):
                return False
            if len(self.template.clauses) != len(match.clauses):
                return False
            for i in range(len(match.clauses)):
                template_clause = self.template.clauses[i]
                clause = match.clauses[i]
                if not self.check_nested_pattern(template_clause.lhs, clause.lhs):
                    return False
                if not self.check_nested_match(template_clause.rhs, clause.rhs):
                    return False
            return True

        # punting on refs for now

    matcher = Matcher(template)
    res = matcher.check_direct_match(target)
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
        # we will use the order of the Relay free_vars pass
        # to determine the order in which pattern calls are made
        self.target_vars = free_vars(target)
        self.compiler_name = compiler_name
        self.composite_name = composite_name
        self.composite_counter = composite_counter

    def extract_target(self, match_args):
        """
        If we found a match for our target, this will
        produce a call to a BYOC-annotated version of the target
        with the pattern-arguments as args to the call

        Format:
        (fn(a1, ..., an, attrs={Compiler: compiler_name}) {
             (fn(b1, ..., bn, attrs={
                                     Composite: composite_name
                                     global_name: composite_name+counter
        }) {
                 target expression
                 # note: b1 ... bn are the free vars from the target
        })(a1, ..., an)
        })(match_args[0], ..., match_args[n-1])
        """
        assert all(map(lambda v: v in match_args, self.target_vars))
        match_ordering = [match_args[v] for v in self.target_vars]

        # we have to deduplicate vars for Relay's well-formedness check
        # (all var definitions must be unique)
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

    def visit(self, expr):
        """
        Whenever we encounter an expression,
        check if we find a match and insert it if we do;
        otherwise visit as before
        """
        # headache-saver: if it's a codegen function, we will ignore it
        if isinstance(expr, relay.Function):
            if expr.attrs is not None and ("Compiler" in expr.attrs or "Composite" in expr.attrs):
                return expr

        found_match, match_args = check_match(self.target, expr)
        if found_match:
            return self.extract_target(match_args)
        return super().visit(expr)

    # warning: will not work on refs because the matcher does not handle them

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
