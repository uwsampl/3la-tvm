import tvm
from tvm import relay

from tvm.relay.testing import annotate_exact_matches, deduplicate_vars, check_compiler_call

def assert_simple_cases(pattern, compiler_name, pattern_name):
    fresh_pattern = deduplicate_vars(pattern)

    self_match = annotate_exact_matches(fresh_pattern, pattern, compiler_name, pattern_name)
    assert check_compiler_call(self_match, pattern)

    a = relay.Var("a")
    plus = fresh_pattern + a
    plus_match = annotate_exact_matches(plus, pattern, compiler_name, pattern_name)
    assert isinstance(plus_match, relay.Call)
    assert plus_match.op.name == "add"
    assert plus_match.args[1] == a
    assert check_compiler_call(plus_match.args[0], pattern)

    in_func = relay.Function([], fresh_pattern)
    in_func_match = annotate_exact_matches(in_func, pattern, compiler_name, pattern_name)
    assert isinstance(in_func_match, relay.Function)
    assert len(in_func_match.params) == 0
    assert check_compiler_call(in_func_match.body, pattern)

    b = relay.Var("b")
    let = relay.Let(b, fresh_pattern, fresh_pattern + b)
    let_match = annotate_exact_matches(let, pattern, compiler_name, pattern_name)
    assert isinstance(let_match, relay.Let)
    assert check_compiler_call(let_match.value, pattern)
    assert isinstance(let_match.body, relay.Call)
    assert let_match.body.args[1] == b
    assert check_compiler_call(let_match.body.args[0], pattern)

    x, y, z = relay.Var("x"), relay.Var("y"), relay.Var("z")
    call = relay.Function([x, y, z], (x + y) * z)(a, fresh_pattern, b)
    call_match = annotate_exact_matches(call, pattern, compiler_name, pattern_name)
    assert isinstance(call_match, relay.Call)
    assert tvm.ir.structural_equal(call_match.op, call.op, True)
    assert len(call_match.args) == 3
    assert call_match.args[0] == a
    assert call_match.args[2] == b
    assert check_compiler_call(call_match.args[1], pattern)

    x, y = relay.Var("x"), relay.Var("y")
    tup = relay.Tuple([x, fresh_pattern, y])
    tup_match = annotate_exact_matches(tup, pattern, compiler_name, pattern_name)
    assert isinstance(tup_match, relay.Tuple)
    assert isinstance(tup_match.fields[0], relay.Var)
    assert isinstance(tup_match.fields[2], relay.Var)
    assert check_compiler_call(tup_match.fields[1], pattern)

    x, y, z, w = relay.Var("x"), relay.Var("y"), relay.Var("z"), relay.Var("w")
    match_clause = relay.Match(x, [
        relay.Clause(relay.PatternWildcard(), fresh_pattern),
        relay.Clause(relay.PatternVar(y), y),
        relay.Clause(relay.PatternTuple([
            relay.PatternVar(z), relay.PatternVar(w)
        ]), fresh_pattern)
    ])
    match_clause_match = annotate_exact_matches(match_clause, pattern, compiler_name, pattern_name)
    assert isinstance(match_clause_match, relay.Match)
    assert len(match_clause_match.clauses) == 3
    assert isinstance(match_clause_match.clauses[0].lhs, relay.PatternWildcard)
    assert check_compiler_call(match_clause_match.clauses[0].rhs, pattern)
    assert tvm.ir.structural_equal(
        match_clause.clauses[1],
        match_clause_match.clauses[1],
        True)
    assert tvm.ir.structural_equal(
        match_clause.clauses[2].lhs,
        match_clause_match.clauses[2].lhs,
        True)
    assert check_compiler_call(match_clause_match.clauses[2].rhs, pattern)

    ref = relay.RefCreate(fresh_pattern)
    ref_match = annotate_exact_matches(ref, pattern, compiler_name, pattern_name)
    assert isinstance(ref_match, relay.RefCreate)
    assert check_compiler_call(ref_match.value, pattern)


def test_match_misses():
    pattern = relay.nn.dense(relay.Var("v"), relay.Var("w"))
    x, y, z, a = relay.Var("x"), relay.Var("y"), relay.Var("z"), relay.Var("a")
    progs = [
        relay.const(1) + relay.const(2),
        relay.Function([x, y, z], relay.Let(a, x + y, a*z)),
        relay.Tuple([]),
        relay.Tuple([relay.nn.bias_add(x, y), relay.transpose(z)])
    ]
    # no match -> don't do anything
    for prog in progs:
        new_prog = annotate_exact_matches(prog, pattern, "MyCompiler", "Dense")
        assert tvm.ir.structural_equal(prog, new_prog), (prog, new_prog)


def test_operator_simple_match():
    pattern = relay.nn.dense(relay.Var("v"), relay.Var("w"))
    assert_simple_cases(pattern, "MyCompiler", "Dense")


def test_nested_operator_match():
    pattern = relay.nn.bias_add(relay.nn.dense(relay.Var("x"), relay.Var("y")), relay.Var("z"))
    assert_simple_cases(pattern, "MyCompiler", "DenseAddBias")


def test_call_match():
    x, y, w, z = relay.Var("x"), relay.Var("y"), relay.Var("w"), relay.Var("z")
    pattern = relay.Function([x, y], x + y)(w, z)
    assert_simple_cases(pattern, "MyCompiler", "LiteralSum")


def test_let_match():
    x = relay.Var("x")
    pattern = relay.Let(x, relay.const(2), x*x)
    assert_simple_cases(pattern, "MyCompiler", "Square")


def test_tuple_match():
    x, y, w, z = relay.Var("x"), relay.Var("y"), relay.Var("w"), relay.Var("z")
    pattern = relay.Tuple([x, y + z, w])
    assert_simple_cases(pattern, "MyCompiler", "TupleSum")


def test_nested_function_match():
    # going to match a function literal with no pattern vars
    x, y, z = relay.Var("x"), relay.Var("y"), relay.Var("z")
    pattern = relay.Function([x, y, z], (x+y)*z)

    fresh_func = deduplicate_vars(pattern)
    f = relay.Var("f")
    let_def = relay.Let(f, fresh_func, f(relay.const(1), relay.const(2), relay.const(3)))
    let_def_match = annotate_exact_matches(let_def, pattern, "MyCompiler", "FuncLiteral")
    assert isinstance(let_def_match, relay.Let)
    assert check_compiler_call(let_def_match.value, pattern)
    # no pattern vars!
    assert len(let_def_match.value.args) == 0
    assert tvm.ir.structural_equal(let_def_match.body, let_def.body, True)


def test_separate_matches():
    # match two separate patterns in different places
    dense_pattern = relay.nn.dense(relay.Var("x"), relay.Var("y"))
    add_pattern = relay.Var("w") + relay.Var("z")

    a, b = relay.Var("a"), relay.Var("b")
    chain = relay.Let(a, relay.const(0) + relay.const(1),
                      relay.Let(b, relay.const(2) + relay.const(3),
                                relay.nn.dense(a, b)))
    match_one = annotate_exact_matches(chain, dense_pattern, "MyCompiler", "Dense")
    chain_match = annotate_exact_matches(match_one, add_pattern, "MyCompiler", "Add")

    assert isinstance(chain_match, relay.Let)
    assert check_compiler_call(chain_match.value, add_pattern)
    assert all(map(lambda c: isinstance(c, relay.Constant), chain_match.value.args))
    next_let = chain_match.body
    assert isinstance(next_let, relay.Let)
    assert check_compiler_call(next_let.value, add_pattern)
    assert all(map(lambda c: isinstance(c, relay.Constant), next_let.value.args))
    assert check_compiler_call(next_let.body, dense_pattern)


def test_nested_matches():
    # match two separate patterns where one is an arg to the other
    dense_pattern = relay.nn.dense(relay.Var("x"), relay.Var("y"))
    add_pattern = relay.Var("w") + relay.Var("z")

    a, b, c, d = relay.Var("a"), relay.Var("b"), relay.Var('c'), relay.Var("d")
    var_set = set([a, b, c, d])
    call = relay.nn.dense(a + b, c + d)

    # let's try both orders
    just_dense = annotate_exact_matches(call, dense_pattern, "MyCompiler", "Dense")
    dense_add = annotate_exact_matches(just_dense, add_pattern, "MyCompiler", "Add")

    just_add = annotate_exact_matches(call, add_pattern, "MyCompiler", "Add")
    add_dense = annotate_exact_matches(just_add, dense_pattern, "MyCompiler", "Dense")

    for match in (dense_add, add_dense):
        assert check_compiler_call(match, dense_pattern)
        for arg in match.args:
            assert check_compiler_call(arg, add_pattern)
            assert all(map(lambda v: v in var_set, arg.args))


def test_internal_matches_blocked():
    # if we match a bigger pattern, instances of the smaller pattern inside it will not do anything
    dense_add_bias_pattern = relay.nn.bias_add(relay.nn.dense(relay.Var("x"), relay.Var("y")), relay.Var("z"))
    dense_pattern = relay.nn.dense(relay.Var("x"), relay.Var("y"))

    a, b, c = relay.Var("a"), relay.Var("b"), relay.Var("c")
    expr = relay.nn.bias_add(relay.nn.dense(a, b), c)
    expr_match = annotate_exact_matches(expr, dense_add_bias_pattern, "MyCompiler", "DenseAddBias")
    assert check_compiler_call(expr_match, dense_add_bias_pattern)

    # now calling the smaller pattern should do nothing
    expr_new_match = annotate_exact_matches(expr_match, dense_pattern, "MyCompiler", "Dense")
    assert tvm.ir.structural_equal(expr_new_match, expr_match, True)


def test_no_improper_capture():
    # if a pattern var matches an internally bound variable, we should not bring it out
    x, y, z, w = relay.Var("x"), relay.Var("y"), relay.Var("z"), relay.Var("w")
    # pattern vars: y, z
    weird_pattern = relay.Let(x, y, relay.Let(z, w, z))

    a, b = relay.Var("a"), relay.Var("b")
    # matching the pattern does not break any bindings
    okay = relay.Let(a, relay.const(0), relay.Let(b, relay.const(1), b))
    okay_match = annotate_exact_matches(okay, weird_pattern, "MyCompiler", "Let")
    assert check_compiler_call(okay_match, weird_pattern)
    assert len(okay_match.args) == 2
    assert isinstance(okay_match.args[0], relay.Constant)
    assert isinstance(okay_match.args[1], relay.Constant)

    # matching here would expose a bound var
    bad = relay.Let(a, relay.const(0), relay.Let(b, a, b))
    bad_match = annotate_exact_matches(bad, weird_pattern, "MyCompiler", "Let")
    # no change
    assert tvm.ir.structural_equal(bad, bad_match, True)


def test_match_whole_match_block():
    # x, z, and w are pattern vars but y is bound
    x, y, z, w = relay.Var("x"), relay.Var("y"), relay.Var("z"), relay.Var("w")
    pattern = relay.Match(x, [
        relay.Clause(
            relay.PatternTuple([
                relay.PatternVar(y),
                relay.PatternWildcard()
            ]),
            relay.Tuple([y, z])),
        relay.Clause(relay.PatternWildcard(), w)
    ])

    a = relay.Var("a")
    b = relay.Var("b")
    instance = relay.Let(
        b,
        relay.Match(relay.Tuple([relay.const(1), relay.const(2)]), [
            relay.Clause(
                relay.PatternTuple([
                    relay.PatternVar(a),
                    relay.PatternWildcard()
                ]),
                relay.Tuple([a, relay.Tuple([])])),
            relay.Clause(relay.PatternWildcard(),
                         relay.Tuple([relay.const(3), relay.Tuple([])]))
        ]),
        relay.TupleGetItem(b, 0))

    match = annotate_exact_matches(instance, pattern, "MyCompiler", "Match")
    assert isinstance(match, relay.Let)
    assert check_compiler_call(match.value, pattern)
    assert isinstance(match.body, relay.TupleGetItem)

    assert_simple_cases(pattern, "MyCompiler", "MatchBlock")


def test_inconsistent_match():
    x, y = relay.Var("x"), relay.Var("y")
    pattern = relay.Tuple([x, y, x])

    # template var is assigned inconsistently
    a, b, c = relay.Var("a"), relay.Var("b"), relay.Var("c")
    bad_match = relay.Tuple([a, b, c])
    result = annotate_exact_matches(bad_match, pattern, "MyCompiler", "Tuple")
    assert not check_compiler_call(result, pattern)
    assert tvm.ir.structural_equal(result, bad_match, True)


def test_ref_match():
    r, s = relay.Var("r"), relay.Var("s")
    pattern = relay.RefCreate(relay.Tuple([r, s]))
    assert_simple_cases(pattern, "MyCompiler", "RefTuple")

    read_pattern = relay.RefRead(r)
    assert_simple_cases(read_pattern, "MyCompiler", "RefRead")

    write_pattern = relay.RefWrite(r, s)
    assert_simple_cases(read_pattern, "MyCompiler", "RefWrite")


def test_multiple_matches():
    # we have multiple instances of a pattern
    # and the matcher should find all matches
    def make_linear(d, w, b):
        return relay.nn.bias_add(relay.nn.dense(d, w), b)

    x, y, z = relay.Var("x"), relay.Var("y"), relay.Var("z")
    pattern = make_linear(x, y, z)

    expr = make_linear(
        make_linear(relay.const(1), relay.const(2), relay.const(3)),
        make_linear(relay.const(4), relay.const(5), relay.const(6)),
        relay.const(7))
    lin_match = annotate_exact_matches(expr, pattern, "MyCompiler", "Let")
    assert isinstance(lin_match, relay.Call)
    assert len(lin_match.args) == 3
    assert check_compiler_call(lin_match, pattern)

    # the first two arguments should also be arguments
    assert check_compiler_call(lin_match.args[0], pattern)
    assert check_compiler_call(lin_match.args[1], pattern)
    assert not check_compiler_call(lin_match.args[2], pattern)


if __name__ == "__main__":
    test_match_misses()
    test_operator_simple_match()
    test_nested_operator_match()
    test_call_match()
    test_let_match()
    test_tuple_match()
    test_nested_function_match()
    test_separate_matches()
    test_nested_matches()
    test_internal_matches_blocked()
    test_no_improper_capture()
    test_match_whole_match_block()
    test_inconsistent_match()
    test_ref_match()
    test_multiple_matches()
