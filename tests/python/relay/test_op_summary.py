import tvm
from tvm import relay
from tvm.relay.testing import count_all_ops, count_all_overloads, count_all_ops_in_overloads
from tvm.relay.testing import annotate_exact_matches, deduplicate_vars

def test_count_chain():
    mod = tvm.IRModule()
    x = relay.Var("x")
    y = relay.Var("y")
    z = relay.Var("z")
    w = relay.Var("w")
    mod["main"] = relay.Function([x, y, z, w], relay.nn.conv2d(x + z, w*y))
    assert count_all_ops(mod) == 3


def test_count_multiple_funcs():
    mod = tvm.IRModule()
    x = relay.Var("x")
    y = relay.Var("y")
    z = relay.Var("z")
    w = relay.Var("w")
    gv = relay.GlobalVar("f2")
    mod["main"] = relay.Function([x, y, z, w], relay.nn.conv2d(x + z, gv(z, w)))
    a = relay.Var("a")
    b = relay.Var("b")
    mod[gv] = relay.Function([a, b], a*b)
    assert count_all_ops(mod) == 3


def test_count_single_overload():
    x = relay.Var("x")
    notnot = relay.logical_not(relay.logical_not(x))

    mod = tvm.IRModule()
    mod["main"] = annotate_exact_matches(
        relay.Function([x], notnot),
        deduplicate_vars(notnot),
        "MyCompiler", "notnot")

    assert count_all_overloads(mod) == 1
    assert count_all_ops_in_overloads(mod) == 2


def test_count_multiple_overloads():
    x = relay.Var("x")
    y = relay.Var("y")
    conv = relay.nn.conv2d(x, y)
    add = x + y

    mod = tvm.IRModule()
    a = relay.Var("a")
    b = relay.Var("b")
    c = relay.Var("c")
    match_conv = annotate_exact_matches(
        relay.Function([a, b, c], relay.nn.conv2d(a + b, c)),
        conv,
        "MyCompiler", "conv"
    )
    match_add = annotate_exact_matches(
        match_conv,
        add,
        "MyCompiler", "add")
    mod["main"] = match_add
    assert count_all_overloads(mod) == 2
    assert count_all_ops_in_overloads(mod) == 2


def test_count_overloads_multiple_funcs():
    x, y, z = relay.Var("x"), relay.Var("y"), relay.Var("z")
    linear_layer = relay.nn.bias_add(relay.nn.dense(x, y), z)
    conv = relay.nn.conv2d(x, y)

    mod = tvm.IRModule()

    a, b, c = relay.Var("a"), relay.Var("b"), relay.Var("c")
    lin_func = relay.Function([a, b, c],
                              relay.nn.bias_add(relay.nn.dense(a, b), c))
    match_lin = annotate_exact_matches(lin_func, linear_layer, "MyCompiler", "linear")

    linear_var = relay.GlobalVar("linear_layer")
    mod[linear_var] = match_lin

    d, e, f, g = relay.Var("d"), relay.Var("e"), relay.Var("f"), relay.Var("g")
    main_func = relay.Function([d, e, f, g],
                               relay.nn.conv2d(linear_var(d, e, f), g))
    match_conv = annotate_exact_matches(main_func, conv, "MyCompiler", "Conv")
    mod["main"] = match_conv

    assert count_all_overloads(mod) == 2
    assert count_all_ops_in_overloads(mod) == 3


if __name__ == "__main__":
    test_count_chain()
    test_count_multiple_funcs()
    test_count_single_overload()
    test_count_multiple_overloads()
    test_count_overloads_multiple_funcs()
