import numpy as np
import tvm
from tvm import relay
from tvm.relay.testing import annotate_exact_matches, check_compiler_call

def relay_lstm_cell(batch_size, input_size, hidden_size):
    state_tensor_type = relay.TensorType((batch_size, hidden_size))
    state_tuple_type = relay.TupleType([state_tensor_type, state_tensor_type])

    inp = relay.var("input", shape=(batch_size, input_size))
    state = relay.Var("state", type_annotation=state_tuple_type)

    w_ih = relay.var("w_ih", shape=(4*hidden_size, input_size))
    w_hh = relay.var("w_hh", shape=(4*hidden_size, hidden_size))
    b_ih = relay.var("b_ih", shape=(4*hidden_size,))
    b_hh = relay.var("b_hh", shape=(4*hidden_size,))

    hidden = relay.TupleGetItem(state, 0)
    cell_state = relay.TupleGetItem(state, 1)

    # PyTorch packs the i2h and h2h weights and biases together so we will match that here
    w_i_splits = relay.split(w_ih, 4, 0)
    w_h_splits = relay.split(w_hh, 4, 0)
    b_i_splits = relay.split(b_ih, 4, 0)
    b_h_splits = relay.split(b_hh, 4, 0)
    w_ii, w_if, w_ig, w_io = w_i_splits[0], w_i_splits[1], w_i_splits[2], w_i_splits[3]
    w_hi, w_hf, w_hg, w_ho = w_h_splits[0], w_h_splits[1], w_h_splits[2], w_h_splits[3]
    b_ii, b_if, b_ig, b_io = b_i_splits[0], b_i_splits[1], b_i_splits[2], b_i_splits[3]
    b_hi, b_hf, b_hg, b_ho = b_h_splits[0], b_h_splits[1], b_h_splits[2], b_h_splits[3]

    def weighted_value(weight, value, bias):
        return relay.transpose(relay.nn.dense(weight, value) + relay.reshape(bias, (hidden_size, 1)))

    i_t = relay.sigmoid(weighted_value(w_ii, inp, b_ii) + weighted_value(w_hi, hidden, b_hi))
    f_t = relay.sigmoid(weighted_value(w_if, inp, b_if) + weighted_value(w_hf, hidden, b_hf))
    g_t = relay.tanh(weighted_value(w_ig, inp, b_ig) + weighted_value(w_hg, hidden, b_hg))
    o_t = relay.sigmoid(weighted_value(w_io, inp, b_io) + weighted_value(w_ho, hidden, b_ho))
    c_t = f_t*cell_state + i_t*g_t
    h_t = o_t*relay.tanh(c_t)

    h_var = relay.Var("h")
    c_var = relay.Var("c")
    return relay.Function([inp, state, w_ih, w_hh, b_ih, b_hh],
                          relay.Let(h_var, h_t,
                                    relay.Let(c_var, c_t,
                                              relay.Tuple([h_var, relay.Tuple([h_var, c_var])]))),
                          ret_type=relay.TupleType([state_tensor_type, state_tuple_type]))


def lstm_body(data, state, i2h_weight, h2h_weight, i2h_bias, h2h_bias,
              batch_size, input_size, hidden_size, time_steps, time_axis=1):
    builder = relay.ScopeBuilder()
    cell = builder.let("lstm_cell", relay_lstm_cell(batch_size, input_size, hidden_size))
    splits = builder.let("splits", relay.split(data, time_steps, time_axis).astuple())
    last_state = state
    seq_outs = []
    for i in range(time_steps):
        squeezed = builder.let(f"squeezed_{i}", relay.squeeze(relay.TupleGetItem(splits, i), axis=[time_axis]))
        cell_out = builder.let(f"cell_out_{i}",
                               cell(squeezed, last_state,
                                    i2h_weight, h2h_weight,
                                    i2h_bias, i2h_bias))
        new_seq_out = builder.let(f"seq_out_{i}", relay.TupleGetItem(cell_out, 0))
        seq_outs.append(new_seq_out)
        new_hidden = builder.let(f"state_update_{i}", relay.TupleGetItem(cell_out, 1))
        last_state = new_hidden

    stacked = builder.let("stacked", relay.stack(seq_outs, axis=time_axis))
    # finally reshape to match pytorch's semantics (one layer)
    reshape_hidden = builder.let("final_hidden",
                                 relay.reshape(relay.TupleGetItem(last_state, 0),
                                               (1, batch_size, hidden_size)))
    reshape_cell = builder.let("final_cell",
                               relay.reshape(relay.TupleGetItem(last_state, 1),
                                             (1, batch_size, hidden_size)))
    builder.ret(relay.Tuple([stacked, reshape_hidden, reshape_cell]))
    return builder.get()


def lstm_definition(batch_size, input_size, hidden_size, time_steps,
                    time_axis=1):
    """
    Wrap the LSTM body in a function
    """
    state_tensor_type = relay.TensorType((batch_size, hidden_size))
    state_tuple_type = relay.TupleType([state_tensor_type, state_tensor_type])

    input_var = relay.var("input", shape=(batch_size, time_steps, input_size))
    state_var = relay.var("state", type_annotation=state_tuple_type)
    i2h_weight_var = relay.var("i2h_weight", shape=(4*hidden_size, input_size))
    h2h_weight_var = relay.var("h2h_weight", shape=(4*hidden_size, hidden_size))
    i2h_bias_var = relay.var("i2h_bias", shape=(4*hidden_size,))
    h2h_bias_var = relay.var("h2h_bias", shape=(4*hidden_size,))

    ret_type = relay.TupleType([
        relay.TensorType((batch_size, time_steps, hidden_size)),
        relay.TensorType((1, batch_size, hidden_size)),
        relay.TensorType((1, batch_size, hidden_size))
    ])

    return relay.Function(
        [input_var, state_var, i2h_weight_var, h2h_weight_var,
         i2h_bias_var, h2h_bias_var],
        lstm_body(input_var, state_var,
                  i2h_weight_var, h2h_weight_var, i2h_bias_var, h2h_bias_var,
                  batch_size, input_size, hidden_size, time_steps, time_axis=time_axis),
        ret_type=ret_type)


def linear_body(data, weight, bias):
    return relay.nn.bias_add(relay.nn.dense(data, weight), bias)


def linear_layer_definition(time_steps, hidden_size, dense_dim):
    input_var = relay.var("input", shape=(time_steps, hidden_size))
    weight_var = relay.var("weight", shape=(dense_dim, hidden_size))
    bias_var = relay.var("bias", shape=(dense_dim,))

    return relay.Function([input_var, weight_var, bias_var],
                          linear_body(input_var, weight_var, bias_var),
                          ret_type=relay.TensorType((time_steps, dense_dim)))


def time_distribute(in_expr, time_steps, construct_body, time_axis=1):
    splits = relay.split(in_expr, time_steps, time_axis)
    rebuilt = [
        construct_body(relay.squeeze(split, axis=[time_axis]))
        for split in splits
    ]
    return relay.stack(rebuilt, axis=time_axis)


def test_lstm_function_match():
    """
    Version where we define functions to handle the LSTM and linear layer
    and match on the *functions*
    (all calls to those functions will go through our codegen)
    """
    batch_size, hidden_size, dense_dim = 1, 64, 64
    input_size, time_steps = 256, 6
    linear_pattern = linear_layer_definition(time_steps, hidden_size, dense_dim)
    lstm_pattern = lstm_definition(batch_size, input_size, hidden_size, time_steps)

    builder = relay.ScopeBuilder()
    lstm_input = relay.Var("lstm_in")
    state = relay.Var("lstm_state")
    i2h_weight = relay.Var("i2h_weight")
    h2h_weight = relay.Var("h2h_weight")
    i2h_bias = relay.Var("i2h_bias")
    h2h_bias = relay.Var("h2h_bias")
    linear_weight = relay.Var("linear_weight")
    linear_bias = relay.Var("linear_bias")

    lstm_var = builder.let("lstm", lstm_definition(batch_size, input_size, hidden_size, time_steps))
    linear_var = builder.let(
        "linear",
        linear_layer_definition(batch_size, hidden_size, dense_dim))
    lstm_res = builder.let(
        "seq_out",
        relay.TupleGetItem(
            lstm_var(
                lstm_input, state, i2h_weight, h2h_weight, i2h_bias, h2h_bias),
            0))
    linear_res = builder.let(
        "linear_out",
        # squeeze away batch size
        linear_var(relay.squeeze(lstm_res, axis=[0]),
                   linear_weight, linear_bias))
        # time_distribute(
        #     lstm_res, time_steps,
        #     lambda split: linear_var(split, linear_weight, linear_bias)))
    builder.ret(relay.nn.softmax(linear_res))

    speech_to_text = builder.get()

    match_lstm = annotate_exact_matches(speech_to_text, lstm_pattern, "ilaflex", "ilaflex.lstm")
    match_linear = annotate_exact_matches(match_lstm, linear_pattern, "ilaflex", "ilaflex.linear")

    try:
        # just check that it type-checks
        relay.transform.InferType()(tvm.IRModule.from_expr(match_linear))
    except:
        assert False, f"{match_linear} failed to type check"

    assert isinstance(match_linear, relay.Let)
    assert check_compiler_call(match_linear.value, lstm_pattern)
    inner_def = match_linear.body
    assert isinstance(inner_def, relay.Let)
    assert check_compiler_call(inner_def.value, linear_pattern)


def test_lstm_body_match():
    """
    Version where we define functions to handle the LSTM and linear layer
    and match on the *bodies* directly inline
    """
    batch_size, hidden_size, dense_dim = 1, 64, 64
    input_size, time_steps = 256, 6
    # we take the bodies, so the free args in there will be pattern vars
    linear_pattern = linear_layer_definition(time_steps, hidden_size, dense_dim).body
    lstm_pattern = lstm_definition(batch_size, input_size, hidden_size, time_steps).body

    builder = relay.ScopeBuilder()
    lstm_input = relay.Var("lstm_in")
    state = relay.Var("lstm_state")
    i2h_weight = relay.Var("i2h_weight")
    h2h_weight = relay.Var("h2h_weight")
    i2h_bias = relay.Var("i2h_bias")
    h2h_bias = relay.Var("h2h_bias")
    linear_weight = relay.Var("linear_weight")
    linear_bias = relay.Var("linear_bias")

    # use the bodies directly
    lstm_var = builder.let("lstm_out", lstm_body(
        lstm_input, state, i2h_weight, h2h_weight, i2h_bias, h2h_bias,
        batch_size, input_size, hidden_size, time_steps))
    linear_res = builder.let(
        "linear",
        linear_body(relay.squeeze(lstm_var, axis=[0]), linear_weight, linear_bias))
    builder.ret(relay.nn.softmax(linear_res))

    speech_to_text = builder.get()

    match_lstm = annotate_exact_matches(speech_to_text, lstm_pattern, "ilaflex", "ilaflex.lstm")
    match_linear = annotate_exact_matches(match_lstm, linear_pattern, "ilaflex", "ilaflex.linear")

    try:
        # just check that it type-checks
        relay.transform.InferType()(tvm.IRModule.from_expr(match_linear))
    except:
        assert False, f"{match_linear} failed to type check"

    assert isinstance(match_linear, relay.Let)
    assert check_compiler_call(match_linear.value, lstm_pattern)
    inner_def = match_linear.body
    assert isinstance(inner_def, relay.Let)
    assert check_compiler_call(inner_def.value, linear_pattern)


if __name__ == "__main__":
    test_lstm_function_match()
    test_lstm_body_match()
