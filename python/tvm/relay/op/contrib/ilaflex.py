import tvm
from tvm import relay
import tvm.ir
from ...dataflow_pattern import wildcard, is_op
from .register import register_pattern_table

def relay_lstm_cell(batch_size, input_size, hidden_size):
    # based on https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
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


def lstm_definition(batch_size, input_size, hidden_size, time_steps,
                    mod, time_axis=1):
    state_tensor_type = relay.TensorType((batch_size, hidden_size))
    state_tuple_type = relay.TupleType([state_tensor_type, state_tensor_type])

    input_var = relay.var("input", shape=(batch_size, time_steps, input_size))
    state_var = relay.var("state", type_annotation=state_tuple_type)
    i2h_weight_var = relay.var("i2h_weight", shape=(4*hidden_size, input_size))
    h2h_weight_var = relay.var("h2h_weight", shape=(4*hidden_size, hidden_size))
    i2h_bias_var = relay.var("i2h_bias", shape=(4*hidden_size,))
    h2h_bias_var = relay.var("h2h_bias", shape=(4*hidden_size,))

    # in this case, we are ignoring the state outputs
    builder = relay.ScopeBuilder()
    cell_var = builder.let("lstm_cell", relay_lstm_cell(batch_size, input_size, hidden_size))
    splits = builder.let("splits", relay.split(input_var, time_steps, time_axis).astuple())
    last_state = state_var
    seq_outs = []
    for i in range(time_steps):
        squeezed = builder.let(f"squeezed_{i}", relay.squeeze(relay.TupleGetItem(splits, i), axis=[time_axis]))
        cell_out = builder.let(f"cell_out_{i}",
                               cell_var(squeezed, last_state,
                                        i2h_weight_var, h2h_weight_var,
                                        i2h_bias_var, i2h_bias_var))
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
    # builder.ret(relay.Tuple([stacked, reshape_hidden, reshape_cell]))
    # for simplicity, we will return only the hidden state
    # builder.ret(reshape_hidden)
    builder.ret(stacked)

    # Ideally, we would want to return all outputs;
    # for now, for simplicity, we will only return one
    #
    # ret_type = relay.TupleType([
    #     relay.TensorType((batch_size, time_steps, hidden_size)),
    #     relay.TensorType((1, batch_size, hidden_size)),
    #     relay.TensorType((1, batch_size, hidden_size))
    # ])
    # ret_type = relay.TensorType((1, batch_size, hidden_size))
    ret_type = relay.TensorType((batch_size, time_steps, hidden_size))

    return relay.Function([input_var, state_var, i2h_weight_var, h2h_weight_var,
                           i2h_bias_var, h2h_bias_var],
                          builder.get(),
                          ret_type=ret_type)


def create_lstm_call(mod, lstm_input, initial_state,
                     i2h_weight, h2h_weight, bias,
                     batch_size, input_size, hidden_size, time_steps):
    """
    Given a module, adds a FlexNLP hook definition if it is not present
    and returns a call using the given arguments

    For now, the tensor sizes, etc, will be included directly in the generated name
    though we can make the definitions parametric later
    """
    # we unroll for a different number of time steps -- in principle, we can make a single definition
    # that will use a Relay list and not need to unroll
    name = f"ILALSTM_{batch_size}_{input_size}_{hidden_size}_{time_steps}"

    lstm_var = relay.Var(name)
    lstm_def = lstm_definition(batch_size, input_size, hidden_size, time_steps, mod)

    # Wrap in a function manually annotated with the compiler attribute.
    f_input, f_state = relay.Var("input"), relay.Var("state")
    f_i2h, f_h2h, f_bias = relay.Var("i2h_weight"), relay.Var("h2h_weight"), relay.Var("bias")

    # we give this function a composite attribute
    # so the codegen recognizes it
    lstm_wrapper = relay.Function(
        [f_input, f_state, f_i2h, f_h2h, f_bias],
        relay.Let(lstm_var, lstm_def,
                  lstm_var(f_input, f_state, f_i2h, f_h2h,
                           # note: zeroing out one of the bias inputs like Keras
                           f_bias, relay.zeros_like(f_bias))))
    lstm_wrapper = lstm_wrapper.with_attr("Composite", "ilaflex.lstm")

    # now we create an outer call with the compiler attribute
    # so that it is passed to the codegen (with the call to the lstm_wrapper)
    outer_vars = [relay.Var(f"v{i}") for i in range(5)]
    outer_wrapper = relay.Function(
        outer_vars, lstm_wrapper(*outer_vars))

    # ugly hack: we need a counter to make sure the global identifier will be unique
    if not hasattr(create_lstm_call, "symbol_count"):
        create_lstm_call.symbol_count = 0

    # See tests/python/relay/test_external_codegen.py:set_external_func_attr
    outer_wrapper = outer_wrapper.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
    outer_wrapper = outer_wrapper.with_attr("Compiler", "ilaflex")
    outer_wrapper = outer_wrapper.with_attr("global_symbol", f"ilaflex.lstm_{create_lstm_call.symbol_count}")
    create_lstm_call.symbol_count += 1

    f = relay.Var("f")
    return relay.Let(f, outer_wrapper,
                     f(lstm_input, initial_state,
                       i2h_weight, h2h_weight, bias))

def make_dot_attention():
    linear_key = wildcard()
    query = wildcard()
    bmm = is_op('nn.batch_matmul')(linear_key, query)
    prod = is_op('transpose')(bmm)
    scores = is_op('nn.softmax')(prod)
    return scores


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
    attention_pat = ("ilflex.attention", make_dot_attention())
    ilaflex_patterns = [linear_pat, attention_pat]
    return ilaflex_patterns
