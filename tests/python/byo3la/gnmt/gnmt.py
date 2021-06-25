"""
A GNMT implementation in Relay, based on https://ngc.nvidia.com/catalog/resources/nvidia:gnmt_v2_for_tensorflow
"""

import numpy as np
import tvm
from tvm import relay

from attention import luong_general_attention

# TODO: We should factor out the LSTM definitions that
# we are also using for the speech-to-text model

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


def bilstm_body(data, state,
                i2h_weight, h2h_weight, i2h_bias, h2h_bias,
                batch_size, input_size, hidden_size, time_steps, time_axis=1):
    builder = relay.ScopeBuilder()
    cell = builder.let("lstm_cell", relay_lstm_cell(batch_size, input_size, hidden_size))

    # split state along second dimension
    init_hidden = relay.TupleGetItem(state, 0)
    init_cell = relay.TupleGetItem(state, 1)
    split_hidden = builder.let("split_hidden", relay.split(init_hidden, 2, 0).astuple())
    split_cell = builder.let("split_cell", relay.split(init_cell, 2, 0).astuple())

    fwd_state = builder.let("fwd_state", relay.Tuple([
        relay.squeeze(relay.TupleGetItem(split_hidden, 0), axis=[0]),
        relay.squeeze(relay.TupleGetItem(split_cell, 0), axis=[0])
    ]))
    bwd_state = builder.let("bwd_state", relay.Tuple([
        relay.squeeze(relay.TupleGetItem(split_hidden, 1), axis=[0]),
        relay.squeeze(relay.TupleGetItem(split_cell, 1), axis=[0])
    ]))

    splits = builder.let("splits", relay.split(data, time_steps, time_axis).astuple())
    squeezed_splits = []
    for i in range(time_steps):
        squeezed = builder.let(f"squeezed_{i}", relay.squeeze(relay.TupleGetItem(splits, i), axis=[time_axis]))
        squeezed_splits.append(squeezed)

    def lstm_loop(input_data, init_state, prefix):
        splits = builder.let(f"{prefix}_splits", relay.split(input_data, time_steps, axis=time_axis).astuple())

        last_state = init_state
        seq_outs = []
        for i in range(time_steps):
            squeezed = builder.let(f"{prefix}_squeezed_{i}", relay.squeeze(relay.TupleGetItem(splits, i), axis=[time_axis]))
            cell_out = builder.let(f"{prefix}_cell_out_{i}",
                                   cell(squeezed, last_state,
                                        i2h_weight, h2h_weight,
                                        i2h_bias, h2h_bias))
            new_seq_out = builder.let(f"{prefix}_seq_out_{i}", relay.TupleGetItem(cell_out, 0))
            seq_outs.append(new_seq_out)
            new_hidden = builder.let(f"{prefix}_state_update_{i}", relay.TupleGetItem(cell_out, 1))
            last_state = new_hidden

        stacked = builder.let(f"{prefix}_stacked", relay.stack(seq_outs, axis=time_axis))
        # finally reshape to match pytorch's semantics (one layer)
        reshape_hidden = builder.let(f"{prefix}_final_hidden",
                                     relay.reshape(relay.TupleGetItem(last_state, 0),
                                                   (1, batch_size, hidden_size)))
        reshape_cell = builder.let(f"{prefix}_final_cell",
                                   relay.reshape(relay.TupleGetItem(last_state, 1),
                                                 (1, batch_size, hidden_size)))
        return stacked, reshape_hidden, reshape_cell

    fwd_seq, fwd_hidden, fwd_cell = lstm_loop(data, fwd_state, "fwd")
    rev_data = relay.reverse(data, time_axis)
    bwd_seq, bwd_hidden, bwd_cell = lstm_loop(rev_data, bwd_state, "bwd")
    bwd_seq = relay.reverse(bwd_seq, time_axis)

    # concat outputs along hidden size dimension, concat hidden states along layer dimension
    final_seq = relay.concatenate([fwd_seq, bwd_seq], 2)
    final_hidden = relay.concatenate([fwd_hidden, bwd_hidden], 0)
    final_cell = relay.concatenate([fwd_cell, bwd_cell], 0)
    builder.ret(relay.Tuple([final_seq, final_hidden, final_cell]))
    return builder.get()


def bilstm_definition(batch_size, input_size, hidden_size, time_steps,
                      time_axis=1):
    """
    Wrap the BiLSTM body in a function
    """
    state_tensor_type = relay.TensorType((2, batch_size, hidden_size))
    state_tuple_type = relay.TupleType([state_tensor_type, state_tensor_type])

    input_var = relay.var("input", shape=(batch_size, time_steps, input_size))
    state_var = relay.var("state", type_annotation=state_tuple_type)
    i2h_weight_var = relay.var("i2h_weight", shape=(4*hidden_size, input_size))
    h2h_weight_var = relay.var("h2h_weight", shape=(4*hidden_size, hidden_size))
    i2h_bias_var = relay.var("i2h_bias", shape=(4*hidden_size,))
    h2h_bias_var = relay.var("h2h_bias", shape=(4*hidden_size,))

    ret_type = relay.TupleType([
        relay.TensorType((batch_size, time_steps, 2*hidden_size)),
        relay.TensorType((2, batch_size, hidden_size)),
        relay.TensorType((2, batch_size, hidden_size))
    ])

    return relay.Function(
        [input_var, state_var, i2h_weight_var, h2h_weight_var,
         i2h_bias_var, h2h_bias_var],
        bilstm_body(input_var, state_var,
                    i2h_weight_var, h2h_weight_var, i2h_bias_var, h2h_bias_var,
                    batch_size, input_size, hidden_size, time_steps, time_axis=time_axis),
        ret_type=ret_type)


def layered_lstm_body(data, state,
                      i2h_weights, h2h_weights, i2h_biases, h2h_biases,
                      batch_size, input_size, hidden_size,
                      time_steps, layers, time_axis=1):
    # split state by number of layers
    # assign weights to layers
    # lstm def for layer 1, lstm def for subsequent layers
    builder = relay.ScopeBuilder()
    first_layer = builder.let("first_lstm", lstm_definition(batch_size, input_size, hidden_size, time_steps, time_axis=time_axis))
    subsequent_layers = builder.let("next_lstm", lstm_definition(batch_size, hidden_size, hidden_size, time_steps, time_axis=time_axis))

    init_hidden = relay.TupleGetItem(state, 0)
    init_cell = relay.TupleGetItem(state, 1)
    split_hidden = builder.let("split_hidden", relay.split(init_hidden, layers, 0).astuple())
    split_cell = builder.let("split_cell", relay.split(init_cell, layers, 0).astuple())

    layer_states = []
    for i in range(layers):
        layer_state = builder.let(f"state_{i}", relay.Tuple([
            relay.squeeze(relay.TupleGetItem(split_hidden, i), axis=[0]),
            relay.squeeze(relay.TupleGetItem(split_cell, i), axis=[0])
        ]))
        layer_states.append(layer_state)

    layer_outs, layer_hiddens, layer_cells = [], [], []
    layer_input = data
    for i in range(layers):
        layer_func = first_layer if i == 0 else subsequent_layers
        layer_result = builder.let(
            f"layer_{i}_res",
            layer_func(
                layer_input, layer_states[i],
                relay.TupleGetItem(i2h_weights, i),
                relay.TupleGetItem(h2h_weights, i),
                relay.TupleGetItem(i2h_biases, i),
                relay.TupleGetItem(h2h_biases, i)))
        layer_out = builder.let(f"layer_{i}_out", relay.TupleGetItem(layer_result, 0))
        layer_hidden = builder.let(f"layer_{i}_hidden", relay.TupleGetItem(layer_result, 1))
        layer_cell = builder.let(f"layer_{i}_cell", relay.TupleGetItem(layer_result, 2))
        layer_outs.append(layer_out)
        layer_hiddens.append(layer_hidden)
        layer_cells.append(layer_cell)

        layer_input = layer_out

    # return the last layer output and stitch together the hidden states
    final_seq = layer_input
    final_hidden = relay.concatenate(layer_hiddens, 0)
    final_cell = relay.concatenate(layer_cells, 0)
    builder.ret(relay.Tuple([final_seq, final_hidden, final_cell]))
    return builder.get()


def layered_lstm_definition(batch_size, input_size, hidden_size, time_steps,
                            layers, time_axis=1):
    state_tensor_type = relay.TensorType((layers, batch_size, hidden_size))
    state_tuple_type = relay.TupleType([state_tensor_type, state_tensor_type])

    input_weight_type = relay.TensorType((4*hidden_size, input_size))
    subsequent_weight_type = relay.TensorType((4*hidden_size, hidden_size))
    bias_type = relay.TensorType((4*hidden_size,))
    i2h_weights_type = relay.TupleType([input_weight_type] + ([subsequent_weight_type] * (layers - 1)))
    h2h_weights_type = relay.TupleType([subsequent_weight_type] * layers)
    bias_weights_type = relay.TupleType([bias_type] * layers)

    input_var = relay.var("input", shape=(batch_size, time_steps, input_size))
    state_var = relay.var("state", type_annotation=state_tuple_type)
    i2h_weight_var = relay.var("i2h_weights", type_annotation=i2h_weights_type)
    h2h_weight_var = relay.var("h2h_weights", type_annotation=h2h_weights_type)
    i2h_bias_var = relay.var("i2h_biases", type_annotation=bias_weights_type)
    h2h_bias_var = relay.var("h2h_biases", type_annotation=bias_weights_type)

    ret_type = relay.TupleType([
        relay.TensorType((batch_size, time_steps, hidden_size)),
        relay.TensorType((layers, batch_size, hidden_size)),
        relay.TensorType((layers, batch_size, hidden_size))
    ])

    return relay.Function(
        [input_var, state_var, i2h_weight_var, h2h_weight_var,
         i2h_bias_var, h2h_bias_var],
        layered_lstm_body(input_var, state_var,
                          i2h_weight_var, h2h_weight_var, i2h_bias_var, h2h_bias_var,
                          batch_size, input_size, hidden_size, time_steps, layers, time_axis=time_axis),
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


def gnmt_definition(batch_size, input_size, in_seq_len, out_seq_len, hidden_size):
    # architecture based on https://ngc.nvidia.com/catalog/resources/nvidia:gnmt_v2_for_tensorflow
    builder = relay.ScopeBuilder()
    enc_bilstm = builder.let("enc_bilstm", bilstm_definition(batch_size, input_size,
                                                             hidden_size, in_seq_len))
    # output is (batch_size, in_seq_len, 2*hidden_size)
    enc_l2_lstm = builder.let("enc_l2_lstm", lstm_definition(batch_size, 2*hidden_size,
                                                             hidden_size, in_seq_len))
    # the final two layers just use hidden size
    enc_later_lstm = builder.let("enc_later_lstm", lstm_definition(batch_size, hidden_size,
                                                                   hidden_size, in_seq_len))

    attn = builder.let("attention", luong_general_attention(batch_size, out_seq_len, in_seq_len, hidden_size))
    attn_params = [relay.Var("attn_weight")]
    
    dec_in_lstm = builder.let("dec_in_lstm", lstm_definition(batch_size, input_size, hidden_size, out_seq_len))
    # 2x hidden size for the input because we concatenate the attention output
    dec_later_lstm = builder.let("dec_later_lstm", lstm_definition(batch_size, 2*hidden_size,
                                                                   hidden_size, out_seq_len))

    ret_type = relay.TensorType((batch_size, out_seq_len, hidden_size))

    # too lazy to annotate everything but the annotations for the individual components
    # should make it possible to infer the rest...
    input_seq = relay.Var("input_seq")
    target_seq = relay.Var("target_seq")

    enc_hidden = [relay.Var(f"enc_l{i+1}_state") for i in range(4)]
    enc_i2h_weight = [relay.Var(f"enc_l{i+1}_i2h_weight") for i in range(4)]
    enc_h2h_weight = [relay.Var(f"enc_l{i+1}_h2h_weight") for i in range(4)]
    enc_i2h_bias = [relay.Var(f"enc_l{i+1}_i2h_bias") for i in range(4)]
    enc_h2h_bias = [relay.Var(f"enc_l{i+1}_h2h_bias") for i in range(4)]

    dec_hidden = [relay.Var(f"dec_l{i+1}_state") for i in range(4)]
    dec_i2h_weight = [relay.Var(f"dec_l{i+1}_i2h_weight") for i in range(4)]
    dec_h2h_weight = [relay.Var(f"dec_l{i+1}_h2h_weight") for i in range(4)]
    dec_i2h_bias = [relay.Var(f"dec_l{i+1}_i2h_bias") for i in range(4)]
    dec_h2h_bias = [relay.Var(f"dec_l{i+1}_h2h_bias") for i in range(4)]

    # so we can have it all in a list
    params = [
        input_seq, target_seq,
        *enc_hidden, *enc_i2h_weight, *enc_h2h_weight, *enc_i2h_bias, *enc_h2h_bias,
        *attn_params,
        *dec_hidden, *dec_i2h_weight, *dec_h2h_weight, *dec_i2h_bias, *dec_h2h_bias
    ]

    enc_l1_res = builder.let("enc_l1_res",
                             enc_bilstm(input_seq, enc_hidden[0],
                                        enc_i2h_weight[0], enc_h2h_weight[0],
                                        enc_i2h_bias[0], enc_h2h_bias[0]))
    enc_l1_out = builder.let("enc_l1_out", relay.TupleGetItem(enc_l1_res, 0))
    enc_l2_res = builder.let("enc_l2_res",
                             enc_l2_lstm(enc_l1_out, enc_hidden[1],
                                         enc_i2h_weight[1], enc_h2h_weight[1],
                                         enc_i2h_bias[1], enc_h2h_bias[1]))
    enc_l2_out = builder.let("enc_l2_out", relay.TupleGetItem(enc_l2_res, 0))
    # for l3 and l4, we implement residual connections: we sum the result with the previous layer's output
    enc_l3_res = builder.let("enc_l3_res",
                             enc_later_lstm(enc_l2_out, enc_hidden[2],
                                            enc_i2h_weight[2], enc_h2h_weight[2],
                                            enc_i2h_bias[2], enc_h2h_bias[2]))
    enc_l3_out = builder.let("enc_l3_out", relay.TupleGetItem(enc_l3_res, 0) + enc_l2_out)
    # last layer
    enc_l4_res = builder.let("enc_l4_res",
                             enc_later_lstm(enc_l3_out, enc_hidden[3],
                                            enc_i2h_weight[3], enc_h2h_weight[3],
                                            enc_i2h_bias[3], enc_h2h_bias[3]))
    enc_l4_out = builder.let("enc_l4_out", relay.TupleGetItem(enc_l4_res, 0) + enc_l3_out)
    encoder_out = enc_l4_out

    dec_l1_res = builder.let("dec_l1_res",
                             dec_in_lstm(target_seq, dec_hidden[0],
                                         dec_i2h_weight[0], dec_h2h_weight[0],
                                         dec_i2h_bias[0], dec_h2h_bias[0]))
    dec_l1_out = builder.let("dec_l1_out", relay.TupleGetItem(dec_l1_res, 0))

    # now we compute attention and concatenate it to future decoder layers' inputs
    attn_out = builder.let("attn_out", attn(dec_l1_out, encoder_out, *attn_params))
    attn_result = builder.let("attn_result", relay.TupleGetItem(attn_out, 0))

    # subsequent layers are all identical
    last_out = dec_l1_out
    for i in range(1, 4):
        concat_inp = builder.let("dec_l{i+1}_inp", relay.concatenate([last_out, attn_result], axis=2))
        layer_res = builder.let(f"dec_l{i+1}_res",
                                dec_later_lstm(concat_inp, dec_hidden[i],
                                               dec_i2h_weight[i], dec_h2h_weight[i],
                                               dec_i2h_bias[i], dec_h2h_bias[i]))
        layer_out = builder.let(f"dec_l{i+1}_out", relay.TupleGetItem(layer_res, 0))
        # for layers 3 and 4, include residual connection (sum) in the output
        if i >= 2:
            layer_out = builder.let(f"dec_l{i+1}_residual", layer_out + last_out)
        last_out = layer_out
    builder.ret(relay.nn.softmax(last_out))

    return relay.Function(params, builder.get(), ret_type=ret_type)


def generate_random_value(ty):
    """
    For instantiating values to run GNMT
    """
    if isinstance(ty, relay.TupleType):
        return tuple([generate_random_value(field) for field in ty.fields])
    assert isinstance(ty, relay.TensorType)
    shape = tuple([int(dim) for dim in ty.shape])
    random_value = np.random.rand(*shape).astype("float32")
    return tvm.nd.array(random_value)


if __name__ == "__main__":
    batch_size, hidden_size, dense_dim = 1, 64, 64
    input_size, time_steps = 256, 6
    in_seq_len = 6
    out_seq_len = 12
    layers = 3

    mod = tvm.IRModule()
    mod["main"] = gnmt_definition(batch_size, input_size, in_seq_len, out_seq_len, hidden_size)
    try:
        mod = relay.transform.InferType()(mod)
        params = mod["main"].params
        # set up some random arguments
        random_args = []
        for param in params:
            param_ty = param.type_annotation
            random_args.append(generate_random_value(param_ty))

        exe = relay.vm.compile(mod, "llvm")
        vm = tvm.runtime.vm.VirtualMachine(exe, tvm.cpu(0))
        vm.invoke("main", *random_args)
    except:
        print(mod)
        assert False
