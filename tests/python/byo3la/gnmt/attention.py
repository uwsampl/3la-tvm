import tvm
from tvm import relay

def bahdanau_attention(batch_size, query_units, key_units, num_units):
    # based on https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Translation/GNMT/seq2seq/models/attention.py
    builder = relay.ScopeBuilder()

    query_type = relay.TensorType((batch_size, query_units, num_units))
    key_type = relay.TensorType((batch_size, key_units, num_units))
    scores_type = relay.TensorType((batch_size, query_units, key_units))

    linear_q_weight_type = relay.TensorType((num_units, num_units))
    linear_k_weight_type = relay.TensorType((num_units, num_units))
    att_weight_type = relay.TensorType((1, num_units))

    query_input = relay.Var("query_input", type_annotation=query_type)
    key_input = relay.Var("key_input", type_annotation=key_type)
    linear_q_weight = relay.Var("linear_q_weight", type_annotation=linear_q_weight_type)
    linear_k_weight = relay.Var("linear_k_weight", type_annotation=linear_k_weight_type)
    att_weight = relay.Var("att_weight", type_annotation=att_weight_type)

    linear_q = builder.let("linear_q", relay.nn.dense(query_input, linear_q_weight))
    linear_k = builder.let("linear_k", relay.nn.dense(key_input, linear_k_weight))

    att_qk = builder.let("att_qk", relay.expand_dims(linear_q, axis=2) + relay.expand_dims(linear_k, axis=1))
    prod_var = relay.Var("prod", type_annotation=relay.TensorType((batch_size, query_units, key_units, 1)))
    prod = builder.let(prod_var, relay.nn.dense(relay.tanh(att_qk), att_weight))
    score_var = relay.Var("score", type_annotation=relay.TensorType((batch_size, query_units, key_units)))
    score = builder.let(score_var, relay.squeeze(prod, axis=[3]))
    scores_normalized = builder.let("scores_normalized", relay.nn.softmax(score, axis=-1))

    context_var = relay.Var("context", type_annotation=query_type)
    context = builder.let(context_var, relay.nn.batch_matmul(scores_normalized, relay.transpose(key_input, axes=[0,2,1])))
    builder.ret(relay.Tuple([context, scores_normalized]))

    return relay.Function(
        [query_input, key_input,
         linear_q_weight, linear_k_weight,
         att_weight],
        builder.get(),
        ret_type=relay.TupleType([query_type, scores_type]))


def luong_general_attention(batch_size, query_units, key_units, num_units):
    # based on https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/global_attention.py#L97
    builder = relay.ScopeBuilder()

    query_type = relay.TensorType((batch_size, query_units, num_units))
    key_type = relay.TensorType((batch_size, key_units, num_units))
    scores_type = relay.TensorType((batch_size, query_units, key_units))
    weight_type = relay.TensorType((num_units, num_units))

    query_input = relay.Var("query_input", type_annotation=query_type)
    key_input = relay.Var("key_input", type_annotation=key_type)
    weight_input = relay.Var("weight", type_annotation=weight_type)

    reshape_query = builder.let("reshape_query", relay.reshape(query_input, (batch_size*query_units, num_units)))
    linear_query = builder.let("linear_query", relay.reshape(relay.nn.dense(reshape_query, weight_input), (batch_size, query_units, num_units)))
    prod_var = relay.Var("prod", type_annotation=scores_type)
    prod = builder.let(prod_var, relay.nn.batch_matmul(linear_query, key_input))
    scores_var = relay.Var("scores", type_annotation=scores_type)
    scores = builder.let(scores_var, relay.reshape(relay.nn.softmax(relay.reshape(prod, (batch_size*query_units, key_units))), (batch_size, query_units, key_units)))
    context = builder.let("context", relay.nn.batch_matmul(scores, relay.transpose(key_input, axes=[0,2,1])))
    builder.ret(relay.Tuple([context, scores]))

    return relay.Function(
        [query_input, key_input, weight_input],
        builder.get(),
        ret_type=relay.TupleType([query_type, scores_type]))
