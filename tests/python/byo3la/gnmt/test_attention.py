"""
Test Relay general attention against PT general attention
(based on the OpenNMT implementation)
"""
import torch

import numpy as np
import tvm
from tvm import relay
from tvm import runtime

from attention import luong_general_attention

def run_tvm_vm(mod):
    target = 'llvm'
    ctx = tvm.cpu(0)
    with tvm.transform.PassContext(opt_level=3):
        exe = relay.vm.compile(mod, target)
        vm = runtime.vm.VirtualMachine(exe, ctx)
        return vm.invoke("main")


def reference_attn(batch_size, query_units, key_units, num_units,
                   in_seq, out_seq, weight):
    linear_in = torch.nn.Linear(num_units, num_units, bias=False)
    linear_in.weight.data = torch.from_numpy(weight)

    h_t = torch.from_numpy(in_seq)
    h_s = torch.from_numpy(out_seq)

    h_t_ = h_t.view(batch_size * query_units, num_units)
    h_t_ = linear_in(h_t_)
    h_t = h_t_.view(batch_size, query_units, num_units)
    h_s_ = h_s.transpose(1, 2)
    score = torch.bmm(h_t, h_s_)

    align_vectors = torch.nn.functional.softmax(score.view(batch_size * query_units, key_units), -1)
    align_vectors = align_vectors.view(batch_size, query_units, key_units)
    return torch.bmm(align_vectors, h_s)


if __name__ == "__main__":
    mod = tvm.IRModule()
    batch_size, hidden_size = 3, 64
    in_seq_len = 6
    out_seq_len = 12
    mod["luong_attn"] = luong_general_attention(batch_size, in_seq_len, out_seq_len, hidden_size)

    attn_var = mod.get_global_var("luong_attn")
    in_shape = (batch_size, in_seq_len, hidden_size)
    out_shape = (batch_size, out_seq_len, hidden_size)
    weight_shape = (hidden_size, hidden_size)

    random_input = np.random.rand(*in_shape)
    random_output = np.random.rand(*out_shape)
    random_weight = np.random.rand(*weight_shape)

    mod["main"] = relay.Function([], attn_var(
        relay.const(random_input),
        relay.const(random_output),
        relay.const(random_weight)))
    res = run_tvm_vm(mod)
    context_res = res[0].numpy()

    torch_res = reference_attn(batch_size, in_seq_len,
                               out_seq_len, hidden_size,
                               random_input, random_output, random_weight)
    torch_res = torch_res.detach().numpy()

    # possibly something going on, as default settings for tolerances don't work
    assert np.allclose(context_res, torch_res,
                       rtol=1e-4, atol=1e-6)
