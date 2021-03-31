import torch
import tvm
import tvm.runtime
import tvm.relay
import numpy as np
import pickle

from torch.utils.data import DataLoader, dataloader
from torchvision import datasets, transforms
from tvm.relay import nn
from tvm import relay

from tvm.relay.op.contrib import ilavta

def get_data_loader(batch_size):
    test_dataset = datasets.MNIST(root='data', 
                                train=False,
                                download=True,
                                transform=transforms.ToTensor())

    test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=batch_size, 
                            shuffle=False)
    return test_loader

def run_passes(mod):
    patterns = ilavta.pattern_table()
    mod = tvm.relay.transform.MergeComposite(patterns)(mod)
    mod = tvm.relay.transform.AnnotateTarget('ilavta')(mod)
    mod = tvm.relay.transform.PartitionGraph()(mod)
    print('[Python] Transformation complete')
    mod = relay.transform.InferType()(mod)
    return mod

def compile_mod(mod):
    target = tvm.target.create('llvm')
    ctx = tvm.cpu()
    vm = relay.create_executor('vm', ctx=ctx, target=target, mod=mod)
    print('[Python] Execute Graph')
    result = vm.evaluate()
    return result

def run_mod(exec, *inputs):
    ctx = tvm.cpu()
    input_tensors = list(map(lambda x: tvm.nd.array(x, ctx=ctx), inputs))
    output = exec(*input_tensors).asnumpy()
    # print('[Python] Done')
    return output

def run_module_graph(mod, *inputs):
    target = tvm.target.create('llvm')
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = tvm.relay.build(mod, target)
    ctx = tvm.cpu()
    runtime_exec = graph_runtime.create(graph, lib, ctx)
    
    input_tensors = list(map(lambda x: tvm.nd.array(x, ctx=ctx), inputs))
    
    print('[Python] Execute Graph')
    for (i, inp) in enumerate(input_tensors):
        runtime_exec.set_input(i, inp)
    runtime_exec.set_input(**params)
    runtime_exec.run()

    output = runtime_exec.get_output(0).asnumpy()
    print('[Python] Done')
    return output

def calculate_scale_zp(x_max, x_min, nbit=8):
    qmax = relay.const(2.0 ** (nbit) - 1, dtype='float32')
    qmin = relay.const(0, dtype='float32')
    scale = relay.divide((x_max - x_min), (qmax - qmin))
    z = qmin - x_min / scale
    zp = relay.If(
        relay.less(z, qmin),
        qmin,
        relay.If(
            relay.greater(z, qmax),
            qmax,
            z
        )
    )
    zp = relay.floor(zp)
    # zp = relay.cast(zp, 'int8')
    return scale, zp

def quantize(tensor: tvm.relay.Var, x_min=None, x_max=None, nbit=8):
    if x_max is None:
        x_max = relay.max(tensor)
    if x_min is None:
        x_min = relay.min(tensor)
    scale, zp = calculate_scale_zp(x_max, x_min, nbit)
    result = tensor / scale + zp
    result = relay.clip(result, 0.0, 2.0 ** nbit - 1)
    result = relay.round(result)
    # result = relay.cast(result, 'int8')
    return result, scale, zp

def dequantize(qtensor, scale, zp):
    assert qtensor is not None
    assert scale is not None
    assert zp is not None
    return scale * (qtensor - zp)

def quantized_linear_layer(inp, weight, bias, stats, scale_inp, zp_inp):
    qw, scale_w, zp_w = quantize(weight)
    qb, scale_b, zp_b = quantize(bias)
    scale_next, zp_next = calculate_scale_zp(relay.const(stats['max']), relay.const(stats['min']))
    x = inp - zp_inp
    qw = (scale_inp * scale_w / scale_next) * (qw - zp_w)
    qb = (scale_b / scale_next) * (qb - zp_b)
    x = relay.cast(x, 'uint8')
    qw = relay.cast(qw, 'uint8')
    qb = relay.cast(qb, 'uint8')
    x = nn.dense(x, qw)
    x = nn.bias_add(x, qb)
    x = relay.cast(x, 'float32')
    x = x / scale_next + zp_next
    return x, scale_next, zp_next

def linear(data, in_dim, out_dim, stats, scale_inp, zp_inp, var_cnt=0, quant=True):
    W = tvm.relay.var('weight_{}'.format(var_cnt), shape=(out_dim, in_dim))
    B = tvm.relay.var('bias_{}'.format(var_cnt), shape=(out_dim,))
    
    if quant:
        out, scale_next, zp_next = quantized_linear_layer(data, W, B, stats, scale_inp, zp_inp)
        return out, scale_next, zp_next
    out = nn.dense(data, W)
    out = nn.bias_add(out, B)
    return out, None, None

def MultiLayerPreceptron(batch, image_shape, num_classes, stats, num_hidden_1=128, num_hidden_2=64):
    input_shape = (batch, ) + image_shape
    num_features = image_shape[0] * image_shape[1] * image_shape[2]
    inputs = tvm.relay.var('input', shape=input_shape)
    inputs = nn.batch_flatten(inputs)
    inputs, scale, zp = quantize(inputs)
    fc1, scale, zp = linear(inputs, num_features, num_hidden_1, stats['linear_2'], scale, zp, 0, quant=True)
    act1 = relay.cast(nn.relu(relay.cast(fc1, 'uint8')), 'float32')
    fc2, scale, zp = linear(act1, num_hidden_1, num_hidden_2, stats['linear_out'], scale, zp, 1, quant=True)
    act2 = relay.cast(nn.relu(relay.cast(fc2, 'uint8')), 'float32')
    fc3, scale, zp = linear(act2, num_hidden_2, num_classes, stats['linear_out'], scale, zp, 2, quant=True)
    fc3 = dequantize(fc3, scale, zp)
    prog = nn.log_softmax(fc3, axis=1)
    return prog

def QuantizedModelNumpy(inp, stats, weight_0, bias_0, weight_1, bias_1, weight_2, bias_2):
    def calculate_scale_zp(x_max, x_min, nbits=8):
        qmax = 2.0 ** nbits - 1
        qmin = 0
        scale = (x_max - x_min) / (qmax - qmin)
        z = qmin - x_min / scale
        if z < qmin:
            z = qmin
        elif z > qmax:
            z = qmax
        zp = int(z)
        return scale, zp

    def quantize(data, x_max=None, x_min=None, nbits=8):
        if x_max is None:
            x_max = np.max(data)
        if x_min is None:
            x_min = np.min(data)
        scale, zp = calculate_scale_zp(x_max, x_min, nbits)
        result = data / scale + zp
        result = np.clip(result, 0, 2 ** nbits - 1)
        result = np.round(result).astype(np.uint8)
        return result, scale, zp
    
    def dequantize(data, scale, zp):
        return scale * (data - zp)
    
    def relu(x):
        return x * (x > 0)
    
    def quantized_linear_layer(inp, weight, bias, stats, scale_inp, zp_inp):
        qw, scale_w, zp_w = quantize(weight)
        qb, scale_b, zp_b = quantize(bias)
        qw = qw.astype(np.float32)
        qb = qb.astype(np.float32)
        scale_next, zp_next = calculate_scale_zp(stats['max'], stats['min'])
        x = inp.astype(np.float32) - zp_inp
        qw = (scale_inp * scale_w / scale_next) * (qw - zp_w)
        qb = (scale_b / scale_next) * (qb - zp_b)
        x = np.matmul(x, qw.transpose())
        x = x + qb
        x = x / scale_next + zp_next
        return x, scale_next, zp_next
    
    def linear(inp, weight, bias):
        return np.matmul(inp, weight.transpose()) + bias
    
    def run_model():
        inputs = inp.reshape(8, 28 * 28)
        inputs, scale, zp = quantize(inputs)
        # inputs = inp.reshape(8, 28 * 28)
        fc1, scale, zp = quantized_linear_layer(inputs, weight_0, bias_0, stats['linear_2'], scale, zp)
        # fc1 = linear(inputs, weight_0, bias_0)
        act1 = relu(fc1)
        fc2, scale, zp = quantized_linear_layer(act1, weight_1, bias_1, stats['linear_out'], scale, zp)
        # fc2 = linear(act1, weight_1, bias_1)
        act2 = relu(fc2)
        act2 = dequantize(act2, scale, zp)
        fc3 = linear(act2, weight_2, bias_2)
        return torch.log_softmax(torch.Tensor(fc3), dim=1).numpy()
    
    return run_model()

def run_model():
    batch_size = 8
    image_shape = (1, 28, 28)
    num_classes = 10
    num_hidden_1 = 128
    num_hidden_2 = 64
    input_shape = (batch_size,) + image_shape

    weight_0_shape = (num_hidden_1, image_shape[0] * image_shape[1] * image_shape[2])
    bias_0_shape = (num_hidden_1,)

    weight_1_shape = (num_hidden_2, num_hidden_1)
    bias_1_shape = (num_hidden_2,)

    weight_2_shape = (num_classes, num_hidden_2)
    bias_2_shape = (num_classes,)

    weight_data = torch.load(open('quantized_mlp.pickle', 'rb'), map_location='cpu')
    assert weight_data['linear_1.weight'].shape == weight_0_shape
    assert weight_data['linear_2.weight'].shape == weight_1_shape
    assert weight_data['linear_out.weight'].shape == weight_2_shape

    assert weight_data['linear_1.bias'].shape == bias_0_shape
    assert weight_data['linear_2.bias'].shape == bias_1_shape
    assert weight_data['linear_out.bias'].shape == bias_2_shape

    data_loader = get_data_loader(batch_size)
    weight_0 = weight_data['linear_1.weight'].to('cpu').numpy()
    bias_0   = weight_data['linear_1.bias'].to('cpu').numpy()
    weight_1 = weight_data['linear_2.weight'].to('cpu').numpy()
    bias_1   = weight_data['linear_2.bias'].to('cpu').numpy()
    weight_2 = weight_data['linear_out.weight'].to('cpu').numpy()
    bias_2   = weight_data['linear_out.bias'].to('cpu').numpy()
    stats = weight_data['stats']

    # print(stats)
    mod = MultiLayerPreceptron(batch_size, image_shape, num_classes, stats, num_hidden_1, num_hidden_2)
    mod = tvm.ir.IRModule.from_expr(mod)
    mod = run_passes(mod)
    print(mod)

    # output = run_module(mod, input_data, weight_0, bias_0, weight_1, bias_1, weight_2, bias_2)
    # print(output)
    correct_pred, num_examples = 0, 0
    # print(weight_0)
    vm = compile_mod(mod)
    for features, targets in data_loader:
        # features = features.view(-1, 28 * 28)
        # print(features.shape)
        features = np.round(features)
        features = features.numpy()
        assert features.shape == input_shape
        probas = run_mod(vm, features, weight_0, bias_0, weight_1, bias_1, weight_2, bias_2)
        _, predicted_labels = torch.max(torch.Tensor(probas), 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
        #if num_examples % 100 == 0:
        print('Current Accuracy: {}'.format(correct_pred.float() / num_examples))
        break
    return correct_pred.float()/num_examples * 100

run_model()
