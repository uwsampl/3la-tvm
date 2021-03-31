import torch
import tvm
import vta 
import tvm.relay
import torch.nn.functional as F
from tvm.contrib import graph_runtime
from tvm.relay.op.contrib import ilavta
from tvm import relay
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class MultilayerPerceprtron(torch.nn.Module):

    def __init__(self, num_features, num_classes, num_hidden_1=128, num_hidden_2=64):
        super(MultilayerPerceprtron, self).__init__()
        self.linear_1 = torch.nn.Linear(num_features, num_hidden_1)
        self.linear_1.bias.detach().zero_()
        self.linear_2 =  torch.nn.Linear(num_hidden_1, num_hidden_2)
        self.linear_2.weight.detach().normal_(0.0, 0.1)
        self.linear_out =  torch.nn.Linear(num_hidden_2, num_classes)
        self.linear_out.weight.detach().normal_(0.0, 0.1)
        self.linear_out.bias.detach().zero_()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        o = self.linear_1(x)
        o = F.relu(o)
        o = self.linear_2(o)
        o = F.relu(o)
        logits = self.linear_out(o)
        logits = self.dequant(logits)
        prob = F.log_softmax(logits, dim=1)
        return logits, prob

def run_passes(mod):
    patterns = ilavta.pattern_table()
    mod = tvm.relay.transform.MergeComposite(patterns)(mod)
    mod = tvm.relay.transform.AnnotateTarget('ilavta')(mod)
    mod = tvm.relay.transform.PartitionGraph()(mod)
    print('[Python] Transformation complete')
    mod = relay.transform.InferType()(mod)
    return mod

def main():
    test_dataset = datasets.MNIST(root='data', 
                                train=False,
                                download=True,
                                transform=transforms.ToTensor())

    test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=8, 
                            shuffle=False)
    weight_data = torch.load(open('quantized_mlp.pickle', 'rb'), map_location='cpu')
    weight_0 = weight_data['linear_1.weight'].to('cpu').numpy()
    bias_0   = weight_data['linear_1.bias'].to('cpu').numpy()
    weight_1 = weight_data['linear_2.weight'].to('cpu').numpy()
    bias_1   = weight_data['linear_2.bias'].to('cpu').numpy()
    weight_2 = weight_data['linear_out.weight'].to('cpu').numpy()
    bias_2   = weight_data['linear_out.bias'].to('cpu').numpy()
    
    model = MultilayerPerceprtron(28 * 28, 10)
    state_dict = torch.load(open('mlp_pt_pretrained.pt', 'rb'), map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model = torch.quantization.prepare(model)
    for i, j in test_loader:
        feature, targets = i, j
        break
    
    model(feature.view(-1, 28 * 28))
    model_int8 = torch.quantization.convert(model)
    trace = torch.jit.trace(model_int8, feature.view(-1, 28 * 28))
    relay_model, params = tvm.relay.frontend.from_pytorch(trace, [('input', (8, 28 * 28))])
    with tvm.transform.PassContext(opt_level=3):
        mod = run_passes(relay_model)
    prog = mod['main']
    print(prog)
    #with vta.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
    #    mod = tvm.relay.build(prog, target=tvm.target.vta(model="sim_1x16_i8w8a32_15_15_18_17"), params=params, target_host=tvm.target.Target('llvm'))
    # print(mod['default'](tvm.cpu()))
    # m = graph_runtime.GraphModule(mod['default'](tvm.cpu()))
    # m.set_input("input", feature.view(-1, 28 * 28))
    # m.set_input("linear_1._packed_params_weight", weight_0) 
    # m.set_input("linear_1._packed_params_bias",   bias_0)
    # m.set_input("linear_2._packed_params_weight", weight_1)
    # m.set_input("linear_2._packed_params_bias",   bias_1)
    # m.set_input("linear_out._packed_params_weight", weight_2)
    # m.set_input("linear_out._packed_params_bias",   bias_2)
    # m.run()
    # o = m.get_output(0).asnumpy()
    # print(o)
    target = tvm.target.create('llvm')
    ctx = tvm.cpu()
    vm = tvm.relay.create_executor('vm', ctx=ctx, target=target, mod=mod)
    executor = vm.evaluate()
    result = executor(feature.view(-1, 28 * 28).numpy(), weight_0, bias_0, weight_1, bias_1, weight_2, bias_2)
    logits, probas = result
    probas = probas.asnumpy()
    _, predicted_labels = torch.max(torch.Tensor(probas), 1)
    print((predicted_labels == targets).sum().item() / targets.size(0))

if __name__ == '__main__':
    main()
