import torch

import tvm
from tvm import relay
import tvm.testing
from tvm.contrib import graph_executor

from model import ResMLP
from linear_rewrite import LinearLayerRewriter

# TODO: Match patterns and call into custom codegen

def assert_shapes_match(tru, est):
    if tru.shape != est.shape:
        msg = "Output shapes {} and {} don't match"
        raise AssertionError(msg.format(tru.shape, est.shape))


def verify_model(
    model_name,
    input_data=[],
    custom_convert_map={},
    rtol=1e-5,
    atol=1e-5,
    expected_ops=[],
    print_model=True,
    run_comparison=False,
):
    """Assert that the output of a compiled model matches with that of its
    baseline."""
    if isinstance(model_name, str):
        baseline_model, baseline_input = load_model(model_name)
    elif isinstance(input_data, list):
        baseline_model = model_name
        baseline_input = input_data
    elif isinstance(input_data, torch.Tensor) or len(input_data.shape) == 0:
        baseline_model = model_name
        baseline_input = [input_data]
    else:
        assert False, "Unexpected input format"

    if torch.cuda.is_available():
        if isinstance(baseline_model, torch.nn.Module):
            baseline_model = baseline_model.cuda()
        baseline_input = [inp.cuda() for inp in baseline_input]

    with torch.no_grad():
        baseline_outputs = baseline_model(*[input.clone() for input in baseline_input])

    if isinstance(baseline_outputs, tuple):
        baseline_outputs = tuple(out.cpu().numpy() for out in baseline_outputs)
    else:
        baseline_outputs = (baseline_outputs.cpu().numpy(),)

    trace = torch.jit.trace(baseline_model, [input.clone() for input in baseline_input])
    if isinstance(baseline_model, torch.nn.Module):
        trace = trace.float().eval()

        if torch.cuda.is_available():
            trace = trace.cuda()
        else:
            trace = trace.cpu()

    input_names = ["input{}".format(idx) for idx, inp in enumerate(baseline_input)]
    input_shapes = list(zip(input_names, [inp.shape for inp in baseline_input]))
    mod, params = relay.frontend.from_pytorch(trace, input_shapes, custom_convert_map)
    for arg in mod["main"].params[: len(input_names)]:
        assert arg.name_hint in input_names
    compiled_input = dict(
        zip(input_names, [inp.clone().cpu().numpy() for inp in baseline_input])
    )

    rewriter = LinearLayerRewriter()
    mod["main"] = rewriter.visit(mod["main"])
    if print_model:
        print(tvm.relay.transform.InferType()(mod))

    if not run_comparison:
        return

    with tvm.transform.PassContext(opt_level=3):
        for target, dev in tvm.testing.enabled_targets():
            relay_graph, relay_lib, relay_params = relay.build(
                mod, target=target, params=params
            )
            relay_model = graph_executor.create(relay_graph, relay_lib, dev)
            relay_model.set_input(**relay_params)
            for name, inp in compiled_input.items():
                relay_model.set_input(name, inp)
            relay_model.run()

            for i, baseline_output in enumerate(baseline_outputs):
                compiled_output = relay_model.get_output(i).asnumpy()

                assert_shapes_match(baseline_output, compiled_output)
                tvm.testing.assert_allclose(
                    baseline_output, compiled_output, rtol=rtol, atol=atol
                )

    del model_name
    del baseline_model
    torch.cuda.empty_cache()

def main():
    model = ResMLP(
        image_size = 256,
        patch_size = 16,
        dim = 512,
        depth = 12,
        num_classes = 1000
    )
    img = torch.randn(1, 3, 256, 256)
    verify_model(
        model.eval(),
        input_data=[img],
        print_model=True,
        run_comparison=True,
    )


if __name__ == "__main__":
    main()
