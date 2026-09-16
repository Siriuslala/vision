"""Verify Qwen3.5 FLA dispatch and GPU outputs against the PyTorch reference."""
import argparse
import importlib.metadata
import inspect
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-id', default='0')
    args = parser.parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    print('flash-linear-attention:', importlib.metadata.version('flash-linear-attention'), flush=True)
    print('torch:', torch.__version__, flush=True)
    for name in ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule'):
        function = getattr(qwen, name)
        while True:
            closure = inspect.getclosurevars(function).nonlocals
            if 'is_new_implementation' in closure:
                if not closure['is_new_implementation']:
                    raise RuntimeError(f'{name} is still using the PyTorch fallback')
                print(f'{name}: {closure["implementation"].__module__}', flush=True)
                break
            if not hasattr(function, '__wrapped__'):
                raise RuntimeError(f'Cannot verify dispatch for {name}')
            function = function.__wrapped__
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    torch.manual_seed(42)
    reference = inspect.unwrap(qwen.torch_recurrent_gated_delta_rule)
    with torch.inference_mode():
        for dtype in (torch.float16, torch.bfloat16):
            q, k, v = [torch.randn(2, 65, 2, 64, device='cuda', dtype=dtype) for _ in range(3)]
            g = -torch.rand(2, 65, 2, device='cuda')
            beta = torch.rand(2, 65, 2, device='cuda', dtype=dtype)
            state = torch.randn(2, 2, 64, 64, device='cuda') * 0.01
            for length, kernel in ((65, qwen.torch_chunk_gated_delta_rule),
                                   (1, qwen.torch_recurrent_gated_delta_rule)):
                inputs = [x[:, :length].contiguous() for x in (q, k, v, g, beta)]
                kwargs = dict(initial_state=state.clone(), output_final_state=True,
                              use_qk_l2norm_in_kernel=True)
                actual = kernel(*inputs[:3], g=inputs[3], beta=inputs[4], **kwargs)
                expected = reference(*inputs[:3], g=inputs[3], beta=inputs[4],
                                     **{**kwargs, 'initial_state': state.clone()})
                for a, e in zip(actual, expected):
                    torch.testing.assert_close(a.float(), e.float(), atol=0.02, rtol=0.02)
                torch.cuda.synchronize()
                print(f'{dtype}, tokens={length}: output and final state match reference', flush=True)
    print('Qwen3.5 FLA GPU checks passed.', flush=True)


if __name__ == '__main__':
    main()
