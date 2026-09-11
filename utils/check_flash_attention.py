"""Report installation paths and verify prebuilt FlashAttention CUDA kernels."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-id', default='0')
    parser.add_argument('--import-only', action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    import torch
    import flash_attn
    import flash_attn_2_cuda

    distribution = importlib.metadata.distribution('flash-attn')
    print(json.dumps({
        'python': sys.version.split()[0],
        'environment': sys.prefix,
        'torch': torch.__version__,
        'torch_cuda': torch.version.cuda,
        'cxx11_abi': torch._C._GLIBCXX_USE_CXX11_ABI,
        'flash_attn': distribution.version,
        'python_package': str(Path(flash_attn.__file__).resolve()),
        'cuda_extension': str(Path(flash_attn_2_cuda.__file__).resolve()),
        'site_packages': str(distribution.locate_file('')),
    }, indent=2), flush=True)
    if args.import_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available; run on a GPU-enabled host.')

    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import torch.nn.functional as F

    torch.manual_seed(0)
    print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
    for dtype in (torch.float16, torch.bfloat16):
        q, k, v = [torch.randn(2, 128, 4, 64, device='cuda', dtype=dtype,
                               requires_grad=True) for _ in range(3)]
        tolerance = 0.005 if dtype == torch.float16 else 0.05
        for causal in (False, True):
            actual = flash_attn_func(q, k, v, dropout_p=0.0, causal=causal)
            with sdpa_kernel(SDPBackend.MATH):
                expected = F.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                    dropout_p=0.0, is_causal=causal).transpose(1, 2)
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
            grad = torch.randn_like(actual)
            actual_grads = torch.autograd.grad(actual, (q, k, v), grad, retain_graph=True)
            expected_grads = torch.autograd.grad(expected, (q, k, v), grad)
            for a, e in zip(actual_grads, expected_grads):
                torch.testing.assert_close(a, e, atol=tolerance, rtol=tolerance)
            print(f'{dtype}, causal={causal}: forward and backward match SDPA math', flush=True)
        # Packed sequences with unequal lengths exercise the interface used by VLMs.
        lengths = (73, 128)
        packed = [torch.cat([x[i, :n].detach() for i, n in enumerate(lengths)]) for x in (q, k, v)]
        cumulative = torch.tensor([0, 73, 201], device='cuda', dtype=torch.int32)
        actual = flash_attn_varlen_func(*packed, cumulative, cumulative, 128, 128,
                                       dropout_p=0.0, causal=True)
        expected_parts = []
        with sdpa_kernel(SDPBackend.MATH):
            for i, n in enumerate(lengths):
                inputs = [x[i:i+1, :n].detach().transpose(1, 2) for x in (q, k, v)]
                expected_parts.append(F.scaled_dot_product_attention(
                    *inputs, dropout_p=0.0, is_causal=True).transpose(1, 2).squeeze(0))
        torch.testing.assert_close(actual, torch.cat(expected_parts), atol=tolerance, rtol=tolerance)
        print(f'{dtype}: packed variable-length attention matches SDPA math', flush=True)
    torch.cuda.synchronize()
    print('FlashAttention GPU checks passed.', flush=True)


if __name__ == '__main__':
    main()
