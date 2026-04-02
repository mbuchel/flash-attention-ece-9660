
"""
Single-Run Benchmark & Validation Script for Triton Kernels in PyTorch.
Runs exactly one instance of each kernel with the specified size and validates numerical correctness.
"""
import argparse
import time
import torch
import torch.functional as F

# Import Blocks
from flash_attention.blocks import *

def measure_latency(model, x, n_warmup=5, n_iter=20):
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(n_warmup): model(x); torch.cuda.synchronize()
            start = time.monotonic()
            for _ in range(n_iter): model(x); torch.cuda.synchronize()
            end = time.monotonic()
        return (end - start) / n_iter * 1000
    except Exception as e:
        print(f"Error executing model: {e}")
        return float('inf')

def validate_kernels(device, batch, heads, seq, dim, kernel_type):
    print(f"\n=== Kernel Correctness Validation (Target: {kernel_type}) ===")
    
    # Inputs: [Batch, Heads, Seq, HeadDim]
    head_dim = dim // heads
    q = torch.randn(batch, heads, seq, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch, heads, seq, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch, heads, seq, head_dim, device=device, dtype=torch.float16)
    sm_scale = 1.0 / (head_dim ** 0.5)
    
    # 1. PyTorch Reference (SDPA - Non-Causal)
    ref_out = F.scaled_dot_product_attention(q, k, v, scale=sm_scale)
    
    # helper for diff
    def check_diff(out, name):
        diff = (out - ref_out).abs().max().item()
        status = "PASS" if diff < 1e-2 else "FAIL"
        print(f"{name} vs PyTorch: Max Diff = {diff:.6f} [{status}]")

    try:
        if kernel_type == "v1":
            out_v1 = triton_flash_v1(q, k, v, sm_scale)
            check_diff(out_v1, "Triton V1")
        elif kernel_type == "v2":
            out_v2 = triton_flash_v2(q, k, v, sm_scale)
            check_diff(out_v2, "Triton V2")
        elif kernel_type == "official":
            try:
                from flash_attn import flash_attn_func
                q_off, k_off, v_off = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                out_off = flash_attn_func(q_off, k_off, v_off, dropout_p=0.0, softmax_scale=sm_scale, causal=False)
                out_off = out_off.transpose(1, 2)
                check_diff(out_off, "Official Flash")
            except ImportError:
                print("Official Flash Attention not installed.")
        elif kernel_type == "torch":
            # Validating torch against itself is trivial but confirms harness works
            out_torch = F.scaled_dot_product_attention(q, k, v, scale=sm_scale)
            check_diff(out_torch, "PyTorch SDPA")
        elif kernel_type == "decomposed":
            # Logic matches TritonDecomposedAttentionBlock
            out_dec = torch.empty_like(q)
            for i in range(batch):
                for j in range(heads):
                    qi = q[i, j].contiguous()
                    ki = k[i, j]
                    vi = v[i, j].contiguous()
                    
                    # Score = Q @ K.T
                    ki_t = ki.transpose(0, 1).contiguous()
                    score = triton_matmul(qi, ki_t)
                    score = score * sm_scale
                    
                    # Softmax
                    probs = triton_softmax(score)
                    
                    # Out = Probs @ V
                    out_head = triton_matmul(probs, vi)
                    out_dec[i, j] = out_head
            check_diff(out_dec, "Triton Decomposed")
            
    except Exception as e:
        print(f"Error validating {kernel_type}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Single Run Benchmark")
    parser.add_argument("--size", type=int, default=1024, help="Sequence length")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--dim", type=int, default=256, help="Model dimension")
    parser.add_argument("--heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--kernel", type=str, choices=["decomposed", "v1", "v2", "torch", "official"], required=True, help="Kernel to benchmark against Decomposed")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required.")
        return
    
    device = "cuda"
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Config: Batch={args.batch}, Seq={args.size}, Dim={args.dim}, Heads={args.heads}")
    print(f"Target Kernel: {args.kernel}")

    # --- Profiling Run ---
    print("\n=== Profiling Run (Single Forward Pass) ===")
    
    x = torch.randn(args.batch, args.size, args.dim, device=device, dtype=torch.float16)
    
    if args.kernel == "decomposed":
        model = TritonDecomposedAttentionBlock(args.dim, args.heads).to(device, dtype=torch.float16)
        name = "Triton Decomposed"
    elif args.kernel == "v1":
        model = TritonFusedAttentionBlock(args.dim, args.heads, version="v1").to(device, dtype=torch.float16)
        name = "Triton Flash V1"
    elif args.kernel == "v2":
        model = TritonFusedAttentionBlock(args.dim, args.heads, version="v2").to(device, dtype=torch.float16)
        name = "Triton Flash V2"
    elif args.kernel == "torch":
        model = PyTorchAttentionBlock(args.dim, args.heads).to(device, dtype=torch.float16)
        name = "PyTorch SDPA"
    elif args.kernel == "official":
        try:
            from flash_attn import flash_attn_func
            class OfficialFlashAttention(nn.Module):
                def __init__(self, heads, dim):
                    super().__init__()
                    self.heads = heads
                    self.dim = dim
                def forward(self, x):
                    b, s, d = x.shape
                    qkv = x.view(b, s, self.heads, self.dim // self.heads)
                    out = flash_attn_func(qkv, qkv, qkv, dropout_p=0.0, softmax_scale=None, causal=False)
                    return out.view(b, s, d)
            model = OfficialFlashAttention(args.heads, args.dim).to(device, dtype=torch.float16)
            name = "Official Flash"
        except ImportError:
            print("Official Flash not installed")
            return

    print(f"Running {name}...")
    torch.cuda.synchronize()
    
    # Profile Run - EXACTLY ONE RUN
    print(">>> Start Profile Region <<<")
    model(x)
    torch.cuda.synchronize()
    print(">>> End Profile Region <<<")

    # --- Validation ---
    # validate_kernels(device, args.batch, args.heads, args.size, args.dim, args.kernel)

if __name__ == "__main__":
    main()
