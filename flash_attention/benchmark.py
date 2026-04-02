
"""
Unified Benchmark Script for Triton Kernels in PyTorch.

Modes:
- 'kernels':  Benchmark raw kernels (add, matmul, softmax) individually.
- 'fused':    Benchmark a full Fused Attention block (Triton vs PyTorch).
- 'decomposed': Benchmark an Attention block built from individual Triton matmul/softmax kernels.
"""
import argparse
import time
import torch

# Import blocks
from flash_attention.blocks import *

# --- Mode 1: Individual Kernel Benchmark Utils ---
def benchmark_kernel(kernel_func, input_factory, name="kernel", sizes=None, n_warmup=10, n_iter=100):
    if sizes is None: sizes = [256, 512, 1024]
    
    print(f"\n--- {name} ---")
    print(f"{'Size':<10} {'Latency (ms)':<15}")
    for size in sizes:
        try:
            inputs = input_factory(size)
            # Warmup
            for _ in range(n_warmup):
                kernel_func(*inputs)
                torch.cuda.synchronize()
            # Run
            start = time.monotonic()
            for _ in range(n_iter):
                kernel_func(*inputs)
                torch.cuda.synchronize()
            end = time.monotonic()
            
            lat = (end - start) / n_iter * 1000
            print(f"{size:<10} {lat:<15.3f}")
        except Exception as e:
            print(f"{size:<10} ERROR: {e}")

def benchmark_pipeline(model, x, name, n_warmup, n_iter):
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(n_warmup): model(x); torch.cuda.synchronize()
            start = time.monotonic()
            for _ in range(n_iter): model(x); torch.cuda.synchronize()
            end = time.monotonic()
        return (end - start) / n_iter * 1000
    except Exception as e:
        print(f"Error in {name}: {e}")
        return float('inf')

def main():
    parser = argparse.ArgumentParser(description="Triton Benchmark Suite")
    parser.add_argument("--mode", type=str, choices=["kernels", "fused", "decomposed", "all"], 
                        default="all", help="Benchmark mode")
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu": print("CUDA required."); return
    print(f"Device: {torch.cuda.get_device_name(0)}")

    HAS_FLASH_ATTN = True

    if args.mode in ["fused", "all"]:
        print("\n=== Fused Attention Comparison ===")
        print("Comparing: Decomposed (Triton) vs Fused V1 (Triton) vs Fused V2 (Triton) vs PyTorch SDPA vs Official FlashAttn")
        dim, heads, batch = 256, 4, 4 # Head dim = 64 for FlashV2 compatibility
        
        # Models
        triton_decomp = TritonDecomposedAttentionBlock(dim, heads).to(device, dtype=torch.float16)
        triton_v1 = TritonFusedAttentionBlock(dim, heads, version="v1").to(device, dtype=torch.float16)
        triton_v2 = TritonFusedAttentionBlock(dim, heads, version="v2").to(device, dtype=torch.float16)
        torch_blk = PyTorchAttentionBlock(dim, heads).to(device, dtype=torch.float16)
        
        
        results = []

        print(f"{'SeqLen':<10} {'Decomp(ms)':<12} {'FlashV1(ms)':<12} {'FlashV2(ms)':<12} {'PyTorch(ms)':<12} {'Official(ms)':<12}")
        
        for s in args.sizes:
            # Input for float16 (Fused/Decomposed)
            x_f16 = torch.randn(batch, s, dim, device=device, dtype=torch.float16)
            
            # Decomposed (now running on float16)
            # Skip decomposed for large sizes to save time
            t_dec = benchmark_pipeline(triton_decomp, x_f16, "Decomp", 2, 5)

            t_v1 = benchmark_pipeline(triton_v1, x_f16, "FlashVk1", 10, 50)
            t_v2 = benchmark_pipeline(triton_v2, x_f16, "FlashV2", 10, 50)
            t_pt = benchmark_pipeline(torch_blk, x_f16, "PyTorch", 10, 50)
            
            off_blk = OfficialFlashAttention(heads, dim).to(device, dtype=torch.float16)
            t_off = benchmark_pipeline(off_blk, x_f16, "Official", 10, 50)
            
            print(f"{s:<10} {t_dec:<12.3f} {t_v1:<12.3f} {t_v2:<12.3f} {t_pt:<12.3f} {t_off:<12.3f}")
            results.append({
                "SeqLen": s,
                "Decomp": t_dec,
                "FlashV1": t_v1,
                "FlashV2": t_v2,
                "PyTorch": t_pt,
                "Official": t_off
            })
            
        # Parse results, save CSV and Plot
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            
            df = pd.DataFrame(results)
            df.to_csv("benchmark_results.csv", index=False)
            print("Saved benchmark_results.csv")
            
            plt.figure(figsize=(10, 6))
            plt.plot(df["SeqLen"], df["Decomp"], label="Decomposed", marker='o')
            plt.plot(df["SeqLen"], df["FlashV1"], label="FlashV1 (Triton)", marker='o')
            plt.plot(df["SeqLen"], df["FlashV2"], label="FlashV2 (Triton)", marker='o')
            plt.plot(df["SeqLen"], df["PyTorch"], label="PyTorch SDPA", marker='o')
            plt.plot(df["SeqLen"], df["Official"], label="Official FlashAttn", marker='o')
            
            plt.xscale("log", base=2)
            plt.yscale("log")
            plt.xlabel("Sequence Length")
            plt.ylabel("Runtime (ms) [Log Scale]")
            plt.title("Flash Attention Benchmark (H100) - Runtime vs SeqLen")
            plt.legend()
            plt.grid(True, which="both", ls="-", alpha=0.5)
            plt.savefig("benchmark_plot.png")
            print("Saved benchmark_plot.png")
        except ImportError:
            print("pandas or matplotlib not installed, skipping plot generation.")
        except Exception as e:
            print(f"Error plotting: {e}")

        # --- Validation Section ---
        print("\n=== Numerical Correctness Validation ===")
        validate_correctness(device)

def validate_correctness(device):
    import torch.nn.functional as F
    
    # Parameters for validation
    BATCH, HEADS, SEQ, SEQ_K = 2, 4, 32768, 32768
    DIM = 64
    sm_scale = 1.0 / (DIM ** 0.5)
    
    print(f"Validating with Batch={BATCH}, Heads={HEADS}, Seq={SEQ}, Dim={DIM} (Float16)")
    
    # Generate inputs
    # Shape: [Batch, Heads, Seq, Dim]
    q = torch.randn(BATCH, HEADS, SEQ, DIM, device=device, dtype=torch.float16)
    k = torch.randn(BATCH, HEADS, SEQ_K, DIM, device=device, dtype=torch.float16)
    v = torch.randn(BATCH, HEADS, SEQ_K, DIM, device=device, dtype=torch.float16)
    
    # PyTorch Reference (SDPA)
    # SDPA supports (Batch, Heads, Seq, Dim) inputs
    # It internally handles the scale if not provided, but we match exactly.
    ref_out = F.scaled_dot_product_attention(q, k, v, scale=sm_scale)
    
    # Triton V1
    try:
        out_v1 = triton_flash_v1(q, k, v, sm_scale)
        diff_v1 = (out_v1 - ref_out).abs().max().item()
        print(f"Triton V1 vs PyTorch SDPA: Max Diff = {diff_v1:.6f} {'[PASS]' if diff_v1 < 1e-2 else '[FAIL]'}")
    except Exception as e:
        print(f"Triton V1 Error: {e}")

    # Triton V2
    try:
        out_v2 = triton_flash_v2(q, k, v, sm_scale)
        diff_v2 = (out_v2 - ref_out).abs().max().item()
        print(f"Triton V2 vs PyTorch SDPA: Max Diff = {diff_v2:.6f} {'[PASS]' if diff_v2 < 1e-2 else '[FAIL]'}")
    except Exception as e:
        print(f"Triton V2 Error: {e}")

    # Official Flash Attention
    try:
        from flash_attn import flash_attn_func
        # flash_attn_func generally expects (Batch, Seq, Heads, Dim) if not using varlen
        # So we need to transpose: [B, H, S, D] -> [B, S, H, D]
        q_official = q.transpose(1, 2)
        k_official = k.transpose(1, 2)
        v_official = v.transpose(1, 2)
        
        out_off = flash_attn_func(q_official, k_official, v_official, dropout_p=0.0, softmax_scale=sm_scale, causal=False)
        # Transpose back for comparison: [B, S, H, D] -> [B, H, S, D]
        out_off = out_off.transpose(1, 2)
        
        diff_off = (out_off - ref_out).abs().max().item()
        print(f"Official Flash vs PyTorch: Max Diff = {diff_off:.6f} {'[PASS]' if diff_off < 1e-2 else '[FAIL]'}")
    except ImportError:
        print("Official Flash Attention not installed, skipping comparison.")
    except Exception as e:
        print(f"Official Flash Error: {e}")

if __name__ == "__main__":
    main()
