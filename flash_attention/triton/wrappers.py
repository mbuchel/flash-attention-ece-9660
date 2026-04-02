import torch
import triton

from triton.tools.tensor_descriptor import TensorDescriptor

from .kernels import add_kernel, matmul_kernel, softmax_kernel, flash_v1_kernel, flash_v2_kernel

def triton_flash_v1(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float) -> torch.Tensor:
    """Flash Attention V1"""
    # q, k, v are [Batch, Head, Seq, Dim]
    BLOCK_M = 128
    BLOCK_N = 64
    # The kernel uses BLOCK_DMODEL as constexpr
    Lk = k.shape[-1]
    
    o = torch.empty_like(q)
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    
    # New kernel requires TMP and M buffers
    # L and M are [Batch, Head, Seq]
    L = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    # TMP size: [Batch, Head, Seq] ? The kernel accesses TMP + off_hz * N_CTX + offs_m
    TMP = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

    num_warps = 4 if Lk <= 64 else 8
    
    # Kernel signature:
    # Q, K, V, sm_scale, TMP, L, M, Out,
    # stride_qz, stride_qh, stride_qm, stride_qk, ...
    # Z, H, N_CTX, ...
    
    flash_v1_kernel[grid](
        q, k, v, sm_scale,
        TMP, L, M,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], q.shape[2],
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=Lk, BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=1,
    )
    return o

def triton_flash_v2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float) -> torch.Tensor:
    """Flash Attention V2 (Triton Tutorial Implementation)"""
    # q, k, v are [Batch, Head, Seq, Dim]
    BATCH, N_HEAD, N_CTX, HEAD_DIM = q.shape
    
    # Block sizes from tutorial optimization
    BLOCK_M = 128
    BLOCK_N = 128
    
    o = torch.empty_like(q, dtype=torch.float16)
    M = torch.empty((BATCH, N_HEAD, N_CTX), device=q.device, dtype=torch.float32)

    y_dim = BATCH * N_HEAD * N_CTX
    
    # Pass correct block shapes directly since we don't have the autotune pre-hook
    desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM])
    desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM])

    grid = (triton.cdiv(N_CTX, BLOCK_M), BATCH * N_HEAD, 1)
    
    flash_v2_kernel[grid](
        sm_scale, M,
        BATCH, N_HEAD, desc_q, desc_k, desc_v, desc_o, N_CTX,
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        FP8_OUTPUT=False,
        STAGE=1,
        warp_specialize=True,
        IS_HOPPER=True,
        num_warps=8, 
        num_stages=3
    )
    return o

def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Element-wise addition using Triton kernel."""
    output = torch.empty_like(x)
    n_elements = output.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    add_kernel[grid](
        x, y, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication using Triton kernel."""
    M, K = a.shape
    K, N = b.shape
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    return c


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Softmax using Triton kernel (row-wise)."""
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    output = torch.empty_like(x)
    
    softmax_kernel[(n_rows,)](
        output, x,
        x.stride(0), output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
