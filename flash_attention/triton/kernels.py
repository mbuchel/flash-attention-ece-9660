import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

@triton.jit
def flash_v1_kernel(
    Q, K, V, sm_scale,
    TMP, L, M,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    off_k = off_hz * stride_qh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    off_v = off_hz * stride_qh + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qk
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    # initialize pointer to m and l
    t_ptrs = TMP + off_hz * N_CTX + offs_m
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    # loop over k, v and update accumulator
    # loop over k, v and update accumulator
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(k_ptrs + start_n * stride_kn)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= sm_scale
        # qk += tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), 0, float("-inf"))
        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        tl.store(t_ptrs, acc_scale)
        acc_scale = tl.load(t_ptrs)  # BUG: have to store and immediately load
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.load(v_ptrs + start_n * stride_vk)
        p = p.to(tl.float16)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    l_ptrs = L + off_hz * N_CTX + offs_m
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, l_i)
    tl.store(m_ptrs, m_i)
    # initialize pointers to output
    offs_n = tl.arange(0, BLOCK_DMODEL)
    off_o = off_hz * stride_oh + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)

@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)

@triton.jit
def flash_v2_kernel(
    sm_scale, M,  #
    Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    warp_specialize: tl.constexpr,  #
    IS_HOPPER: tl.constexpr,  #
):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    c0 = offset_y * 0 # Scalar zero derived from offset
    
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = desc_q.load([qo_offset_y, 0])
    
    # -----------------------------------------------------------
    # Stage 1: Off-band (Causal=False -> STAGE=1)
    # -----------------------------------------------------------
    if STAGE & 1:
        # Inlined _attn_fwd_inner for STAGE=1 case
        # STAGE passed to inner was (4 - STAGE). If STAGE=1, inner gets 3.
        # inner STAGE=3: else: lo, hi = 0, N_CTX
        
        # Inner vars
        curr_stage = 4 - STAGE # 3
        # Range logic
        if curr_stage == 1:
            lo, hi = 0, start_m * BLOCK_M
        elif curr_stage == 2:
            lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
            lo = tl.multiple_of(lo, BLOCK_M)
        else: # 3
            lo, hi = 0, N_CTX
            
        offsetk_y = offset_y + lo
        offsetv_y = offset_y + lo
            
        for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            k = desc_k.load([offsetk_y.to(tl.int32), c0]).T
            qk = tl.dot(q, k)
            if curr_stage == 2:
                mask = offs_m[:, None] >= (start_n + offs_n[None, :])
                qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
                m_ij = tl.maximum(m_i, tl.max(qk, 1))
                qk -= m_ij[:, None]
            else:
                m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
                qk = qk * qk_scale - m_ij[:, None]
            p = tl.math.exp2(qk)
            # -- compute correction factor
            alpha = tl.math.exp2(m_i - m_ij)
            l_ij = tl.sum(p, 1)
            # -- update output accumulator --
            if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
                BM: tl.constexpr = acc.shape[0]
                BN: tl.constexpr = acc.shape[1]
                acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
                acc0 = acc0 * alpha[:, None]
                acc1 = acc1 * alpha[:, None]
                acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
            else:
                acc = acc * alpha[:, None]
            # prepare p and v for the dot
            v = desc_v.load([offsetv_y.to(tl.int32), c0])
            p = p.to(dtype)
            acc = tl.dot(p, v, acc)
            # update m_i and l_i
            l_i = l_i * alpha + l_ij
            m_i = m_ij
            offsetk_y += BLOCK_N
            offsetv_y += BLOCK_N

    # -----------------------------------------------------------
    # Stage 2: On-band (Causal=True -> Diag masked block)
    # -----------------------------------------------------------
    if STAGE & 2:
        # Inner vars for Causal Masking (Diagonal)
        curr_stage = 2
        # Range logic for Stage 2
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
            
        offsetk_y = offset_y + lo
        offsetv_y = offset_y + lo
            
        for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            k = desc_k.load([offsetk_y.to(tl.int32), c0]).T
            qk = tl.dot(q, k)
            
            # Causal Mask application (Stage 2)
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
            
            p = tl.math.exp2(qk)
            # -- compute correction factor
            alpha = tl.math.exp2(m_i - m_ij)
            l_ij = tl.sum(p, 1)
            # -- update output accumulator --
            if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
                BM: tl.constexpr = acc.shape[0]
                BN: tl.constexpr = acc.shape[1]
                acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
                acc0 = acc0 * alpha[:, None]
                acc1 = acc1 * alpha[:, None]
                acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
            else:
                acc = acc * alpha[:, None]
            # prepare p and v for the dot
            v = desc_v.load([offsetv_y.to(tl.int32), c0])
            p = p.to(dtype)
            acc = tl.dot(p, v, acc)
            # update m_i and l_i
            l_i = l_i * alpha + l_ij
            m_i = m_ij
            offsetk_y += BLOCK_N
            offsetv_y += BLOCK_N
            
    # -----------------------------------------------------------
    # Stage 2: On-band (Causal=True -> Diag masked block)
    # -----------------------------------------------------------
    if STAGE & 2:
        # Inner vars for Causal Masking (Diagonal)
        curr_stage = 2
        # Range logic for Stage 2
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
            
        offsetk_y = offset_y + lo
        offsetv_y = offset_y + lo
            
        c0 = offset_y * 0 # Scalar zero derived from offset
            
        for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            k = desc_k.load([offsetk_y.to(tl.int32), c0]).T
            qk = tl.dot(q, k)
            
            # Causal Mask application (Stage 2)
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
            
            p = tl.math.exp2(qk)
            # -- compute correction factor
            alpha = tl.math.exp2(m_i - m_ij)
            l_ij = tl.sum(p, 1)
            # -- update output accumulator --
            if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
                BM: tl.constexpr = acc.shape[0]
                BN: tl.constexpr = acc.shape[1]
                acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
                acc0 = acc0 * alpha[:, None]
                acc1 = acc1 * alpha[:, None]
                acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
            else:
                acc = acc * alpha[:, None]
            # prepare p and v for the dot
            v = desc_v.load([offsetv_y.to(tl.int32), c0])
            p = p.to(dtype)
            acc = tl.dot(p, v, acc)
            # update m_i and l_i
            l_i = l_i * alpha + l_ij
            m_i = m_ij
            offsetk_y += BLOCK_N
            offsetv_y += BLOCK_N
            
 

    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    
    # Store with strict types
    # We derive the zero scaler from a scalar variable (start_m) to ensure i32 scalar type
    c0 = start_m * 0
    desc_o.store([qo_offset_y.to(tl.int32), c0.to(tl.int32)], acc.to(tl.float16))
