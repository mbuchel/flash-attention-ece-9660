import torch
import torch.nn as nn

from flash_attn import flash_attn_func

from .triton.wrappers import triton_add, triton_matmul, triton_softmax, triton_flash_v1, triton_flash_v2

class TritonFusedAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, version="v1"):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.scale = 1.0 / (self.head_dim ** 0.5)
        self.version = version

    def forward(self, x):
        b, s, d = x.shape
        q = self.w_q(x).view(b, s, self.n_head, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(b, s, self.n_head, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(b, s, self.n_head, self.head_dim).transpose(1, 2)
        
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        
        if self.version == "v2":
            attn_out = triton_flash_v2(q, k, v, self.scale)
        else:
            attn_out = triton_flash_v1(q, k, v, self.scale)
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, s, d)
        return self.w_o(attn_out)

class TritonDecomposedAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.scale = 1.0 / (self.head_dim ** 0.5)

    def forward(self, x):
        b, s, d = x.shape
        q = self.w_q(x).view(b, s, self.n_head, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(b, s, self.n_head, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(b, s, self.n_head, self.head_dim).transpose(1, 2)
        
        attn_out = torch.empty_like(q)
        for i in range(b):
            for j in range(self.n_head):
                qi = q[i, j].contiguous()          # [S, D_head]
                ki = k[i, j]                       # [S, D_head]
                vi = v[i, j].contiguous()          # [S, D_head]
                
                # Score = Q @ K.T
                ki_t = ki.transpose(0, 1).contiguous()
                score = triton_matmul(qi, ki_t)
                score = score * self.scale
                
                # Softmax
                probs = triton_softmax(score)
                
                # Out = Probs @ V
                out_head = triton_matmul(probs, vi)
                attn_out[i, j] = out_head
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, s, d)
        return self.w_o(attn_out)

class PyTorchAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True, bias=False)
    def forward(self, x):
        return self.attn(x, x, x, need_weights=False)[0]

class OfficialFlashAttention(torch.nn.Module):
    def __init__(self, heads, dim):
        super().__init__()
        self.heads = heads
        self.dim = dim
    def forward(self, x):
        b, s, d = x.shape
        qkv = x.view(b, s, self.heads, self.dim // self.heads)
        out = flash_attn_func(qkv, qkv, qkv, dropout_p=0.0, softmax_scale=None, causal=False)
        return out.view(b, s, d)

