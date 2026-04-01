import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim:int,max_seq_len:int = 2048,base:float=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self._set_cos_sin_cache(seq_len = max_seq_len)

    def _set_cos_sin_cache(self, seq_len :int):
        """预计算位置 m 的 cos(mθ), sin(mθ)"""
        pos = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", pos, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)
        
    def forward(self,q:torch.Tensor,k:torch.Tensor,seq_dim:int = 1) -> tuple[torch.Tensor, torch.Tensor]: 
        """
        q/k shape: [batch, seq_len, num_heads, head_dim]
        返回: 旋转后的 q, k
        """
        seq_len = q.shape[seq_dim]
        cos = self._set_cos_sin_cache[:seq_len,:]
        sin = self._set_cos_sin_cache[:seq_len,:]

        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
        sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]

        q_rot = self._rotate(q,cos,sin)
        k_rot = self._rotate(k,cos,sin)

        return  q_rot, k_rot
    
    def _rotate(self,x:torch.Tensor,cos:torch.Tensor,sin:torch.Tensor) -> torch.Tensor:
        x1 = x[...,::2]  # 偶数维
        x2 = x[...,1::2] # 奇数维
        x_rot = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return x_rot