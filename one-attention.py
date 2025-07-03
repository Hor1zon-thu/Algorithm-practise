import torch
from torch import nn
import math


class Onehead_Atten(nn.Module):
    def __init__(self, emd_size, q_k_size, v_size):
        super().__init__()

        # 初始化矩阵Wk,Wv,Wq
        # 注意x为(batch_size,q_seq_len,emd_size),且要实现(Q*KT)所以，Q(q_len,q_k_size),K为(k_len,q_k_size)
        self.Wk = nn.Linear(emd_size, q_k_size)
        self.Wq = nn.Linear(emd_size, q_k_size)
        self.Wv = nn.Linear(emd_size, v_size)

        # softmax((Q*KT/sqrt(dk)))*V
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_q, x_k_v, mask=None):
        q = self.Wq(x_q)  # (batch_size,q_len,q_k_size)
        k = self.Wk(x_k_v)  # (batch_size,k_v_len,q_k_size)
        v = self.Wv(x_k_v)  # (batch_size,k_v_len,v_size)

        # 为了便于相乘对K转置
        k = k.transpose(1, 2)  # (batch_size, q_k_size, k_v_len)

        # 计算注意力分数 (Q*K^T)/√dk
        attention_scores = torch.matmul(q, k) / math.sqrt(q.size(-1))

        # 应用掩码（如果提供）
        if mask is not None:
            # 注意：mask中为True的位置表示需要被mask掉
            attention_scores = attention_scores.masked_fill(mask, -1e9)

        # 应用softmax获取注意力权重
        attention_weights = self.softmax(attention_scores)

        # 加权求和得到输出
        output = torch.matmul(attention_weights, v)  # (batch_size, q_len, v_size)

        return output

if __name__ == '__main__':
    batch_size = 1
    emd_size = 128
    seq_len = 5
    q_k_size = 128
    v_size = 128
    x = torch.rand(size = (batch_size,seq_len,emd_size),dtype=torch.float)

    self_atten = Onehead_Atten(emd_size =emd_size,q_k_size=q_k_size,v_size=v_size)
    mask = torch.randn(size=(batch_size,seq_len,seq_len))
    mask = mask.bool()
    print("单头注意力的结果",self_atten(x,x,mask).size())
