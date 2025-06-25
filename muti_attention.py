import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Multi_attention(nn.Module):
    def __init__(self,emd_size,q_k_size,v_size,head_num,dropout = 0.0):
        super(Multi_attention,self).__init__()
        # 输入的x为(batch_size,seq_len,emd_size)
        self.head_num =head_num
        self.Wq = nn.Linear(emd_size,q_k_size*head_num)
        self.Wk = nn.Linear(emd_size,q_k_size*head_num)
        self.Wv = nn.Linear(emd_size,v_size*head_num)
        self.head_dim = emd_size // head_num
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(v_size*head_num,emd_size)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self,x_q,x_k_v,mask):
        batch_size,seq_len_q = x_q.size(0),x_q.size(1)
        seq_len_k_v =x_k_v.size(1)

        q = self.Wq(x_q)
        k = self.Wk(x_k_v)
        v = self.Wv(x_k_v)

        q = q.view(batch_size,seq_len_q,self.head_num,-1).transpose(1,2)
        k = k.view(batch_size,seq_len_k_v,self.head_num,-1).transpose(1,2)
        v = v.view(batch_size,seq_len_k_v,self.head_num,-1).transpose(1,2)

        k = k.transpose(2,3)

        atten_scores = torch.matmul(q,k)/math.sqrt(q.size(-1))   # [batch_size, num_heads, seq_len_q, seq_len_kv]
        if mask is not None :
            atten_scores = atten_scores.masked_fill(mask.unsqueeze(1) == False,-1e9)

        weights = self.softmax(atten_scores)
        weights = self.dropout(weights)

        outputs =torch.matmul(weights,v)  # [batch_size, num_heads, seq_len_q, head_dim]

        outputs = outputs.transpose(1, 2).contiguous().view(batch_size, seq_len_q, -1)

        return self.out_proj(outputs)

if __name__ == '__main__':

    batch_size = 1
    emd_size = 128
    seq_len = 5
    q_k_size = emd_size//8
    v_size = emd_size//8

    x = torch.rand(size=(batch_size, seq_len, emd_size), dtype=torch.float)

    self_atten = Multi_attention(emd_size=emd_size, q_k_size=q_k_size, v_size=v_size,head_num=8)

    mask = torch.randn(size=(batch_size, seq_len, seq_len))
    mask = mask.bool()

    print('多头的自注意力结果', self_atten(x, x, mask).size())

    batch_size = 1  # 批量也就是句子的数量
    emd_size = 128  # 一个token嵌入的维度
    q_seq_len = 5  # q源的token长度
    q_k_size = emd_size//8  # q和k的嵌入维度/head
    k_v_seq_len = 7  # k_v源的token长度
    v_size = emd_size//8  # v的嵌入维度/head
    head=8 # 头的数量

    x_q = torch.rand(size=(batch_size, q_seq_len, emd_size), dtype=torch.float)
    x_k_v = torch.rand(size=(batch_size, k_v_seq_len, emd_size), dtype=torch.float)

    cross_atten = Multi_attention(emd_size=emd_size, q_k_size=q_k_size, v_size=v_size,head_num=head)
    # 初始化mask(batch,len_k,len_q)
    mask = torch.randn(size=(batch_size, q_seq_len, k_v_seq_len))
    mask = mask.bool()

    print('多头的交叉注意力结果', cross_atten(x_q, x_k_v, mask).size())