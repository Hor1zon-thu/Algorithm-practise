import multi_attention
import torch
import torch.nn as nn
import  torch.nn.functional as F
import math

#位置编码器
class PositionalEncoder(nn.Module):
    def __init__(self,d_model,max_Seq_len = 80):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_Seq_len,d_model)
        for pos in range(max_Seq_len):
            for i in range(0,d_model,2):
                pe[pos,i] = math.sin(pos/10000**((2*i)/d_model))
                pe[pos,i+1] = math.cos(pos/10000**((2* (i+1))/d_model))

        pe.unsqueeze(0)
        self.register_buffer("pe",pe)

    def forward(self,x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.shape[1]
        x = x + self.pe[:,:seq_len]
        return x

#前馈神经网络
class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff = 2048,dropout = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff,d_model)
        self.relu = F.relu
    def forward(self,x):
        x = self.dropout(self.relu(self.linear1(x)))
        x = self.linear2(x)
        
        return x

#层归一化
class NormLayer(nn.Module):
    def __init__(self,d_model,esp = 1e-6):
        super().__init__()
        