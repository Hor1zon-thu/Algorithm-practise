import torch.nn as nn
import torch
import torch.nn.functional as F

class One_attention(nn.Module):
    def __init__(self,d_model,mask,dropout = 0.01):
        super().__init__()
        self.d_model = d_model
        self.mask = mask
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax

        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)


    def forward(self,x):
        batch_size,len,d_model = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        K = K.transpose(2,3)
        score = torch.matmul(Q,K)/torch.sqrt(d_model)

        if mask is not None:
            score = score.masked_fill(mask,-1e9)

        atten_weight = self.softmax(score,dim = -1)
        atten_weight = self.dropout(atten_weight)

        output = torch.matmul(atten_weight,V)

        return output

