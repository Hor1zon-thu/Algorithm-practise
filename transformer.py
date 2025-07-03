from multi_attention import Multi_attention
import torch
import torch.nn as nn
import torch.nn.functional as F
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

        pe = pe.unsqueeze(0)
        self.register_buffer("pe",pe)

    def forward(self,x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.shape[1]
        x = x + self.pe[:,:seq_len,:]
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
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size)) #可学习的缩放参数
        self.bias= nn.Parameter(torch.zeros(self.size)) #可学习的偏置参数
        self.esp = esp

    def forward(self,x):
        norm =self.alpha*(x -x.mean(dim=-1,keepdim=True))/(x.std(dim = -1,keepdim =True)+self.esp) + self.bias
        return norm

#编码器层
class EncoderLayers(nn.Module):
    def __init__(self,d_model,heads,dropout=0.1):
        super().__init__()
        head_dim = d_model // heads
        self.norm_1 = NormLayer(d_model)
        self.norm_2 = NormLayer(d_model)
        self.atten = Multi_attention(emd_size=d_model,q_k_size=head_dim,v_size=head_dim,head_num=heads,dropout = dropout)
        self.ff =FeedForward(d_model,dropout = dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self,x,mask):
        x2 = self.norm_1(x)
        x= x+self.dropout_1(self.atten(x2,x2,mask))
        x2 =self.norm_2(x)
        x =x +self.dropout_2(self.ff(x2))

        return x


#编码器
class Encoder(nn.Module):
    def __init__(self,vocal_size,d_model,N,heads,dropout):
        super().__init__()
        self.N = N
        self.embed = nn.Embedding(vocal_size,d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = nn.ModuleList([EncoderLayers(d_model,heads,dropout)for _ in range(N)])
        self.norm = NormLayer(d_model)

    def forward(self,src,mask):
        x = self.embed(src)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x,mask)

        x = self.norm(x)
        return x



#解码器层
class DecoderLayers(nn.Module):
    def __init__(self,d_model,heads,dropout = 0.1):
        super().__init__()
        head_dim = d_model // heads
        self.norm_1 = NormLayer(d_model)
        self.norm_2 = NormLayer(d_model)
        self.norm_3 = NormLayer(d_model)
        self.atten1 = Multi_attention(emd_size=d_model,q_k_size=head_dim,v_size=head_dim,head_num=heads,dropout = dropout)
        self.atten2 = Multi_attention(emd_size=d_model,q_k_size=head_dim,v_size=head_dim,head_num=heads,dropout = dropout)
        self.ff = FeedForward(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self,x,encoder_outputs,src_mask,trg_mask):
        x2 = self.norm_1(x)
        x = x +self.dropout1(self.atten1(x2,x2,trg_mask))
        x2 = self.norm_2(x)
        x = x+ self.dropout2(self.atten2(x2,encoder_outputs,src_mask))
        x2 = self.norm_3(x)
        x = x +self.dropout3(self.ff(x2))
        return x

#解码器
class Decoder(nn.Module):
    def __init__(self,vocal_size,d_model,N,heads,dropout):
        super().__init__()
        self.N = N
        self.embed = nn.Embedding(vocal_size,d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = nn.ModuleList([DecoderLayers(d_model,heads,dropout)for _ in range(N)])
        self.norm = NormLayer(d_model)

    def forward(self,trg,e_outputs,cross_mask,trg_mask):
        x =self.embed(trg)
        x = self.pe(x)
        for layer in self.layers :
            x = layer(x,e_outputs,cross_mask,trg_mask)
        x = self.norm(x)

        return x



class Transformer(nn.Module):
    def __init__(self,src_vocab,trg_vocab,N,d_model,heads,dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab,d_model,N,heads,dropout)
        self.decoder = Decoder(trg_vocab,d_model,N,heads,dropout)
        self.out = nn.Linear(d_model,trg_vocab)

    def forward(self,src,trg,encoder_mask,decoder_mask,cross_mask):
        """
             :param encoder_mask: 编码器自注意力掩码，形状 [batch, src_len, src_len]
             :param decoder_self_mask: 解码器自注意力掩码，形状 [batch, trg_len, trg_len]
             :param cross_mask: 解码器交叉注意力掩码，形状 [batch, trg_len, src_len]
        """
        e_outpus =self.encoder(src,encoder_mask)
        d_outputs =self.decoder(trg,e_outpus,cross_mask,decoder_mask)

        output = self.out(d_outputs)
        return output


