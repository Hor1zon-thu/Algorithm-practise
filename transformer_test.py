import torch
from transformer import Transformer

# 参数设置
src_vocab_size = 1000
trg_vocab_size = 1000
d_model = 512
n_layers = 2
n_heads = 8
dropout = 0.1
batch_size = 2
src_seq_len = 10
trg_seq_len = 8

# 初始化模型
model = Transformer(
    src_vocab=src_vocab_size,
    trg_vocab=trg_vocab_size,
    N=n_layers,
    d_model=d_model,
    heads=n_heads,
    dropout=dropout
)

# 输入数据
src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
trg = torch.randint(0, trg_vocab_size, (batch_size, trg_seq_len))

## 1. 编码器自注意力掩码：[batch, src_len, src_len]
encoder_mask = torch.ones(batch_size, src_seq_len, src_seq_len).bool()  # [2,10,10]

## 2. 解码器自注意力掩码：[batch, trg_len, trg_len]（下三角，屏蔽未来信息）
decoder_self_mask = torch.tril(torch.ones(trg_seq_len, trg_seq_len)).bool()  # [8,8]
decoder_self_mask = decoder_self_mask.unsqueeze(0).expand(batch_size, -1, -1)  # [2,8,8]

## 3. 交叉注意力掩码：[batch, trg_len, src_len]（对齐Q和K/V的长度）
cross_mask = torch.ones(batch_size, trg_seq_len, src_seq_len).bool()

# 前向传播
with torch.no_grad():
    output = model(
        src=src,
        trg=trg,
        encoder_mask=encoder_mask,
        decoder_mask=decoder_self_mask,
        cross_mask=cross_mask
    )


print(f"源序列形状: {src.shape}")
print(f"目标序列形状: {trg.shape}")
print(f"输出形状: {output.shape}")  # 预期 [2, 8, 1000]