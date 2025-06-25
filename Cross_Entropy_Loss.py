import torch
import torch.nn.functional as F


def cross_entropy(input,target,limit = 1e-12):
    input = torch.clamp(input,limit,1.-limit)
    output = - torch.sum(target * torch.log(input),dtype =torch.float32)/ target.size()[0]
    return output

input = torch.tensor([[0.1, 0.2, 0.7],
                      [0.3, 0.6, 0.1]], dtype=torch.float32)

target = torch.tensor([[0,0,1],
                       [0,1,0]],dtype = torch.float32)

loss1 = cross_entropy(input,target)

classes = torch.argmax(target, dim=1)  # [2, 1]
logits = torch.log(input)  # 转换为logits（因为输入已经是softmax后的结果）
loss2 = F.nll_loss(logits, classes)


print(f"自定义交叉熵损失:{loss1.item():.6f}")
print(f"Pytorch实现的损失: {loss2.item():.6f}")
