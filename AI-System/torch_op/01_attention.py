import torch
import torch.nn.functional as F
from torch import nn
import math

class SimpleAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim  # dim是特征长度
        self.head_dim = dim // num_heads

    def forward(self, query, key, value, mask=None):
        B, N, _ = query.shape

       # 将输入的Q、K、V拆分为多头。 num_heads多头、head_dim每个头的特征维度、N是序列长度
        query = query.view(B, N, self.num_heads, self.head_dim)
        value = value.view(B, N, self.num_heads, self.head_dim)
        key = key.view(B, N, self.num_heads, self.head_dim)

        query = query.transpose(1, 2)
        value = value.transpose(1, 2)
        key = key.transpose(1, 2)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 如果提供了mask，将其应用到注意力分数上
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 计算加权的V
        output = torch.matmul(attn_weights, value)
        output = output.transpose(1, 2)

        # 合并多头
        output = output.contiguous().view(B, N, -1)

        return output


if __name__ == '__main__':
    query = torch.randn(2, 20, 768)
    key = torch.randn(2, 20, 768)
    value = torch.randn(2, 20, 768)

    attention = SimpleAttention(768, 8)
    output = attention(query, key, value)
    print(output.shape)
