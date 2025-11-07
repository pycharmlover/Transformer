"""
Encoder Layer 和 Encoder（多层叠加）
使用 Pre-LN（在每个子层前 LayerNorm），即：
y = x + Sublayer(LayerNorm(x))

使用T5风格的相对位置编码
"""

import torch
import torch.nn as nn
from src.models.attention import MultiHeadAttention
from src.models.ffn import PositionwiseFeedForward
from src.models.relative_positional_encoding import RelativePositionBias

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, 
                 attention_type: str = "standard"):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈层维度
            dropout: dropout 概率
            attention_type: 注意力机制类型（standard/local_sparse/strided_sparse/block_sparse/linear/performer）
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, attention_type=attention_type)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None, relative_position_bias: torch.Tensor = None):
        """
        x: (B, T, d_model)
        src_mask: (B, T_k) 或 (B, 1, 1, T_k)
        relative_position_bias: (1, num_heads, T, T) 相对位置偏置
        """
        # Pre-LN
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, mask=src_mask, causal=False, relative_position_bias=relative_position_bias)
        x = x + self.dropout(attn_out)

        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1,
                 attention_type: str = "standard"):
        """
        Args:
            num_layers: Encoder 层数
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈层维度
            dropout: dropout 概率
            attention_type: 注意力机制类型
        """
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, attention_type=attention_type) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)  # 最后再 norm 一次
        
        # 相对位置偏置（Encoder使用双向）
        self.relative_position_bias = RelativePositionBias(
            num_heads=num_heads,
            num_buckets=32,
            max_distance=128,
            bidirectional=True
        )

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None):
        """
        Args:
            x: (B, T, d_model)
            src_mask: (B, T) 或 (B, 1, 1, T)
        """
        T = x.size(1)
        
        # 计算相对位置偏置（只需要计算一次，所有层共享）
        relative_bias = self.relative_position_bias(T, T, device=x.device)
        
        # 通过所有层
        for layer in self.layers:
            x = layer(x, src_mask=src_mask, relative_position_bias=relative_bias)
        
        x = self.norm(x)
        return x

