"""
Decoder Layer 和 Decoder（用于 Seq2Seq 的自回归解码器）
注意: decoder 里包含两种 attention:
  1) masked self-attention（causal=True）使用单向相对位置偏置
  2) encoder-decoder cross attention（query=decoder, key/value=encoder output）使用双向相对位置偏置
同样采用 Pre-LN 风格。
"""

import torch
import torch.nn as nn
from src.models.attention import MultiHeadAttention
from src.models.ffn import PositionwiseFeedForward
from src.models.relative_positional_encoding import RelativePositionBiasForDecoder

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1,
                 attention_type: str = "standard"):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈层维度
            dropout: dropout 概率
            attention_type: 注意力机制类型（用于 self-attention 和 cross-attention）
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, attention_type=attention_type)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, attention_type=attention_type)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        enc_output: torch.Tensor, 
        tgt_mask: torch.Tensor = None, 
        memory_mask: torch.Tensor = None,
        self_attn_bias: torch.Tensor = None,
        cross_attn_bias: torch.Tensor = None
    ):
        """
        x: (B, T_tgt, d_model)
        enc_output: (B, T_src, d_model)
        tgt_mask: 为decoder self-attention掩码 (padding mask + causal)
        memory_mask: 为encoder-decoder attention掩码 (对encoder keys的padding mask)
        self_attn_bias: (1, num_heads, T_tgt, T_tgt) decoder self-attention的相对位置偏置
        cross_attn_bias: (1, num_heads, T_tgt, T_src) cross-attention的相对位置偏置
        """
        # 1) 掩码self-attention (causal=True)
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, mask=tgt_mask, causal=True, relative_position_bias=self_attn_bias)
        x = x + self.dropout(attn_out)

        # 2) encoder-decoder cross attention
        x_norm = self.norm2(x)
        attn_out, _ = self.cross_attn(x_norm, enc_output, enc_output, mask=memory_mask, causal=False, relative_position_bias=cross_attn_bias)
        x = x + self.dropout(attn_out)

        # 3) FFN
        x_norm = self.norm3(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1,
                 attention_type: str = "standard"):
        """
        Args:
            num_layers: Decoder 层数
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈层维度
            dropout: dropout 概率
            attention_type: 注意力机制类型
        """
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout, attention_type=attention_type) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
        # 相对位置偏置模块（包含self-attention和cross-attention两种）
        self.relative_position_bias = RelativePositionBiasForDecoder(
            num_heads=num_heads,
            num_buckets=32,
            max_distance=128
        )
    
    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None):
        """
        Args:
            x: (B, T_tgt, d_model)
            enc_output: (B, T_src, d_model)
            tgt_mask: decoder self-attention mask
            memory_mask: cross-attention mask
        """
        T_tgt = x.size(1)
        T_src = enc_output.size(1)
        
        # 计算相对位置偏置（只需要计算一次，所有层共享）
        self_attn_bias = self.relative_position_bias.get_self_attn_bias(T_tgt, T_tgt, device=x.device)
        cross_attn_bias = self.relative_position_bias.get_cross_attn_bias(T_tgt, T_src, device=x.device)
        
        # 通过所有层
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask=tgt_mask, memory_mask=memory_mask, 
                     self_attn_bias=self_attn_bias, cross_attn_bias=cross_attn_bias)
        
        x = self.norm(x)
        return x