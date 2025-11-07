"""
Multi-Head Attention 模块
- 返回 attention 输出和 attention 权重（可选）
- 输入 X 的形状: (B, T, d_model)
- 支持多种注意力机制：标准注意力、稀疏注意力、线性注意力
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from src.models.sparse_attention import LocalSparseAttention, StridedSparseAttention, BlockSparseAttention
from src.models.linear_attention import LinearAttention, CausalLinearAttention, PerformerAttention

class ScaledDotProductAttention(nn.Module):
    """
    带相对位置偏置的 Scaled Dot-Product Attention
    """
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        relative_position_bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q: (B, head_num, T_q, head_dim)
        k: (B, head_num, T_k, head_dim)
        v: (B, head_num, T_k, head_dim)
        mask: 可以是 None，或形状兼容于 (B, 1, 1, T_k) 或 (B, 1, T_q, T_k)
              mask 中 0 表示被 mask（不可见），1 表示可见
        relative_position_bias: (1, head_num, T_q, T_k) 或 None
              相对位置偏置，直接加到注意力分数上
        返回:
          output: (B, head_num, T_q, head_dim)
          attn: (B, head_num, T_q, T_k)
        """
        head_dim = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # 加上相对位置偏置（T5的核心机制）
        if relative_position_bias is not None:
            scores = scores + relative_position_bias

        if mask is not None:
            # mask 里 0 表示要被屏蔽，把这些位置设为 -inf（用 -1e9 近似）
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):
    """
    多头注意力模块
    输入:
      x_q, x_k, x_v: 形状均为 (B, T, d_model)
    mask:
      - padding mask: shape (B, T_k) 或 (B, 1, 1, T_k)
      - causal mask（如需要）通过参数 causal=True 由模块内部生成并与 padding mask 合并
    
    支持多种注意力机制：
      - standard: 标准的 O(n^2) 注意力（支持T5相对位置编码）
      - local_sparse: 局部稀疏注意力（窗口注意力）
      - strided_sparse: 跨步稀疏注意力
      - block_sparse: 块稀疏注意力
      - linear: 线性注意力 O(n)
      - causal_linear: 因果线性注意力 O(n)（用于decoder）
      - performer: Performer 注意力（FAVOR+）
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, 
                 attention_type: str = "standard", 
                 sparse_window_size: int = 128,
                 sparse_stride: int = 16,
                 sparse_block_size: int = 64):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: dropout 概率
            attention_type: 注意力机制类型
            sparse_window_size: 稀疏注意力的窗口大小（用于 local_sparse）
            sparse_stride: 跨步大小（用于 strided_sparse）
            sparse_block_size: 块大小（用于 block_sparse）
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attention_type = attention_type
        
        # 线性层：W_q, W_k, W_v, W_o
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # 根据 attention_type 选择注意力机制
        if attention_type == "standard":
            self.attention = ScaledDotProductAttention(dropout=dropout)
        elif attention_type == "local_sparse":
            self.attention = LocalSparseAttention(window_size=sparse_window_size, dropout=dropout)
        elif attention_type == "strided_sparse":
            self.attention = StridedSparseAttention(stride=sparse_stride, dropout=dropout)
        elif attention_type == "block_sparse":
            self.attention = BlockSparseAttention(block_size=sparse_block_size, dropout=dropout)
        elif attention_type == "linear":
            self.attention = LinearAttention(feature_dim=self.head_dim, dropout=dropout)
        elif attention_type == "causal_linear":
            self.attention = CausalLinearAttention(feature_dim=self.head_dim, dropout=dropout)
        elif attention_type == "performer":
            self.attention = PerformerAttention(head_dim=self.head_dim, 
                                               num_random_features=max(self.head_dim, 256), 
                                               dropout=dropout)
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}. "
                           f"Supported types: standard, local_sparse, strided_sparse, "
                           f"block_sparse, linear, causal_linear, performer")
        
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self,x: torch.Tensor) -> torch.Tensor:
        """
        将 (B, T, d_model) -> (B, head_num, T, head_dim)
        """
        B, T, _ = x.size()
        x = x.view(B, T, self.num_heads, self.head_dim).transpose(1,2) # (B, head_num, T, head_dim)
        return x

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 (B, h, T, d_k) -> (B, T, d_model)
        """
        B, h, T, d_k = x.size()
        x = x.transpose(1, 2).contiguous().view(B, T, h * d_k)
        return x

    def forward(
        self, 
        x_q: torch.Tensor, 
        x_k: torch.Tensor, 
        x_v: torch.Tensor, 
        mask: Optional[torch.Tensor] = None, 
        causal: bool = False,
        relative_position_bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x_q, x_k, x_v: (B, T, d_model)
        mask: padding mask with shape (B, T_k) 或 (B, 1, 1, T_k) 或 (B, 1, T_q, T_k)
              数值为 1（可见）或 0（屏蔽）
        causal: 如果 True，则在 attention 中加入 causal 上三角屏蔽
        relative_position_bias: (1, num_heads, T_q, T_k) 或 None
              相对位置偏置，用于T5风格的相对位置编码
        返回:
          out: (B, T_q, d_model)
          attn: (B, h, T_q, T_k) 或 None（线性注意力不返回注意力权重）
        """
        B, T_q, _ = x_q.size()
        T_k = x_k.size(1)

        # 线性投影
        q = self.w_q(x_q)  # (B, T_q, d_model)
        k = self.w_k(x_k)  # (B, T_k, d_model)
        v = self.w_v(x_v)  # (B, T_k, d_model)

        # 拆成多头
        q = self._split_heads(q)  # (B, h, T_q, d_k)
        k = self._split_heads(k)  # (B, h, T_k, d_k)
        v = self._split_heads(v)  # (B, h, T_k, d_k)

        # 处理 mask：转换为 (B, 1, 1, T_k) 或 (B, 1, T_q, T_k)
        attn_mask = None
        if mask is not None:
             # 支持 mask 是 (B, T_k) 或 (B, 1, 1, T_k) 或 (B, 1, T_q, T_k)
             if mask.dim() == 2:
                attn_mask = mask.unsqueeze(1).unsqueeze(1) # (B,1,1,T_k)
             elif mask.dim() == 3:
                attn_mask = mask.unsqueeze(1) # (B,1,T_q,T_k)
             else:
                attn_mask = mask # (B,1,1,T_k)
            
        # 如果需要 causal mask（上三角屏蔽），构造并与 attn_mask 合并
        # 注意：causal_linear 类型已经内部处理了因果性，不需要额外的 causal mask
        if causal and self.attention_type in ["standard", "linear", "performer"]:
            # causal_mask: (1, 1, T_q, T_k)
            causal_mask = torch.tril(torch.ones((T_q, T_k), dtype=torch.uint8, device=q.device)).unsqueeze(0).unsqueeze(0) # (1,1,T_q,T_k)
            if attn_mask is None:
                attn_mask = causal_mask
            else:
                # attn_mask 与 causal_mask 逻辑与：只有同时为 1 的位置可见
                attn_mask = attn_mask & causal_mask
            
        # 注意力计算
        # 对于标准注意力，传递相对位置偏置
        if self.attention_type == "standard" and relative_position_bias is not None:
            attn_output, attn_weights = self.attention(q, k, v, mask=attn_mask, relative_position_bias=relative_position_bias)
        else:
            # 对于稀疏和线性注意力，不使用相对位置偏置
            attn_output, attn_weights = self.attention(q, k, v, mask=attn_mask)

        # 合并多头
        out = self._combine_heads(attn_output) # (B, T_q, d_model)
        out = self.w_o(out)
        out = self.dropout(out)
        return out, attn_weights
                






