"""
Sparse Attention (稀疏注意力)
通过限制每个 token 只关注局部窗口内的其他 token，将复杂度从 O(n^2) 降低到 O(n*w)
其中 w 是窗口大小
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class LocalSparseAttention(nn.Module):
    """
    局部稀疏注意力
    每个位置只关注其周围固定窗口大小内的位置
    适用于序列数据，其中局部信息更重要
    """
    def __init__(self, window_size: int = 128, dropout: float = 0.0):
        """
        Args:
            window_size: 局部窗口大小（每个位置向前和向后关注的范围）
            dropout: dropout 概率
        """
        super().__init__()
        self.window_size = window_size
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> tuple:
        """
        Args:
            q: (B, num_heads, T_q, head_dim)
            k: (B, num_heads, T_k, head_dim)
            v: (B, num_heads, T_k, head_dim)
            mask: (B, 1, 1, T_k) 或 None
            
        Returns:
            output: (B, num_heads, T_q, head_dim)
            attn: (B, num_heads, T_q, T_k) - 稀疏的注意力权重
        """
        B, num_heads, T_q, head_dim = q.size()
        T_k = k.size(2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)  # (B, h, T_q, T_k)
        
        # 创建局部窗口 mask
        # 对于位置 i，只能关注 [i - window_size, i + window_size] 范围内的位置
        range_q = torch.arange(T_q, device=q.device).unsqueeze(1)  # (T_q, 1)
        range_k = torch.arange(T_k, device=k.device).unsqueeze(0)  # (1, T_k)
        
        # 计算相对距离
        distance = torch.abs(range_q - range_k)  # (T_q, T_k)
        
        # 创建窗口 mask：距离超过 window_size 的位置设为 0
        window_mask = (distance <= self.window_size).float()
        window_mask = window_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T_q, T_k)
        
        # 应用窗口 mask（超出窗口的位置设为 -inf）
        scores = scores.masked_fill(window_mask == 0, -1e9)
        
        # 如果提供了额外的 mask（如 padding mask），也要应用
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax 和 dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 计算输出
        output = torch.matmul(attn, v)
        
        return output, attn


class StridedSparseAttention(nn.Module):
    """
    跨步稀疏注意力
    每个位置只关注固定步长的位置
    可以捕获长距离依赖，但使用更少的计算
    """
    def __init__(self, stride: int = 16, dropout: float = 0.0):
        """
        Args:
            stride: 步长（每隔 stride 个位置关注一次）
            dropout: dropout 概率
        """
        super().__init__()
        self.stride = stride
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> tuple:
        """
        Args:
            q: (B, num_heads, T_q, head_dim)
            k: (B, num_heads, T_k, head_dim)
            v: (B, num_heads, T_k, head_dim)
            mask: (B, 1, 1, T_k) 或 None
            
        Returns:
            output: (B, num_heads, T_q, head_dim)
            attn: (B, num_heads, T_q, T_k)
        """
        B, num_heads, T_q, head_dim = q.size()
        T_k = k.size(2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # 创建跨步 mask
        # 位置 i 可以关注：i 本身、以及 i % stride == 0 的所有位置
        range_q = torch.arange(T_q, device=q.device).unsqueeze(1)  # (T_q, 1)
        range_k = torch.arange(T_k, device=k.device).unsqueeze(0)  # (1, T_k)
        
        # 允许关注：1) 自己 2) stride 的倍数位置
        stride_mask = ((range_k % self.stride == 0) | (range_q == range_k)).float()
        stride_mask = stride_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T_q, T_k)
        
        # 应用跨步 mask
        scores = scores.masked_fill(stride_mask == 0, -1e9)
        
        # 应用额外的 mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax 和 dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 计算输出
        output = torch.matmul(attn, v)
        
        return output, attn


class BlockSparseAttention(nn.Module):
    """
    块稀疏注意力
    将序列分成多个块，每个位置只关注同一块和相邻块内的位置
    这是一种平衡局部和全局信息的方法
    """
    def __init__(self, block_size: int = 64, dropout: float = 0.0):
        """
        Args:
            block_size: 块大小
            dropout: dropout 概率
        """
        super().__init__()
        self.block_size = block_size
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> tuple:
        """
        Args:
            q: (B, num_heads, T_q, head_dim)
            k: (B, num_heads, T_k, head_dim)
            v: (B, num_heads, T_k, head_dim)
            mask: (B, 1, 1, T_k) 或 None
            
        Returns:
            output: (B, num_heads, T_q, head_dim)
            attn: (B, num_heads, T_q, T_k)
        """
        B, num_heads, T_q, head_dim = q.size()
        T_k = k.size(2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # 创建块 mask
        # 计算每个位置所属的块
        block_idx_q = torch.arange(T_q, device=q.device) // self.block_size  # (T_q,)
        block_idx_k = torch.arange(T_k, device=k.device) // self.block_size  # (T_k,)
        
        # 位置 i 可以关注块 block_idx_q[i] 及其相邻块（block_idx_q[i] - 1, block_idx_q[i] + 1）中的所有位置
        block_idx_q = block_idx_q.unsqueeze(1)  # (T_q, 1)
        block_idx_k = block_idx_k.unsqueeze(0)  # (1, T_k)
        
        # 允许关注相邻块
        block_mask = (torch.abs(block_idx_q - block_idx_k) <= 1).float()
        block_mask = block_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T_q, T_k)
        
        # 应用块 mask
        scores = scores.masked_fill(block_mask == 0, -1e9)
        
        # 应用额外的 mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax 和 dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 计算输出
        output = torch.matmul(attn, v)
        
        return output, attn

