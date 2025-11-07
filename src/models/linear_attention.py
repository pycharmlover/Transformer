"""
Linear Attention (线性注意力)
通过核函数技巧将注意力复杂度从 O(n^2) 降低到 O(n)
基于 "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" 论文
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class LinearAttention(nn.Module):
    """
    线性注意力机制
    使用特征映射将 softmax attention 近似为线性操作
    复杂度：O(n * d^2) 而不是 O(n^2 * d)
    """
    def __init__(self, feature_dim: int = 256, eps: float = 1e-6, dropout: float = 0.0):
        """
        Args:
            feature_dim: 特征映射的维度（通常设为 head_dim 或更大）
            eps: 数值稳定性的小常数
            dropout: dropout 概率
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.eps = eps
        self.dropout = nn.Dropout(dropout)
        
    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        特征映射函数：使用 ELU + 1 作为核函数
        这是一种常用的非负特征映射，可以保证注意力权重非负
        
        Args:
            x: (B, num_heads, T, head_dim)
            
        Returns:
            features: (B, num_heads, T, head_dim) - 映射后的特征
        """
        # 使用 ELU(x) + 1 作为特征映射
        # 这保证了非负性，类似于 softmax 的效果
        return F.elu(x) + 1
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> tuple:
        """
        线性注意力的前向传播
        
        标准注意力：Attention(Q, K, V) = softmax(QK^T / sqrt(d))V
        线性注意力：Attention(Q, K, V) = phi(Q)(phi(K)^T V) / phi(Q)phi(K)^T
        
        其中 phi 是特征映射函数
        
        Args:
            q: (B, num_heads, T_q, head_dim)
            k: (B, num_heads, T_k, head_dim)
            v: (B, num_heads, T_k, head_dim)
            mask: (B, 1, 1, T_k) - padding mask（注意：线性注意力处理 mask 的方式与标准注意力不同）
            
        Returns:
            output: (B, num_heads, T_q, head_dim)
            attn: None（线性注意力不显式计算注意力矩阵，这里返回 None 以保持接口一致）
        """
        B, num_heads, T_q, head_dim = q.size()
        T_k = k.size(2)
        
        # 应用特征映射
        # phi(Q): (B, num_heads, T_q, head_dim)
        # phi(K): (B, num_heads, T_k, head_dim)
        q_prime = self.feature_map(q)
        k_prime = self.feature_map(k)
        
        # 如果有 mask，需要将被 mask 的位置的 k 和 v 置零
        if mask is not None:
            # mask: (B, 1, 1, T_k) -> (B, 1, T_k, 1)
            mask = mask.squeeze(2).unsqueeze(-1)  # (B, 1, T_k, 1)
            k_prime = k_prime * mask
            v = v * mask
        
        # 计算 K^T V：先计算 key 和 value 的乘积
        # k_prime^T: (B, num_heads, head_dim, T_k)
        # v: (B, num_heads, T_k, head_dim)
        # kv: (B, num_heads, head_dim, head_dim)
        kv = torch.matmul(k_prime.transpose(-2, -1), v)
        
        # 计算归一化项：K^T 1（每个 key 的特征和）
        # k_prime^T: (B, num_heads, head_dim, T_k)
        # ones: (B, num_heads, T_k, 1)
        # z: (B, num_heads, head_dim, 1)
        k_sum = k_prime.sum(dim=2, keepdim=True)  # (B, num_heads, 1, head_dim)
        k_sum = k_sum.transpose(-2, -1)  # (B, num_heads, head_dim, 1)
        
        # 计算输出：Q (K^T V)
        # q_prime: (B, num_heads, T_q, head_dim)
        # kv: (B, num_heads, head_dim, head_dim)
        # output: (B, num_heads, T_q, head_dim)
        output = torch.matmul(q_prime, kv)
        
        # 归一化：除以 Q K^T 1
        # q_prime: (B, num_heads, T_q, head_dim)
        # k_sum: (B, num_heads, head_dim, 1)
        # z: (B, num_heads, T_q, 1)
        z = torch.matmul(q_prime, k_sum)
        z = z + self.eps  # 避免除零
        
        # 归一化输出
        output = output / z
        
        # 应用 dropout
        output = self.dropout(output)
        
        # 线性注意力不显式计算注意力权重矩阵（因为它永远不会被具体化）
        # 返回 None 表示没有注意力权重
        return output, None


class CausalLinearAttention(nn.Module):
    """
    因果线性注意力（用于自回归解码）
    通过累积计算保持因果性，适用于 decoder
    """
    def __init__(self, feature_dim: int = 256, eps: float = 1e-6, dropout: float = 0.0):
        """
        Args:
            feature_dim: 特征映射的维度
            eps: 数值稳定性的小常数
            dropout: dropout 概率
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.eps = eps
        self.dropout = nn.Dropout(dropout)
        
    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """特征映射函数"""
        return F.elu(x) + 1
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> tuple:
        """
        因果线性注意力的前向传播
        使用累积和来保持因果性（每个位置只能看到之前的位置）
        
        Args:
            q: (B, num_heads, T_q, head_dim)
            k: (B, num_heads, T_k, head_dim)
            v: (B, num_heads, T_k, head_dim)
            mask: padding mask（可选）
            
        Returns:
            output: (B, num_heads, T_q, head_dim)
            attn: None
        """
        B, num_heads, T_q, head_dim = q.size()
        T_k = k.size(2)
        
        # 应用特征映射
        q_prime = self.feature_map(q)  # (B, num_heads, T_q, head_dim)
        k_prime = self.feature_map(k)  # (B, num_heads, T_k, head_dim)
        
        # 如果有 padding mask，应用它
        if mask is not None:
            mask = mask.squeeze(2).unsqueeze(-1)  # (B, 1, T_k, 1)
            k_prime = k_prime * mask
            v = v * mask
        
        # 对于因果注意力，我们需要逐步累积 K^T V
        # 这样每个查询位置只能看到它之前（包括自己）的键值对
        
        # 创建因果 mask（下三角矩阵）
        causal_mask = torch.tril(torch.ones(T_q, T_k, device=q.device))  # (T_q, T_k)
        
        # 扩展 k, v 以便进行因果累积
        # k_prime: (B, num_heads, T_k, head_dim)
        # v: (B, num_heads, T_k, head_dim)
        
        # 使用 einsum 进行带因果 mask 的计算
        # 为每个查询位置 t 计算累积的 K^T V
        output = torch.zeros(B, num_heads, T_q, head_dim, device=q.device)
        z = torch.zeros(B, num_heads, T_q, 1, device=q.device)
        
        for t in range(T_q):
            # 只考虑 t 之前（包括 t）的键值对
            t_end = min(t + 1, T_k)
            
            # 计算当前位置的累积 K^T V
            # k_prime_t: (B, num_heads, t+1, head_dim)
            # v_t: (B, num_heads, t+1, head_dim)
            k_prime_t = k_prime[:, :, :t_end, :]
            v_t = v[:, :, :t_end, :]
            
            # kv_t: (B, num_heads, head_dim, head_dim)
            kv_t = torch.matmul(k_prime_t.transpose(-2, -1), v_t)
            
            # 计算归一化项
            k_sum_t = k_prime_t.sum(dim=2, keepdim=True).transpose(-2, -1)  # (B, num_heads, head_dim, 1)
            
            # 当前查询
            q_t = q_prime[:, :, t:t+1, :]  # (B, num_heads, 1, head_dim)
            
            # 计算输出
            output_t = torch.matmul(q_t, kv_t)  # (B, num_heads, 1, head_dim)
            z_t = torch.matmul(q_t, k_sum_t) + self.eps  # (B, num_heads, 1, 1)
            
            output[:, :, t:t+1, :] = output_t / z_t
            z[:, :, t:t+1, :] = z_t
        
        # 应用 dropout
        output = self.dropout(output)
        
        return output, None


class PerformerAttention(nn.Module):
    """
    Performer 注意力机制
    使用随机特征近似（FAVOR+）来实现线性注意力
    基于 "Rethinking Attention with Performers" 论文
    """
    def __init__(self, head_dim: int, num_random_features: int = 256, 
                 eps: float = 1e-6, dropout: float = 0.0):
        """
        Args:
            head_dim: 每个注意力头的维度
            num_random_features: 随机特征的数量（通常设为 head_dim 或更大）
            eps: 数值稳定性的小常数
            dropout: dropout 概率
        """
        super().__init__()
        self.head_dim = head_dim
        self.num_random_features = num_random_features
        self.eps = eps
        self.dropout = nn.Dropout(dropout)
        
        # 随机投影矩阵（固定，不训练）
        # 使用正交随机特征
        self.register_buffer(
            'random_features',
            torch.randn(head_dim, num_random_features) / math.sqrt(num_random_features)
        )
        
    def orthogonal_random_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        使用正交随机特征进行映射
        这种方法比普通随机特征有更好的近似性能
        
        Args:
            x: (B, num_heads, T, head_dim)
            
        Returns:
            features: (B, num_heads, T, num_random_features)
        """
        # 计算 x @ random_features
        # x: (B, num_heads, T, head_dim)
        # random_features: (head_dim, num_random_features)
        # projection: (B, num_heads, T, num_random_features)
        projection = torch.matmul(x, self.random_features)
        
        # 计算 ||x||^2 / 2
        x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True) / 2.0  # (B, num_heads, T, 1)
        
        # 应用 exp(projection - ||x||^2 / 2)
        features = torch.exp(projection - x_norm_sq)
        
        # 归一化
        features = features / math.sqrt(self.num_random_features)
        
        return features
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> tuple:
        """
        Performer 注意力的前向传播
        
        Args:
            q: (B, num_heads, T_q, head_dim)
            k: (B, num_heads, T_k, head_dim)
            v: (B, num_heads, T_k, head_dim)
            mask: (B, 1, 1, T_k) 或 None
            
        Returns:
            output: (B, num_heads, T_q, head_dim)
            attn: None
        """
        B, num_heads, T_q, head_dim = q.size()
        T_k = k.size(2)
        
        # 应用随机特征映射
        q_prime = self.orthogonal_random_features(q)  # (B, num_heads, T_q, num_random_features)
        k_prime = self.orthogonal_random_features(k)  # (B, num_heads, T_k, num_random_features)
        
        # 如果有 mask，应用它
        if mask is not None:
            mask = mask.squeeze(2).unsqueeze(-1)  # (B, 1, T_k, 1)
            k_prime = k_prime * mask
            v = v * mask
        
        # 计算 K^T V
        # k_prime^T: (B, num_heads, num_random_features, T_k)
        # v: (B, num_heads, T_k, head_dim)
        # kv: (B, num_heads, num_random_features, head_dim)
        kv = torch.matmul(k_prime.transpose(-2, -1), v)
        
        # 计算归一化项
        k_sum = k_prime.sum(dim=2, keepdim=True).transpose(-2, -1)  # (B, num_heads, num_random_features, 1)
        
        # 计算输出
        # q_prime: (B, num_heads, T_q, num_random_features)
        # kv: (B, num_heads, num_random_features, head_dim)
        output = torch.matmul(q_prime, kv)  # (B, num_heads, T_q, head_dim)
        
        # 归一化
        z = torch.matmul(q_prime, k_sum) + self.eps  # (B, num_heads, T_q, 1)
        output = output / z
        
        # 应用 dropout
        output = self.dropout(output)
        
        return output, None

