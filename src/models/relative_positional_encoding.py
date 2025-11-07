"""
相对位置编码（T5）
"""

import torch
import torch.nn as nn
import math


class RelativePositionBias(nn.Module):
    """
    1. 将相对位置距离映射到bucket
    2. 使用可学习的embedding表示每个bucket的偏置
    3. 在注意力计算时将偏置加到注意力分数上
    """
    
    def __init__(
        self, 
        num_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
        bidirectional: bool = True
    ):
        """
        Args:
            num_heads: 注意力头的数量
            num_buckets: 相对位置bucket的数量
            max_distance: 最大相对位置距离
            bidirectional: 是否双向（encoder使用True，decoder的self-attention使用False）
        """
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        
        # 相对位置偏置表：为每个bucket和每个head存储一个可学习的偏置值
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
        
    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128
    ) -> torch.Tensor:
        """
        将相对位置映射到bucket索引
        
        T5的映射策略：
        1. 对于较小的相对距离（< max_exact），每个距离对应一个bucket
        2. 对于较大的相对距离，使用对数缩放映射到剩余的bucket
        
        Args:
            relative_position: 相对位置矩阵 (query_length, key_length)
            bidirectional: 是否考虑正负方向
            num_buckets: bucket总数
            max_distance: 最大距离
            
        Returns:
            bucket索引 (query_length, key_length)
        """
        relative_buckets = 0
        
        if bidirectional:
            # 双向情况：一半bucket用于负方向，一半用于正方向
            num_buckets //= 2
            # 正向位置加上 num_buckets 的偏移
            relative_buckets += (relative_position > 0).long() * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            # 单向情况（因果）：将正向位置设为0，只使用负向位置
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        
        # 现在 relative_position 都是非负的
        # 将 [0, max_distance) 的范围映射到 [0, num_buckets) 的bucket
        
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        
        # 对于较大的距离，使用对数缩放
        # 公式: bucket = max_exact + log(distance/max_exact) / log(max_distance/max_exact) * (num_buckets - max_exact)
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).long()
        
        # 确保不超过bucket范围
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        
        # 根据距离大小选择bucket
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        
        return relative_buckets
    
    def forward(self, query_length: int, key_length: int, device=None) -> torch.Tensor:
        """
        计算相对位置偏置
        
        Args:
            query_length: 查询序列长度
            key_length: 键序列长度
            
        Returns:
            相对位置偏置 (1, num_heads, query_length, key_length)
        """
        if device is None:
            device = self.relative_attention_bias.weight.device
        
        # 创建位置索引
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        
        # 计算相对位置: query_pos - key_pos
        # shape: (query_length, key_length)
        relative_position = memory_position - context_position
        
        # 将相对位置映射到bucket
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance
        )
        
        # 查找每个bucket对应的偏置值
        # relative_position_bucket: (query_length, key_length)
        # bias: (query_length, key_length, num_heads)
        bias = self.relative_attention_bias(relative_position_bucket)
        
        # 转置为 (1, num_heads, query_length, key_length)
        bias = bias.permute(2, 0, 1).unsqueeze(0)
        
        return bias


class RelativePositionBiasForDecoder(nn.Module):
    """
    用于Decoder的相对位置偏置
    """
    
    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128
    ):
        """
        Args:
            num_heads: 注意力头的数量
            num_buckets: 相对位置bucket的数量
            max_distance: 最大相对位置距离
        """
        super().__init__()
        
        # Self-attention的相对位置偏置（单向）
        self.self_attn_bias = RelativePositionBias(
            num_heads=num_heads,
            num_buckets=num_buckets,
            max_distance=max_distance,
            bidirectional=False
        )
        
        # Cross-attention的相对位置偏置（双向）
        self.cross_attn_bias = RelativePositionBias(
            num_heads=num_heads,
            num_buckets=num_buckets,
            max_distance=max_distance,
            bidirectional=True
        )
    
    def get_self_attn_bias(self, query_length: int, key_length: int, device=None) -> torch.Tensor:
        """获取self-attention的相对位置偏置"""
        return self.self_attn_bias(query_length, key_length, device)
    
    def get_cross_attn_bias(self, query_length: int, key_length: int, device=None) -> torch.Tensor:
        """获取cross-attention的相对位置偏置"""
        return self.cross_attn_bias(query_length, key_length, device)
