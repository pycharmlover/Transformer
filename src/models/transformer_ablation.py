"""
Encoder-Decoder Transformer 模型（消融实验版本：移除位置编码）

包含:
- token embedding（可选择共享 embedding）
- encoder / decoder（移除相对位置偏置）
- 输出投影到 vocab size（用于交叉熵损失）

注意: 
1. 消融实验：完全移除位置编码
2. 仅使用 token embedding，不使用任何位置信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.encoder_ablation import TransformerEncoder
from src.models.decoder_ablation import TransformerDecoder
import math
from typing import Optional

class Seq2SeqOutput:
    """包装模型输出"""
    def __init__(self, loss=None, logits=None):
        self.loss = loss
        self.logits = logits

class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, num_layers: int = 6, 
                 num_heads: int = 8, d_ff: int = 2048, max_len: int = 512, 
                 dropout: float = 0.1, share_embeddings: bool = False,
                 attention_type: str = "standard"):
        """
        消融实验版本的Transformer（移除位置编码）
        
        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度
            num_layers: Encoder/Decoder 层数
            num_heads: 注意力头数
            d_ff: 前馈层维度
            max_len: 最大序列长度（保留参数用于兼容性，但不使用）
            dropout: dropout 概率
            share_embeddings: 是否共享 encoder 和 decoder 的 embedding
            attention_type: 注意力机制类型
                - "standard": 标准注意力 O(n^2)（消融版本：不含位置编码）
                - "local_sparse": 局部稀疏注意力
                - "strided_sparse": 跨步稀疏注意力
                - "block_sparse": 块稀疏注意力
                - "linear": 线性注意力 O(n)
                - "causal_linear": 因果线性注意力 O(n)（用于decoder）
                - "performer": Performer 注意力（FAVOR+）
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.share_embeddings = share_embeddings
        self.attention_type = attention_type

        # token embeddings（消融实验：仅使用token embedding，不添加任何位置信息）
        self.src_tok_emb = nn.Embedding(vocab_size, d_model)
        if share_embeddings:
            self.tgt_tok_emb = self.src_tok_emb
        else:
            self.tgt_tok_emb = nn.Embedding(vocab_size, d_model)

        # encoder / decoder（消融版本：内部不含相对位置偏置）
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout, 
                                         attention_type=attention_type)
        self.decoder = TransformerDecoder(num_layers, d_model, num_heads, d_ff, dropout,
                                         attention_type=attention_type)

        # 输出投影（将 decoder 最终 hidden states 映射到 vocab logits）
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # 参数初始化
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src_input_ids: torch.Tensor, pad_token_id: int):
        """
        src_input_ids: (B, T_src)
        返回 shape (B, 1, 1, T_src) 的 mask（1 表示可见，0 表示 PAD）
        """
        src_mask = (src_input_ids != pad_token_id).int() # (B, T_src)
        return src_mask.unsqueeze(1).unsqueeze(1) # (B,1,1,T_src)

    def make_tgt_mask(self, tgt_input_ids: torch.Tensor, pad_token_id: int):
        """
        tgt_input_ids: (B, T_tgt)
        返回用于 decoder self-attn 的 mask，结合了 padding 与 causal：
        shape 最终为 (B, 1, T_tgt, T_tgt)
        """
        B, T = tgt_input_ids.size()
        pad_mask = (tgt_input_ids != pad_token_id).int() # (B, T_tgt)
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, T_tgt)
        # causal mask (1 for allowed positions)
        causal_mask = torch.tril(torch.ones((T, T), dtype=torch.uint8, device=tgt_input_ids.device)).unsqueeze(0).unsqueeze(0) # (1,1,T_tgt,T_tgt)
        combined = pad_mask & causal_mask 
        return combined # (B,1,T,T)
    
    def forward(self,
                src_input_ids: torch.Tensor,
                tgt_input_ids: torch.Tensor,
                src_attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                src_pad_id: Optional[int] = None,
                tgt_pad_id: Optional[int] = None,
                label_smoothing: float = 0.0):
        """
        src_input_ids: (B, T_src)
        tgt_input_ids: (B, T_tgt) - decoder 输入
        src_attention_mask: (B, T_src) - 1 表示有效 token，0 表示 padding
        labels: (B, T_tgt) - 用于计算 loss，-100 表示忽略位置
        src_pad_id: padding token id (如果未提供 src_attention_mask 则使用)
        tgt_pad_id: padding token id (如果未提供则使用 0)
        
        返回: Seq2SeqOutput 对象，包含 loss 和 logits
        """
        # Embedding + scale
        # 消融实验：仅使用token embedding，完全不添加位置编码
        src_emb = self.src_tok_emb(src_input_ids) * math.sqrt(self.d_model) # (B, T_src, d_model)
        tgt_emb = self.tgt_tok_emb(tgt_input_ids) * math.sqrt(self.d_model) # (B, T_tgt, d_model)

        # 构造 masks
        # src_mask: 使用 src_attention_mask (如果提供) 或从 src_pad_id 生成
        if src_attention_mask is not None:
            # src_attention_mask: (B, T_src) -> (B, 1, 1, T_src)
            src_mask = src_attention_mask.unsqueeze(1).unsqueeze(1)
        elif src_pad_id is not None:
            src_mask = self.make_src_mask(src_input_ids, src_pad_id)
        else:
            # 如果都没有提供，假设所有位置都有效
            src_mask = torch.ones(src_input_ids.size(0), 1, 1, src_input_ids.size(1), 
                                 device=src_input_ids.device, dtype=torch.long)
        
        # tgt_mask: 结合 padding 和 causal
        if tgt_pad_id is None:
            tgt_pad_id = 0  # 默认 pad id
        tgt_mask = self.make_tgt_mask(tgt_input_ids, tgt_pad_id)
        memory_mask = src_mask

        # Encoder -> memory
        memory = self.encoder(src_emb, src_mask) # (B, T_src, d_model)

        # Decoder
        out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=memory_mask) # (B, T_tgt, d_model)

        # 输出投影
        logits = self.output_proj(out)  # (B, T_tgt, vocab_size)
        
        # 计算 loss (如果提供了 labels)
        loss = None
        if labels is not None:
            # labels: (B, T_tgt), logits: (B, T_tgt, vocab_size)
            # CrossEntropyLoss 期望 (N, C) 和 (N,)，所以需要 reshape
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),  # (B*T_tgt, vocab_size)
                labels.reshape(-1),  # (B*T_tgt,)
                ignore_index=-100,  # 忽略 padding 位置
                label_smoothing=label_smoothing  # 标签平滑，有助于防止过拟合
            )
        
        return Seq2SeqOutput(loss=loss, logits=logits)

