#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Translator: 使用训练好的Transformer模型进行翻译
"""

import torch
import argparse
from transformers import AutoTokenizer
from src.models.transformer import TransformerSeq2Seq
import os


class Translator:
    """多语言翻译器"""
    
    # 支持的语言
    SUPPORTED_LANGUAGES = {
        'en': 'English',
        'de': 'German'
    }
    
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化翻译器
        
        Args:
            model_path: 模型权重文件路径
            device: 运行设备 ('cuda', 'cpu', 或 'auto')
        """
        # 处理 'auto' 设备选择
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        print(f"🔧 加载模型中...")
        print(f"   设备: {device}")
        print(f"   模型路径: {model_path}")
        
        # 加载模型checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # 获取模型参数
        self.args = argparse.Namespace(**checkpoint['args'])
        
        # 加载tokenizer（尝试加载训练时保存的包含语言标记的版本）
        tokenizer_with_tags_path = "data/tokenizer_with_tags"
        if os.path.exists(tokenizer_with_tags_path):
            print(f"   ✅ 使用训练时保存的tokenizer（含语言标记）")
            print(f"      路径: {tokenizer_with_tags_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_with_tags_path, local_files_only=True)
        else:
            print(f"   ⚠️  未找到 {tokenizer_with_tags_path}")
            print(f"   使用原始tokenizer并手动添加语言标记: {self.args.tokenizer_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_path)
            # 添加语言标记（必须与训练时一致）
            special_tokens = ['<2en>', '<2de>']  # 英语和德语
            num_added = self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
            if num_added > 0:
                print(f"   Added {num_added} language tags: {special_tokens}")
        
        # 验证特殊token是否正确
        print("\n   🔍 验证语言标记:")
        for tag in ['<2en>', '<2de>']:  # 英语和德语
            token_id = self.tokenizer.convert_tokens_to_ids(tag)
            decoded = self.tokenizer.decode([token_id])
            print(f"      {tag} → ID={token_id} → decode='{decoded}'")
        
        # 获取正确的vocab_size（使用len()而不是.vocab_size属性）
        # tokenizer.vocab_size可能返回原始大小，len(tokenizer)返回实际大小
        actual_vocab_size = len(self.tokenizer)
        print(f"\n   📊 Tokenizer信息:")
        print(f"      tokenizer.vocab_size = {self.tokenizer.vocab_size}")
        print(f"      len(tokenizer) = {actual_vocab_size}")
        print(f"      使用vocab_size = {actual_vocab_size}")
        
        # 初始化模型（使用len(tokenizer)而不是tokenizer.vocab_size）
        self.model = TransformerSeq2Seq(
            vocab_size=actual_vocab_size,
            d_model=self.args.d_model,
            num_layers=self.args.num_layers,
            num_heads=self.args.num_heads,
            d_ff=self.args.d_ff,
            max_len=max(self.args.max_src_len, self.args.max_tgt_len),
            dropout=self.args.dropout,
            share_embeddings=False,
            attention_type=self.args.attention_type
        ).to(device)
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        
        print(f"✅ 模型加载完成！")
        print(f"   参数量: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")
        print(f"   注意力机制: {self.args.attention_type}")
        print(f"   支持的语言: {', '.join([f'{k} ({v})' for k, v in self.SUPPORTED_LANGUAGES.items()])}")
        print()
    
    def translate(self, text, src_lang='en', tgt_lang='zh', beam_size=5, max_length=128):
        """
        翻译文本
        
        Args:
            text: 输入文本
            src_lang: 源语言代码 ('en', 'zh', 'ja')
            tgt_lang: 目标语言代码 ('en', 'zh', 'ja')
            beam_size: Beam search宽度
            max_length: 最大生成长度
            
        Returns:
            dict: 包含以下字段的字典
                - 'input': 输入文本
                - 'translation': 翻译后的文本
                - 'src_lang': 源语言
                - 'tgt_lang': 目标语言
        """
        if src_lang not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"不支持的源语言: {src_lang}. 支持的语言: {list(self.SUPPORTED_LANGUAGES.keys())}")
        if tgt_lang not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"不支持的目标语言: {tgt_lang}. 支持的语言: {list(self.SUPPORTED_LANGUAGES.keys())}")
        if src_lang == tgt_lang:
            return {
                'input': text,
                'translation': text,
                'src_lang': src_lang,
                'tgt_lang': tgt_lang
            }
        
        print(f"🌍 翻译: {self.SUPPORTED_LANGUAGES[src_lang]} → {self.SUPPORTED_LANGUAGES[tgt_lang]}")
        print(f"📝 输入: {text}")
        
        # 在输入前添加目标语言标记
        text_with_tag = f"<2{tgt_lang}> {text}"
        print(f"   (添加语言标记: <2{tgt_lang}>)")
        
        # Tokenize输入
        src_tokens = self.tokenizer(
            text_with_tag,
            max_length=self.args.max_src_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        src_input_ids = src_tokens['input_ids'].to(self.device)
        src_attention_mask = src_tokens['attention_mask'].to(self.device)
        
        # 使用Beam Search生成翻译
        with torch.no_grad():
            if beam_size > 1:
                translation_ids = self._beam_search(
                    src_input_ids, 
                    src_attention_mask, 
                    beam_size=beam_size, 
                    max_length=max_length
                )
            else:
                translation_ids = self._greedy_decode(
                    src_input_ids, 
                    src_attention_mask, 
                    max_length=max_length
                )
        
        # 解码输出
        translation = self.tokenizer.decode(
            translation_ids[0], 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        
        print(f"✅ 输出: {translation}")
        print()
        
        # 返回字典格式，包含输入和输出
        return {
            'input': text,
            'translation': translation,
            'src_lang': src_lang,
            'tgt_lang': tgt_lang
        }
    
    def _greedy_decode(self, src_input_ids, src_attention_mask, max_length=128):
        """
        Greedy解码（逐token生成，选择概率最高的）
        
        Args:
            src_input_ids: 源语言输入 [batch_size, src_len]
            src_attention_mask: 源语言attention mask
            max_length: 最大生成长度
            
        Returns:
            生成的token ids [batch_size, tgt_len]
        """
        batch_size = src_input_ids.size(0)
        
        # 初始化：只有[CLS]
        tgt_input_ids = torch.full(
            (batch_size, 1), 
            self.tokenizer.cls_token_id, 
            dtype=torch.long, 
            device=self.device
        )
        
        for _ in range(max_length - 1):
            # 前向传播
            outputs = self.model(
                src_input_ids=src_input_ids,
                tgt_input_ids=tgt_input_ids,
                src_attention_mask=src_attention_mask,
                src_pad_id=self.tokenizer.pad_token_id,
                tgt_pad_id=self.tokenizer.pad_token_id
            )
            
            # 获取最后一个位置的logits
            next_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            
            # 选择概率最高的token
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [batch_size, 1]
            
            # 拼接到已生成的序列
            tgt_input_ids = torch.cat([tgt_input_ids, next_token], dim=1)
            
            # 如果生成了[SEP]，停止
            if (next_token == self.tokenizer.sep_token_id).all():
                break
        
        return tgt_input_ids
    
    def _beam_search(self, src_input_ids, src_attention_mask, beam_size=5, max_length=128):
        """
        Beam Search解码
        
        Args:
            src_input_ids: 源语言输入 [1, src_len]
            src_attention_mask: 源语言attention mask
            beam_size: beam宽度
            max_length: 最大生成长度
            
        Returns:
            最佳生成序列 [1, tgt_len]
        """
        batch_size = src_input_ids.size(0)
        assert batch_size == 1, "Beam search只支持batch_size=1"
        
        # 初始化beam
        # 每个beam: (sequence, score)
        beams = [(torch.full((1, 1), self.tokenizer.cls_token_id, dtype=torch.long, device=self.device), 0.0)]
        
        for step in range(max_length - 1):
            all_candidates = []
            
            for seq, score in beams:
                # 如果已经生成了[SEP]，不再扩展
                if seq[0, -1].item() == self.tokenizer.sep_token_id:
                    all_candidates.append((seq, score))
                    continue
                
                # 前向传播
                outputs = self.model(
                    src_input_ids=src_input_ids,
                    tgt_input_ids=seq,
                    src_attention_mask=src_attention_mask,
                    src_pad_id=self.tokenizer.pad_token_id,
                    tgt_pad_id=self.tokenizer.pad_token_id
                )
                
                # 获取下一个token的log概率
                next_token_logits = outputs.logits[0, -1, :]  # [vocab_size]
                log_probs = torch.log_softmax(next_token_logits, dim=-1)
                
                # 获取top-k个候选
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)
                
                # 扩展beam
                for log_prob, token_id in zip(topk_log_probs, topk_indices):
                    new_seq = torch.cat([seq, token_id.unsqueeze(0).unsqueeze(0)], dim=1)
                    new_score = score + log_prob.item()
                    all_candidates.append((new_seq, new_score))
            
            # 选择score最高的beam_size个候选
            beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]
            
            # 如果所有beam都结束了，停止
            if all(seq[0, -1].item() == self.tokenizer.sep_token_id for seq, _ in beams):
                break
        
        # 返回得分最高的序列
        best_seq, _ = beams[0]
        return best_seq
    
    def translate_batch(self, texts, src_lang='en', tgt_lang='zh', beam_size=5, max_length=128):
        """
        批量翻译
        
        Args:
            texts: 文本列表
            src_lang: 源语言代码
            tgt_lang: 目标语言代码
            beam_size: Beam search宽度（设为1使用greedy）
            max_length: 最大生成长度
            
        Returns:
            翻译结果列表
        """
        translations = []
        for text in texts:
            translation = self.translate(text, src_lang, tgt_lang, beam_size, max_length)
            translations.append(translation)
        return translations


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description='Transformer多语言翻译器')
    parser.add_argument('--model_path', type=str, default='results/best_model.pt',
                       help='模型权重文件路径')
    parser.add_argument('--text', type=str, required=True,
                       help='要翻译的文本')
    parser.add_argument('--src_lang', type=str, default='en',
                       choices=['en', 'zh', 'ja'],
                       help='源语言 (en=英语, zh=中文, ja=日语)')
    parser.add_argument('--tgt_lang', type=str, default='zh',
                       choices=['en', 'zh', 'ja'],
                       help='目标语言 (en=英语, zh=中文, ja=日语)')
    parser.add_argument('--beam_size', type=int, default=5,
                       help='Beam search宽度 (1=greedy decoding)')
    parser.add_argument('--max_length', type=int, default=128,
                       help='最大生成长度')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='运行设备')
    
    args = parser.parse_args()
    
    # 确定设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # 创建翻译器
    translator = Translator(args.model_path, device=device)
    
    # 执行翻译
    translation = translator.translate(
        text=args.text,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        beam_size=args.beam_size,
        max_length=args.max_length
    )
    
    # 输出结果
    print("="*60)
    print(f"源语言 ({args.src_lang}): {args.text}")
    print(f"目标语言 ({args.tgt_lang}): {translation}")
    print("="*60)


if __name__ == "__main__":
    main()

