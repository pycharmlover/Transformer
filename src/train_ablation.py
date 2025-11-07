"""
训练脚本：把数据载入、模型构建、训练循环、验证与 checkpoint 保存/加载
"""

import os
import math
import time
import random
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.data_utils import TextPairDataset
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup,AutoModelForMaskedLM

from src.models.transformer_ablation import TransformerSeq2Seq

import math
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from src.utils.plot_utils import plot_training_curves
from src.data_process import data_process

# -------------------------
# 工具函数
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(state: dict, save_dir: str, prefix: str = "ckpt"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{prefix}.pt")
    torch.save(state, path)
    print(f"Saved checkpoint to {path}")

def load_checkpoint(model, optimizer, scheduler, ckpt_path: str, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optim_state_dict"])
    if "scheduler_state_dict" in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint.get("epoch", 0)
    print(f"Loaded checkpoint {ckpt_path} (start_epoch={start_epoch})")
    return start_epoch

# -------------------------
# collate_fn：把原始 dataset 中的 tokenized outputs batch 化
# 这里采用 tokenizer.batch_encode_plus 动态 padding，避免固定 max_length 导致空间浪费
# -------------------------
def collate_fn(batch, tokenizer, max_src_len=128, max_tgt_len=128):
    """
    batch: list of dicts returned by TranslationDataset (源/目标原文本，或已 tokenized)
    返回：
      src_input_ids, src_attention_mask, tgt_input_ids (decoder inputs), labels
    labels 中 pad 部分用 -100（CrossEntropy 的 ignore_index）
    """
    src_texts = [ex["translation"][tokenizer.model_input_names[0].split("_")[0] if False else 'src'] 
                 if "translation" not in ex else ex["translation"].get("src", None) for ex in batch]
    # 假设 batch 中的 item 是 dict: {"src_text":..., "tgt_text":...}

    if isinstance(batch[0], dict) and "src_text" in batch[0] and "tgt_text" in batch[0]:
        src_texts = [ex["src_text"] for ex in batch]
        tgt_texts = [ex["tgt_text"] for ex in batch]
    else:
        src_texts = []
        tgt_texts = []
        for ex in batch:
            if "translation" in ex:
                trans = ex["translation"]
                keys = list(trans.keys())
                if len(keys) >= 2:
                    src_texts.append(trans[keys[0]])
                    tgt_texts.append(trans[keys[1]])
                else:
                    raise ValueError("translation dict has <2 languages")
            else:
                raise ValueError("Batch format not supported by collate_fn. Expected keys 'src_text'/'tgt_text' or 'translation' dict.")

    # tokenizer.batch_encode_plus 支持动态 padding
    src_enc = tokenizer(src_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_src_len)
    tgt_enc = tokenizer(tgt_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_tgt_len)

    src_input_ids = src_enc["input_ids"]
    src_attention_mask = src_enc["attention_mask"]
    tgt_input_ids = tgt_enc["input_ids"]  # 这包含 [CLS] ... [SEP] [PAD]

    # 构造 decoder 输入：将 labels 右移一位并在开头填 BOS (使用 tokenizer.cls_token_id)
    # labels: 真实目标 tokens（用于计算 loss）
    labels = tgt_input_ids.clone()
    pad_token_id = tokenizer.pad_token_id
    cls_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id

    # prepare decoder_input_ids by shifting right
    decoder_input_ids = torch.full(labels.size(), pad_token_id, dtype=torch.long)
    decoder_input_ids[:, 0] = cls_token_id
    decoder_input_ids[:, 1:] = labels[:, :-1].clone()

    # 将 labels 中 pad token 替换为 -100，以便 CrossEntropy 忽略
    labels_masked = labels.masked_fill(labels == pad_token_id, -100)

    batch_out = {
        "src_input_ids": src_input_ids,
        "src_attention_mask": src_attention_mask,
        "tgt_input_ids": decoder_input_ids,
        "labels": labels_masked,
        "raw_labels": labels,  # 方便调试
        "pad_token_id": pad_token_id
    }
    return batch_out

# -------------------------
# 训练与验证函数
# -------------------------
def evaluate(model, dataloader, tokenizer, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            src = batch["src_input_ids"].to(device)
            tgt_in = batch["tgt_input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(src, tgt_in, src_pad_id=batch["pad_token_id"], tgt_pad_id=batch["pad_token_id"])
            B, T, V = logits.size()
            loss = F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100, reduction="sum")
            n_tokens = (labels != -100).sum().item()

            total_loss += loss.item()
            total_tokens += n_tokens
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl

def train(args):
    # 1. 随机种子
    set_seed(args.seed)

    # 2. 设备
    device = torch.device("cuda:1" if torch.cuda.is_available() and args.cuda else "cpu")
    print("Device:", device)

    # 3. 数据处理（如果数据文件不存在则自动下载和处理）
    # 使用通用文件名（包含所有语言对的互译）
    src_path = "data/train.src"  # 多语言源文本
    tgt_path = "data/train.tgt"  # 多语言目标文本
    
    if not (Path(src_path).exists() and Path(tgt_path).exists()):
        print(f"\n📥 数据文件不存在，开始处理数据集...")
        print(f"   🌍 将加载TED Talks多语言数据集（所有语言对互译，所有年份）")
        print(f"   📊 这将包含109种语言之间的所有可能翻译组合")
        print(f"   💾 预计数据量：500K-1M条")
        print(f"   ⏳ 首次加载需要较长时间（约30-60分钟），请耐心等待...\n")
        data_process(args)
        print(f"\n✅ 数据处理完成！\n")
    else:
        print(f"✅ 发现已有数据文件: {src_path}, {tgt_path}")
        print(f"   注意：这是多语言多向数据（109种语言互译）\n")

    # 4. tokenizer（从本地路径加载）
    # 4. 加载tokenizer并添加语言标记
    print("\n" + "="*60)
    print("4. 加载并配置Tokenizer")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    original_vocab_size = len(tokenizer)
    print(f"原始vocab size: {original_vocab_size}")
    
    # 添加语言标记作为特殊token（这样会被tokenize成单个token）
    special_tokens = ['<2en>', '<2de>']  # 英语和德语
    num_added = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    
    new_vocab_size = len(tokenizer)
    print(f"添加 {num_added} 个语言标记: {special_tokens}")
    print(f"新vocab size: {new_vocab_size}")

    # 保存更新后的tokenizer到临时目录，确保Dataset使用相同的tokenizer
    tokenizer_with_tags_path = "data/tokenizer_with_tags"
    os.makedirs(tokenizer_with_tags_path, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_with_tags_path)
    print(f"✅ 已保存更新后的tokenizer到: {tokenizer_with_tags_path}")
    
    # 验证特殊token是否正确添加
    print("\n验证语言标记:")
    for tag in special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(tag)
        decoded = tokenizer.decode([token_id])
        print(f"  {tag} → token_id={token_id} → decoded='{decoded}'")
    
    vocab_size = new_vocab_size

    # 5. 从本地 train.src / train.tgt 加载数据
    print("\n" + "="*60)
    print("5. 加载训练数据")
    print("="*60)

    dataset = TextPairDataset(
        src_path=src_path,
        tgt_path=tgt_path,
        tokenizer_path=tokenizer_with_tags_path,  # 使用包含语言标记的tokenizer
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len
    )

    # 随机划分训练/验证
    total_size = len(dataset)
    val_split = int(total_size * args.val_ratio)
    indices = list(range(total_size))
    random.shuffle(indices)
    val_indices = indices[:val_split]
    train_indices = indices[val_split:]

    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"Train examples: {len(train_dataset)}, Val examples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn
    )

    # 5. 模型（使用T5风格的相对位置编码）
    model = TransformerSeq2Seq(
        vocab_size=vocab_size, 
        d_model=args.d_model, 
        num_layers=args.num_layers,
        num_heads=args.num_heads, 
        d_ff=args.d_ff, 
        max_len=args.max_src_len, 
        dropout=args.dropout,
        share_embeddings=args.share_embeddings,
        attention_type=args.attention_type
    )
    model = model.to(device)
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} | Trainable: {trainable_params:,}")
    print(f"Position Encoding: T5-style Relative Position Bias")
    print(f"Attention Type: {args.attention_type}")


    # 6. 优化器 + scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    print("total_steps: ", total_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # 7. 模型训练
    global_step = 0
    train_losses, val_losses = [], []
    train_ppls, val_ppls = [], []
    val_accuracies = []

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0  # Early stopping 计数器

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for step, batch in enumerate(train_loader):
            src_input_ids = batch["src_input_ids"].to(device)
            src_attention_mask = batch["src_attention_mask"].to(device)
            tgt_input_ids = batch["tgt_input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                src_input_ids=src_input_ids,
                src_attention_mask=src_attention_mask,
                tgt_input_ids=tgt_input_ids,
                labels=labels,
                label_smoothing=args.label_smoothing
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            if step % 100 == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"[Epoch {epoch+1}] Step {step:04d} | Loss: {loss.item():.4f} | LR: {lr:.6f}")

        # ====== Training统计 ======
        # 除以batch数量，因为每个loss已经是batch内的平均
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_ppl = math.exp(avg_train_loss)
        train_ppls.append(train_ppl)

        # ====== Validation ======
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total_tokens = 0

        with torch.no_grad():
            for batch in val_loader:
                src_input_ids = batch["src_input_ids"].to(device)
                src_attention_mask = batch["src_attention_mask"].to(device)
                tgt_input_ids = batch["tgt_input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    src_input_ids=src_input_ids,
                    src_attention_mask=src_attention_mask,
                    tgt_input_ids=tgt_input_ids,
                    labels=labels,
                    label_smoothing=0.0  # 验证时不使用label smoothing
                )
                val_loss += outputs.loss.item()

                # ====== token-level accuracy ======
                predictions = outputs.logits.argmax(dim=-1)  # (B, T_tgt)
                mask = (labels != -100)
                val_correct += ((predictions == labels) & mask).sum().item()
                val_total_tokens += mask.sum().item()

        # 除以batch数量，因为每个loss已经是batch内的平均
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_ppl = math.exp(avg_val_loss)
        val_ppls.append(val_ppl)
        val_accuracy = val_correct / val_total_tokens if val_total_tokens > 0 else 0.0
        val_accuracies.append(val_accuracy)

        print(f"\n📘 Epoch {epoch+1} Summary:")
        print(f"  Train Loss={avg_train_loss:.4f}, Train PPL={train_ppl:.2f}")
        print(f"  Val   Loss={avg_val_loss:.4f}, Val PPL={val_ppl:.2f}, Val Acc={val_accuracy:.4f}\n")

        # ====== 定期保存检查点（每20个epoch或最后一个epoch）======
        if (epoch + 1) % 20 == 0 or (epoch + 1) == args.epochs:
            ckpt_dir = os.path.join(args.output_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch{epoch+1}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss
            }, ckpt_path)
            print(f"✅ 定期检查点已保存至 {ckpt_path}")

        # ====== 最佳模型追踪 & Early Stopping ======
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0  # 重置early stopping计数器
            best_path = os.path.join(args.output_dir, "best_model.pt")
            
            # 保存checkpoint（使用包含语言标记的tokenizer路径）
            checkpoint_args = vars(args).copy()
            checkpoint_args['tokenizer_path'] = tokenizer_with_tags_path  # 使用更新后的tokenizer路径
            
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss": avg_val_loss,
                "tokenizer": tokenizer_with_tags_path,  # 保存包含语言标记的tokenizer路径
                "args": checkpoint_args,
            }, best_path)
            print(f"🏆 最佳模型更新并保存至 {best_path}")
        else:
            patience_counter += 1
            print(f"⚠️  验证loss未改善 ({patience_counter}/{args.patience})")
            
            if patience_counter >= args.patience:
                print(f"\n🛑 Early Stopping 触发！")
                print(f"   最佳Epoch: {best_epoch}, Val Loss={best_val_loss:.4f}, Val PPL={math.exp(best_val_loss):.2f}")
                print(f"   当前Epoch: {epoch+1}, Val Loss={avg_val_loss:.4f}, Val PPL={val_ppl:.2f}")
                break

    print(f"\n✅ 训练完成！")
    print(f"   最佳模型: Epoch {best_epoch}, Val Loss={best_val_loss:.4f}, Val PPL={math.exp(best_val_loss):.2f}")
    print(f"   所有结果与日志已保存在 results/")


    plot_training_curves(train_losses, val_losses, save_path=os.path.join(args.output_dir, "training_curve.png"))
    
    # 保存所有指标到文件
    metrics_file = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write("Epoch\tTrain_Loss\tTrain_PPL\tVal_Loss\tVal_PPL\tVal_Accuracy\n")
        for i in range(len(train_losses)):
            f.write(f"{i+1}\t{train_losses[i]:.4f}\t{train_ppls[i]:.4f}\t"
                   f"{val_losses[i]:.4f}\t{val_ppls[i]:.4f}\t{val_accuracies[i]:.4f}\n")
    print(f"Metrics saved to {metrics_file}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_lang", type=str, default="nl", help="source language code")
    parser.add_argument("--tgt_lang", type=str, default="en", help="target language code")
    parser.add_argument("--tokenizer_path", type=str, default="/home/extra_home/lc/google-bert/rembert", help="local tokenizer path")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.05, help="用于抽取小验证集比例")
    parser.add_argument("--max_src_len", type=int, default=128)
    parser.add_argument("--max_tgt_len", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--share_embeddings", action="store_true")

    # 注意力机制类型参数
    parser.add_argument("--attention_type", type=str, default="standard",
                       choices=["standard", "local_sparse", "strided_sparse", "block_sparse", "linear", "causal_linear", "performer"],
                       help="注意力机制类型: standard（标准O(n^2)，使用T5风格相对位置编码）, local_sparse（局部稀疏）, strided_sparse（跨步稀疏）, block_sparse（块稀疏）, linear（线性O(n)）, causal_linear（因果线性，用于decoder）, performer（Performer）")
    
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_steps", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset_size", type=int, default=0, help="数据子集大小（0=使用全部数据，>0=使用指定数量）")
    parser.add_argument("--cuda", action="store_true", help="if set, use cuda when available")
    
    # 正则化与早停参数
    parser.add_argument("--patience", type=int, default=10, help="Early stopping: 验证loss不改善的容忍epoch数")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing factor (0.0-0.2)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # 自动处理数据（如果数据不存在）
    # data_process(args)
    # 开始训练
    train(args)

