#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IWSLT2017 数据处理脚本
处理英德(en-de)双向翻译数据集
"""

import os
import sys
import tarfile
import random
from pathlib import Path


def extract_iwslt_data(tar_path, extract_dir):
    """
    解压IWSLT2017数据集
    
    Args:
        tar_path: tar.gz文件路径
        extract_dir: 解压目标目录
    
    Returns:
        (en_file_path, de_file_path): 解压后的文件路径
    """
    print(f"📦 解压数据集: {tar_path}")
    
    # 确保目录存在
    os.makedirs(extract_dir, exist_ok=True)
    
    # 解压
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(extract_dir)
    
    # 查找解压后的文件
    en_file = os.path.join(extract_dir, 'train.en')
    de_file = os.path.join(extract_dir, 'train.de')
    
    if not os.path.exists(en_file) or not os.path.exists(de_file):
        raise FileNotFoundError(f"解压后未找到train.en或train.de文件")
    
    print(f"✅ 解压完成")
    print(f"   - {en_file}")
    print(f"   - {de_file}")
    
    return en_file, de_file


def load_parallel_data(en_file, de_file, min_len=5, max_len=150):
    """
    加载平行语料，进行基本过滤
    
    Args:
        en_file: 英文文件路径
        de_file: 德文文件路径
        min_len: 最小字符长度（过滤太短的句子）
        max_len: 最大字符长度（过滤太长的句子）
    
    Returns:
        List[tuple]: [(en_text, de_text), ...]
    """
    print(f"\n📖 加载平行语料...")
    
    pairs = []
    
    with open(en_file, 'r', encoding='utf-8') as f_en, \
         open(de_file, 'r', encoding='utf-8') as f_de:
        
        for line_num, (en_line, de_line) in enumerate(zip(f_en, f_de), 1):
            en_text = en_line.strip()
            de_text = de_line.strip()
            
            # 过滤空行
            if not en_text or not de_text:
                continue
            
            # 过滤长度不合适的句子
            if len(en_text) < min_len or len(de_text) < min_len:
                continue
            if len(en_text) > max_len * 10 or len(de_text) > max_len * 10:
                continue
            
            pairs.append((en_text, de_text))
            
            if line_num % 50000 == 0:
                print(f"  已处理 {line_num:,} 行...")
    
    print(f"✅ 加载完成: {len(pairs):,} 条有效句对")
    return pairs


def create_bidirectional_dataset(pairs, val_ratio=0.1, seed=42, max_pairs=None):
    """
    创建双向翻译数据集（en→de 和 de→en）
    
    Args:
        pairs: List[(en_text, de_text)]
        val_ratio: 验证集比例
        seed: 随机种子
        max_pairs: 最大句对数量（None表示使用全部）
    
    Returns:
        train_data: List[(src, tgt)]
        val_data: List[(src, tgt)]
    """
    print(f"\n🔀 创建双向翻译数据集...")
    
    random.seed(seed)
    
    # 打乱数据
    pairs_shuffled = pairs.copy()
    random.shuffle(pairs_shuffled)
    
    # 限制数据量
    if max_pairs is not None and max_pairs < len(pairs_shuffled):
        pairs_shuffled = pairs_shuffled[:max_pairs]
        print(f"  ⚠️  限制数据量: 使用 {max_pairs:,} / {len(pairs):,} 句对")
    
    # 划分train/val
    val_size = int(len(pairs_shuffled) * val_ratio)
    train_pairs = pairs_shuffled[val_size:]
    val_pairs = pairs_shuffled[:val_size]
    
    print(f"  原始句对: {len(pairs):,}")
    print(f"  Train句对: {len(train_pairs):,}")
    print(f"  Val句对: {len(val_pairs):,}")
    
    # 创建双向数据（每个句对产生两个样本：en→de 和 de→en）
    train_data = []
    val_data = []
    
    # 处理训练集
    for en_text, de_text in train_pairs:
        # en → de
        train_data.append((f"<2de> {en_text}", de_text))
        # de → en
        train_data.append((f"<2en> {de_text}", en_text))
    
    # 处理验证集
    for en_text, de_text in val_pairs:
        # en → de
        val_data.append((f"<2de> {en_text}", de_text))
        # de → en
        val_data.append((f"<2en> {de_text}", en_text))
    
    # 再次打乱（混合en→de和de→en）
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    print(f"\n📊 双向数据统计:")
    print(f"  Train: {len(train_data):,} 样本 (en→de: {len(train_pairs):,}, de→en: {len(train_pairs):,})")
    print(f"  Val:   {len(val_data):,} 样本 (en→de: {len(val_pairs):,}, de→en: {len(val_pairs):,})")
    
    return train_data, val_data


def save_to_files(train_data, val_data, output_dir):
    """
    保存处理后的数据到文件
    
    Args:
        train_data: List[(src, tgt)]
        val_data: List[(src, tgt)]
        output_dir: 输出目录
    """
    print(f"\n💾 保存数据到: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存训练集
    train_src_file = os.path.join(output_dir, 'train.src')
    train_tgt_file = os.path.join(output_dir, 'train.tgt')
    
    with open(train_src_file, 'w', encoding='utf-8') as f_src, \
         open(train_tgt_file, 'w', encoding='utf-8') as f_tgt:
        for src, tgt in train_data:
            f_src.write(src + '\n')
            f_tgt.write(tgt + '\n')
    
    print(f"✅ 训练集保存完成:")
    print(f"   - {train_src_file} ({len(train_data):,} 行)")
    print(f"   - {train_tgt_file} ({len(train_data):,} 行)")
    
    # 保存验证集
    val_src_file = os.path.join(output_dir, 'val.src')
    val_tgt_file = os.path.join(output_dir, 'val.tgt')
    
    with open(val_src_file, 'w', encoding='utf-8') as f_src, \
         open(val_tgt_file, 'w', encoding='utf-8') as f_tgt:
        for src, tgt in val_data:
            f_src.write(src + '\n')
            f_tgt.write(tgt + '\n')
    
    print(f"✅ 验证集保存完成:")
    print(f"   - {val_src_file} ({len(val_data):,} 行)")
    print(f"   - {val_tgt_file} ({len(val_data):,} 行)")


def data_process(args=None):
    """
    数据处理函数（兼容train.py调用）
    
    Args:
        args: 命令行参数（可选，这里不使用）
    """
    # 配置
    TAR_PATH = "/home/extra_home/lc/IWSLT2017-en-de-v2.tar.gz"
    EXTRACT_DIR = "/home/extra_home/lc/iwslt2017_extracted"
    OUTPUT_DIR = "data"
    VAL_RATIO = 0.1
    SEED = 42
    MAX_PAIRS = 100000  # 最大句对数（双向后 train ≈ 18K样本）
    # MAX_PAIRS = None
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    output_dir = project_root / OUTPUT_DIR
    
    # # 检查数据是否已存在
    # train_src = output_dir / "train.src"
    # train_tgt = output_dir / "train.tgt"
    # val_src = output_dir / "val.src"
    # val_tgt = output_dir / "val.tgt"
    
    # if all(f.exists() for f in [train_src, train_tgt, val_src, val_tgt]):
    #     print("=" * 80)
    #     print("✅ 数据文件已存在，跳过数据处理")
    #     print("=" * 80)
    #     print(f"  - {train_src} ({sum(1 for _ in open(train_src)):,} 行)")
    #     print(f"  - {val_src} ({sum(1 for _ in open(val_src)):,} 行)")
    #     print()
    #     return
    
    print("=" * 80)
    print("🚀 IWSLT2017 英德双向翻译数据集处理")
    print("=" * 80)
    
    print(f"\n⚙️  配置:")
    print(f"  数据集: {TAR_PATH}")
    print(f"  解压目录: {EXTRACT_DIR}")
    print(f"  输出目录: {output_dir}")
    print(f"  验证集比例: {VAL_RATIO * 100}%")
    # print(f"  最大句对数: {MAX_PAIRS:,} (训练样本约 {int(MAX_PAIRS * (1-VAL_RATIO) * 2):,} 条)")
    print(f"  随机种子: {SEED}")
    print()
    
    # Step 1: 解压数据
    en_file, de_file = extract_iwslt_data(TAR_PATH, EXTRACT_DIR)
    
    # Step 2: 加载平行语料
    pairs = load_parallel_data(en_file, de_file)
    
    # Step 3: 创建双向数据集（限制数据量）
    train_data, val_data = create_bidirectional_dataset(pairs, val_ratio=VAL_RATIO, seed=SEED, max_pairs=MAX_PAIRS)
    
    # Step 4: 保存到文件
    save_to_files(train_data, val_data, output_dir)
    
    print("\n" + "=" * 80)
    print("✅ 数据处理完成！")
    print("=" * 80)
    print(f"\n📝 数据格式说明:")
    print(f"  - 源文本包含语言标记: <2de> (目标德语) 或 <2en> (目标英语)")
    print(f"  - 支持双向翻译: English ↔ German")
    print(f"\n🎯 下一步:")
    print(f"  1. 检查数据: head -n 5 {output_dir}/train.src")
    print(f"  2. 开始训练: bash scripts/run.sh")
    print()


def main():
    """主函数（直接调用时使用）"""
    data_process()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
