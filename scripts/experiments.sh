#!/bin/bash

# 实验脚本：对比不同配置
# 使用方法：bash scripts/experiments.sh

set -e

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# 基本配置
export TOKENIZER_PATH=/home/extra_home/lc/google-bert/bert-base-multilingual-cased
export SRC_LANG=src        # 新数据集使用通用标识
export TGT_LANG=tgt        # 新数据集使用通用标识
export SEED=42
export CUDA_VISIBLE_DEVICES=0,1,2

# 通用训练参数
BATCH_SIZE=32
EVAL_BATCH_SIZE=64
EPOCHS=20
SUBSET_SIZE=0              # 0=使用全部数据（500K-1M条）
D_MODEL=512
D_FF=2048
LR=3e-4
NUM_LAYERS=6

echo "=========================================="
echo "开始实验：对比不同注意力头数"
echo "=========================================="

# 实验1：对比不同注意力头数（固定d_model=256）
# for NUM_HEADS in 2 4 6 10; do
#     NUM_LAYERS=6
#     OUTPUT_DIR=home/extra_home/lc/Transformer/comparison_heads/heads_${NUM_HEADS}
    
#     echo ""
#     echo "运行实验: NUM_HEADS=$NUM_HEADS, D_MODEL=$D_MODEL"
#     echo "输出目录: $OUTPUT_DIR"
    
#     mkdir -p $OUTPUT_DIR/logs
#     mkdir -p $OUTPUT_DIR/checkpoints
    
#     python src/train.py \
#       --src_lang $SRC_LANG \
#       --tgt_lang $TGT_LANG \
#       --tokenizer_path $TOKENIZER_PATH \
#       --batch_size $BATCH_SIZE \
#       --eval_batch_size $EVAL_BATCH_SIZE \
#       --epochs $EPOCHS \
#       --subset_size $SUBSET_SIZE \
#       --num_layers $NUM_LAYERS \
#       --d_model $D_MODEL \
#       --num_heads $NUM_HEADS \
#       --d_ff $D_FF \
#       --lr $LR \
#       --attention_type standard \
#       --output_dir $OUTPUT_DIR \
#       --seed $SEED \
#       --cuda
    
#     echo "✅ 完成: NUM_HEADS=$NUM_HEADS"
# done

#!/bin/bash

set -e  # 确保遇到错误时退出

for NUM_HEADS in 1 16 32; do 
    # for NUM_LAYERS in 2 4 6 8 10; do 
      OUTPUT_DIR=/home/extra_home/lc/Transformer/comparison_heads/heads_${NUM_HEADS}
      
      echo ""
      echo "运行实验: NUM_HEADS=$NUM_HEADS"
      echo "输出目录: $OUTPUT_DIR"
      
      mkdir -p $OUTPUT_DIR/logs
      mkdir -p $OUTPUT_DIR/checkpoints
      
      python src/train.py \
        --src_lang $SRC_LANG \
        --tgt_lang $TGT_LANG \
        --tokenizer_path $TOKENIZER_PATH \
        --batch_size $BATCH_SIZE \
        --eval_batch_size $EVAL_BATCH_SIZE \
        --epochs $EPOCHS \
        --subset_size $SUBSET_SIZE \
        --num_layers $NUM_LAYERS \
        --d_model $D_MODEL \
        --num_heads $NUM_HEADS \
        --d_ff $D_FF \
        --lr $LR \
        --attention_type standard \
        --output_dir $OUTPUT_DIR \
        --seed $SEED \
        --cuda
      
      echo "✅ 完成: NUM_HEADS=$NUM_HEADS, NUM_LAYERS=$NUM_LAYERS"
    # done
done


# echo ""
# echo "=========================================="
# echo "开始实验：对比不同层数"
# echo "=========================================="

# # 实验2：对比不同层数
# for NUM_LAYERS in 2 4 8 10; do
#     NUM_HEADS=8
#     OUTPUT_DIR=home/extra_home/lc/Transformer/comparison_layers/layers_${NUM_LAYERS}
    
#     echo ""
#     echo "运行实验: NUM_LAYERS=$NUM_LAYERS"
#     echo "输出目录: $OUTPUT_DIR"
    
#     mkdir -p $OUTPUT_DIR/logs
#     mkdir -p $OUTPUT_DIR/checkpoints
    
#     python src/train.py \
#       --src_lang $SRC_LANG \
#       --tgt_lang $TGT_LANG \
#       --tokenizer_path $TOKENIZER_PATH \
#       --batch_size $BATCH_SIZE \
#       --eval_batch_size $EVAL_BATCH_SIZE \
#       --epochs $EPOCHS \
#       --subset_size $SUBSET_SIZE \
#       --num_layers $NUM_LAYERS \
#       --d_model $D_MODEL \
#       --num_heads $NUM_HEADS \
#       --d_ff $D_FF \
#       --lr $LR \
#       --attention_type standard \
#       --output_dir $OUTPUT_DIR \
#       --seed $SEED \
#       --cuda
    
#     echo "✅ 完成: NUM_LAYERS=$NUM_LAYERS"
# done

# echo ""
# echo "=========================================="
# echo "开始实验：对比不同注意力机制"
# echo "=========================================="

# 实验3：对比不同注意力机制
# for ATTN_TYPE in standard linear performer; do
#     NUM_LAYERS=6
#     NUM_HEADS=8
#     OUTPUT_DIR=home/extra_home/lc/Transformer/comparison_attention/attn_${ATTN_TYPE}
    
#     echo ""
#     echo "运行实验: ATTENTION_TYPE=$ATTN_TYPE"
#     echo "输出目录: $OUTPUT_DIR"
    
#     mkdir -p $OUTPUT_DIR/logs
#     mkdir -p $OUTPUT_DIR/checkpoints
    
#     python src/train.py \
#       --src_lang $SRC_LANG \
#       --tgt_lang $TGT_LANG \
#       --tokenizer_path $TOKENIZER_PATH \
#       --batch_size $BATCH_SIZE \
#       --eval_batch_size $EVAL_BATCH_SIZE \
#       --epochs $EPOCHS \
#       --subset_size $SUBSET_SIZE \
#       --num_layers $NUM_LAYERS \
#       --d_model $D_MODEL \
#       --num_heads $NUM_HEADS \
#       --d_ff $D_FF \
#       --lr $LR \
#       --attention_type $ATTN_TYPE \
#       --output_dir $OUTPUT_DIR \
#       --seed $SEED \
#       --cuda
    
#     echo "✅ 完成: ATTENTION_TYPE=$ATTN_TYPE"
# done

# echo ""
# echo "=========================================="
# echo "所有对比实验完成！"
# echo "=========================================="
# echo ""
# echo "结果保存在以下目录："
# echo "  - results/comparison_heads/    (不同注意力头数)"
# echo "  - results/comparison_layers/   (不同层数)"
# echo "  - results/comparison_attention/ (不同注意力机制)"
# echo ""
# echo "查看每个实验的 metrics.txt 文件获取详细指标"
