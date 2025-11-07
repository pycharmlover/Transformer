set -e  # 一旦出错立即退出

# ========== 切换到项目根目录 ==========
cd "$(dirname "$0")/.." || exit 1  # 切换到脚本所在目录的上一级（项目根目录）
export PYTHONPATH="${PWD}:${PYTHONPATH}"  # 将项目根目录加入 Python 路径

# ========== 基本环境变量 ==========
export TOKENIZER_PATH=/home/extra_home/lc/google-bert/distilbert-base-multilingual-cased 
export SRC_LANG=src        # 源语言标识
export TGT_LANG=tgt        # 目标语言标识
export SEED=42             # 随机种子保证可复现
export CUDA_VISIBLE_DEVICES=0,1,2

# ========== 数据参数 ==========
SUBSET_SIZE=0              # 加载全部数据

# ========== 训练参数 ==========
# 针对500K-1M数据量优化的超参数配置
BATCH_SIZE=32              # batch size
EVAL_BATCH_SIZE=64
EPOCHS=20                 # epochs（翻译任务通常需要20-50个epoch）
NUM_LAYERS=6               # 层数
D_MODEL=512                # 模型维度
NUM_HEADS=8                # 注意力头
D_FF=2048                  # FFN维度
LR=3e-4                    # 学习率（数据多用更稳定的学习率）
OUTPUT_DIR=results_ablation         # 输出目录
DROPOUT=0.15               # 降低dropout
WEIGHT_DECAY=0.01          # 保持权重衰减

# 预计模型参数量：约180M（vs之前60M）
# 数据/参数比：500K/180M ≈ 0.0028（健康比例）

# ========== 正则化与早停 ==========
PATIENCE=5                 # Early stopping patience（验证loss不改善的容忍epoch数）
LABEL_SMOOTHING=0.1        # Label smoothing（0.0-0.2，有助于防止过拟合）
WARMUP_RATIO=0.06          # Warmup比例（前6%的steps线性warmup）

# ========== 注意力机制配置 ==========
# 注意力机制类型: 
#   - standard: 标准O(n^2)（使用T5风格相对位置编码）
#   - local_sparse: 局部稀疏注意力
#   - strided_sparse: 跨步稀疏注意力
#   - block_sparse: 块稀疏注意力
#   - linear: 线性注意力O(n)
#   - causal_linear: 因果线性注意力O(n)（用于decoder）
#   - performer: Performer注意力（FAVOR+）
ATTENTION_TYPE=standard

# ========== 创建结果目录 ==========
mkdir -p $OUTPUT_DIR/logs
mkdir -p $OUTPUT_DIR/checkpoints

# ========== 启动训练 ==========
echo "🚀 启动 Transformer 训练..."
echo "保存目录: $OUTPUT_DIR"
echo "--------------------------------------"

python src/train_ablation.py \
  --src_lang $SRC_LANG \
  --tgt_lang $TGT_LANG \
  --tokenizer_path $TOKENIZER_PATH \
  --batch_size $BATCH_SIZE \
  --eval_batch_size $EVAL_BATCH_SIZE \
  --epochs $EPOCHS \
  --num_layers $NUM_LAYERS \
  --d_model $D_MODEL \
  --num_heads $NUM_HEADS \
  --d_ff $D_FF \
  --lr $LR \
  --attention_type $ATTENTION_TYPE \
  --output_dir $OUTPUT_DIR \
  --seed $SEED \
  --dropout $DROPOUT \
  --weight_decay $WEIGHT_DECAY \
  --patience $PATIENCE \
  --label_smoothing $LABEL_SMOOTHING \
  --subset_size $SUBSET_SIZE \
  --warmup_ratio $WARMUP_RATIO \
  --cuda 

echo "✅ 训练完成！日志与模型保存在 $OUTPUT_DIR"
