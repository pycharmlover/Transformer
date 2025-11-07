#!/bin/bash

# ================================================================
# Transformer英德双向翻译脚本
# 支持: 英语 (en) ↔ 德语 (de)
# ================================================================

# 进入项目根目录
cd "$(dirname "$0")/.."

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# ====== 翻译配置 ======
MODEL_PATH="results/best_model.pt"
BEAM_SIZE=5                            # Beam search宽度
MAX_LENGTH=128                         # 最大生成长度
DEVICE="auto"                          # 设备 (auto/cuda/cpu)

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型文件不存在: $MODEL_PATH"
    echo ""
    echo "请先训练模型: bash scripts/run.sh"
    exit 1
fi

# 检查tokenizer
TOKENIZER_WITH_TAGS="data/tokenizer_with_tags"
if [ ! -d "$TOKENIZER_WITH_TAGS" ]; then
    echo "⚠️  警告: 未找到包含语言标记的tokenizer: $TOKENIZER_WITH_TAGS"
    echo "   需要重新训练模型"
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║        🌍 Transformer 英德双向翻译系统 (IWSLT2017)             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 配置:"
echo "  - 模型: $MODEL_PATH"
echo "  - Beam Size: $BEAM_SIZE"
echo "  - 最大长度: $MAX_LENGTH"
echo "  - 设备: $DEVICE"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# ====== 翻译示例 ======

# 示例1: 英语 → 德语
echo "🔹 示例 1: English → German"
python -c "
from translator import Translator
translator = Translator('$MODEL_PATH', device='$DEVICE')
result = translator.translate(
    text='Thank you so much, Chris.',
    src_lang='en',
    tgt_lang='de',
    beam_size=$BEAM_SIZE,
    max_length=$MAX_LENGTH
)
print(f'输入: {result[\"input\"]}')
print(f'输出: {result[\"translation\"]}')
print()
"

# 示例2: 英语 → 德语 (长句子)
echo "🔹 示例 2: English → German (长句)"
python -c "
from translator import Translator
translator = Translator('$MODEL_PATH', device='$DEVICE')
result = translator.translate(
    text='And it is truly a great honor to have the opportunity to come to this stage twice; I am extremely grateful.',
    src_lang='en',
    tgt_lang='de',
    beam_size=$BEAM_SIZE,
    max_length=$MAX_LENGTH
)
print(f'输入: {result[\"input\"]}')
print(f'输出: {result[\"translation\"]}')
print()
"

# 示例3: 德语 → 英语
echo "🔹 示例 3: German → English"
python -c "
from translator import Translator
translator = Translator('$MODEL_PATH', device='$DEVICE')
result = translator.translate(
    text='Vielen Dank, Chris.',
    src_lang='de',
    tgt_lang='en',
    beam_size=$BEAM_SIZE,
    max_length=$MAX_LENGTH
)
print(f'输入: {result[\"input\"]}')
print(f'输出: {result[\"translation\"]}')
print()
"

# 示例4: 德语 → 英语 (长句子)
echo "🔹 示例 4: German → English (长句)"
python -c "
from translator import Translator
translator = Translator('$MODEL_PATH', device='$DEVICE')
result = translator.translate(
    text='Es ist mir wirklich eine Ehre, zweimal auf dieser Bühne stehen zu dürfen. Tausend Dank dafür.',
    src_lang='de',
    tgt_lang='en',
    beam_size=$BEAM_SIZE,
    max_length=$MAX_LENGTH
)
print(f'输入: {result[\"input\"]}')
print(f'输出: {result[\"translation\"]}')
print()
"

echo "════════════════════════════════════════════════════════════════"
echo "✅ 翻译演示完成！"
echo ""
echo "💡 使用说明:"
echo "   1. 支持的语言对: en ↔ de"
echo "   2. 模型使用语言标记: <2en> (目标英语), <2de> (目标德语)"
echo "   3. 数据集: IWSLT2017 英德双向翻译 (370K训练样本)"
echo ""
echo "📝 自定义翻译:"
echo "   python -c \""
echo "   from translator import Translator"
echo "   t = Translator('$MODEL_PATH')"
echo "   print(t.translate('Your text here', 'en', 'de')['translation'])"
echo "   \""
echo ""
