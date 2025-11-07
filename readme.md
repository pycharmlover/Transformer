# Transformer 神经机器翻译项目



本项目从零实现了完整的 Transformer 编码器-解码器架构，用于英德双向机器翻译任务。实现基于原始论文 "[Attention is All You Need](https://arxiv.org/abs/1706.03762)"，并集成了多项现代改进技术。

## ✨ 主要特性

- 🔥 **完整实现**：包含缩放点积注意力、多头注意力、位置前馈网络、残差连接和层归一化
- 🚀 **现代改进**：
  - T5风格相对位置编码（替代绝对位置编码）
  - Pre-Layer Normalization（提升训练稳定性）
  - Label Smoothing、Gradient Clipping、Warmup调度
- 💡 **高效注意力**：实现了稀疏注意力、线性注意力、Performer等机制
- 📊 **实验完善**：支持多种对比实验和消融实验
- 🎯 **高性能**：在IWSLT2017数据集上达到8.20 PPL，60.71%准确率
- 🔧 **易于使用**：模块化设计，一键启动训练，完整日志和可视化

## 📁 项目结构

```
Transformer/
├── src/                          # 核心源代码
│   ├── models/                   # 模型架构实现
│   │   ├── transformer.py        # 主模型（编码器-解码器）
│   │   ├── encoder.py            # Transformer编码器
│   │   ├── decoder.py            # Transformer解码器
│   │   ├── attention.py          # 注意力机制
│   │   ├── relative_positional_encoding.py  # T5相对位置编码
│   │   ├── ffn.py                # 位置前馈网络
│   │   ├── sparse_attention.py   # 稀疏注意力
│   │   └── linear_attention.py   # 线性注意力与Performer
│   ├── utils/                    # 工具函数
│   │   ├── data_utils.py         # 数据处理工具
│   │   └── plot_utils.py         # 可视化工具
│   ├── train.py                  # 主训练脚本
│   └── data_process.py           # 数据加载与预处理
│
├── scripts/                      # 实验脚本
│   ├── run.sh                    # 主实验启动脚本
│   ├── experiments.sh            # 批量对比实验
│   └── run_translate.sh          # 翻译推理脚本
│
├── results/                      # 主实验结果（自动生成）
├── data/                         # 数据处理结果
├── figures/                      # 不同num_heads结果对比报告图
├── translator.py                 # 交互式翻译工具
├── requirements.txt              # Python依赖
└── README.md                     # 本文件
```

## 🚀 快速开始

### 1. 环境配置

**使用 Conda（推荐）**
```bash
# 创建虚拟环境
conda create -n transformer python=3.10
conda activate transformer

# 安装PyTorch（根据你的CUDA版本选择）
# CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install -r requirements.txt
```

**使用 pip**
```bash
pip install -r requirements.txt
```

### 2. 数据准备
建议：将数据集下载到本地，之后运行下述代码
```bash
python src/data_process.py
```

### 3. 训练模型

**方式1：使用脚本（推荐）**
```bash
bash scripts/run.sh
```

**方式2：直接运行Python**
```bash
python src/train.py \
    --batch_size 32 \
    --num_epochs 20 \
    --lr 5e-4 \
    --d_model 512 \
    --num_heads 8 \
    --num_layers 6 \
    --d_ff 2048 \
    --dropout 0.1 \
    --max_seq_length 128
```

### 4. 翻译推理

**交互式翻译**
```bash
python translator.py --checkpoint results/checkpoints/best_model.pt
```

**批量翻译**
```bash
bash scripts/run_translate.sh
```

## 🙏 致谢

感谢以下开源项目：
- PyTorch
- Hugging Face Transformers
- IWSLT数据集

---

**注意**：本项目用于学习和研究目的。如需在生产环境使用，请进行充分的测试和优化。
