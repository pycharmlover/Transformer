import os
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class TranslationDataset:
    """
    一个自定义 Dataset，用于加载和处理 TED Talks IWSLT 翻译数据。
    """
    def __init__(self, data,tokenizer,src_lang="en",tgt_lang="nl",max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]["translation"]
        src_text = item[self.src_lang]
        tgt_text = item[self.tgt_lang]
        src = self.tokenizer(
            src_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tgt = self.tokenizer(
            tgt_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": src["input_ids"].squeeze(0),
            "attention_mask": src["attention_mask"].squeeze(0),
            "labels": tgt["input_ids"].squeeze(0),
        }

class DatasetLoader:
    """
    封装数据加载、tokenizer创建与 DataLoader 构造。
    """
    def __init__(self,src_lang="nl",tgt_lang="en",year="2016", tokenizer_name="/home/extra_home/lc/google-bert/bert-base-multilingual-cased"):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.year = year
        self.tokenizer_name = tokenizer_name

    def load_raw_dataset(self):
        """
        加载 TED Talks IWSLT 数据集（使用本地的 ted_talks_iwslt.py）
        """
        dataset = load_dataset(
            "data/ted_talks_iwslt.py",
            name=f"{self.src_lang}_{self.tgt_lang}_{self.year}",
            trust_remote_code=True
        )
        return dataset
    
    def build_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        return tokenizer
    
    def build_dataloaders(self,dataset,tokenizer,batch_size=16):
        """
        构建 PyTorch DataLoader。
        """
        train_data = dataset["train"]
        train_dataset = TranslationDataset(train_data,tokenizer,self.src_lang,self.tgt_lang)
        train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
        return train_loader

class TextPairDataset(Dataset):
    """
    从 train.en / train.nl 文件加载双语文本对。
    每一行对应一个句对。
    """

    def __init__(self, src_path, tgt_path, tokenizer_path, max_src_len=128, max_tgt_len=128):
        assert os.path.exists(src_path) and os.path.exists(tgt_path), f"文件不存在: {src_path}, {tgt_path}"
        self.src_texts = [line.strip() for line in open(src_path, encoding="utf-8")]
        self.tgt_texts = [line.strip() for line in open(tgt_path, encoding="utf-8")]
        assert len(self.src_texts) == len(self.tgt_texts), "源语言和目标语言句子数不一致"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]
        return {"src_text": src_text, "tgt_text": tgt_text}

    def collate_fn(self, batch):
        """
        自定义 batch 拼接逻辑，动态 padding。
        """
        src_texts = [x["src_text"] for x in batch]
        tgt_texts = [x["tgt_text"] for x in batch]

        tokenizer = self.tokenizer

        src_enc = tokenizer(
            src_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_src_len
        )
        tgt_enc = tokenizer(
            tgt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_tgt_len
        )

        src_input_ids = src_enc["input_ids"]
        src_attention_mask = src_enc["attention_mask"]
        tgt_input_ids = tgt_enc["input_ids"]

        pad_token_id = tokenizer.pad_token_id
        cls_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id

        # decoder 输入：右移一位，并在开头加 BOS
        decoder_input_ids = torch.full_like(tgt_input_ids, pad_token_id)
        decoder_input_ids[:, 0] = cls_token_id
        decoder_input_ids[:, 1:] = tgt_input_ids[:, :-1].clone()

        labels = tgt_input_ids.clone()
        labels[labels == pad_token_id] = -100  # CrossEntropy 忽略 pad

        return {
            "src_input_ids": src_input_ids,
            "src_attention_mask": src_attention_mask,
            "tgt_input_ids": decoder_input_ids,
            "labels": labels,
            "pad_token_id": pad_token_id
        }

# if __name__ == "__main__":
#     """
#     测试脚本: 验证数据是否能正常加载和编码。
#     """
#     loader = DatasetLoader(src_lang="nl", tgt_lang="en", year="2016")
#     dataset = loader.load_raw_dataset()
#     tokenizer = loader.build_tokenizer()
#     dataloader = loader.build_dataloaders(dataset, tokenizer, batch_size=4)

#     print("✅ 数据加载成功！示例:")
#     for batch in dataloader:
#         print("input_ids:", batch["input_ids"].shape)
#         print("labels:", batch["labels"].shape)
#         break

# """
# 4	batch size	每次 DataLoader 返回 4 个样本（句子）
# 128	序列长度（max_length）	每个句子被截断或补齐到 128 个 token
# """