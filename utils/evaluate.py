# utils/evaluate.py
import torch
from torch.utils.data import DataLoader
from seqeval.metrics import f1_score, precision_score, recall_score

# -----------------------------
# 标准化与过滤
# -----------------------------
def _normalize_examples(examples):
    """
    将评估输入标准化为“可索引的样本序列”或 HF Dataset：
      - 支持 List[dict] / List[List[dict]] / Dict[Any, dict] / Dict[Any, List[dict]]
      - 对嵌套结构自动展平
      - 对 datasets.Dataset 原样返回
    """
    if examples is None:
        return []

    # HF Dataset：直接返回
    try:
        import datasets  # 可选依赖
        if isinstance(examples, datasets.Dataset):
            return examples
    except Exception:
        pass

    # dict -> 取 values
    if isinstance(examples, dict):
        vals = list(examples.values())
        # Dict[key, List[dict]] 情况：展平
        if len(vals) > 0 and isinstance(vals[0], list):
            flat = []
            for v in vals:
                flat.extend(v)
            return flat
        return vals  # Dict[Any, dict]

    # List[List[dict]] -> 展平
    if isinstance(examples, list) and len(examples) > 0 and isinstance(examples[0], list):
        flat = []
        for sub in examples:
            flat.extend(sub)
        return flat

    return examples  # 已是 List[dict] 或空列表


def _only_sentence_dicts(items):
    """
    只保留形如 {'tokens': [...], 'labels': [...]} 的句子字典。
    其余类型（str/int/不含关键键的 dict 等）全部忽略。
    """
    out = []
    for x in items or []:
        if isinstance(x, dict) and "tokens" in x and "labels" in x:
            out.append(x)
    return out


# -----------------------------
# 单条样本：tokenize + 标签对齐
# -----------------------------
def align_labels_with_tokens(tokenizer, tokens, labels, label2id, max_length=128, pad_to_max=True):
    """
    将 word-level 标签对齐到 tokenizer 的 subword 序列上。
    返回包含 input_ids/attention_mask/labels 的 dict（张量形状为 [1, L]）。
    需要 fast tokenizer（支持 .word_ids）
    """
    if "O" not in label2id:
        raise ValueError("label2id 必须包含 'O'。")

    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding=("max_length" if pad_to_max else True),
        max_length=max_length,
        return_tensors="pt",
    )

    # 对齐标签
    word_ids = enc.word_ids(batch_index=0)  # fast tokenizer
    aligned = []
    for wi in word_ids:
        if wi is None:
            aligned.append(-100)
        else:
            lab = labels[wi]
            if lab not in label2id:
                raise KeyError(f"未知标签：{lab}")
            aligned.append(label2id[lab])

    enc["labels"] = torch.tensor([aligned], dtype=torch.long)
    return enc


# -----------------------------
# 构建评估 DataLoader（可复用）
# -----------------------------
def build_eval_dataloader(test_examples, tokenizer, label_list, batch_size=32, max_length=128,
                          num_workers=0, pin_memory=False):
    """
    test_examples: List[{'tokens': [...], 'labels': [...]}] 或兼容结构
    """
    test_examples = _normalize_examples(test_examples)
    test_examples = _only_sentence_dicts(test_examples)

    # 空数据：返回空 DataLoader，后续 evaluate_model 会得到 0 指标
    if not test_examples:
        return DataLoader([], batch_size=batch_size)

    label2id = {l: i for i, l in enumerate(label_list)}
    if "O" not in label2id:
        raise ValueError("label_list 必须包含 'O'。")

    def collate_fn(batch):
        tokens_batch = [ex["tokens"] for ex in batch]
        labels_batch = [ex["labels"] for ex in batch]

        enc = tokenizer(
            tokens_batch,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        aligned_labels = []
        for i, lab_seq in enumerate(labels_batch):
            word_ids = enc.word_ids(batch_index=i)
            lab_ids = []
            for wi in word_ids:
                if wi is None:
                    lab_ids.append(-100)
                else:
                    lab = lab_seq[wi]
                    if lab not in label2id:
                        raise KeyError(f"未知标签：{lab}")
                    lab_ids.append(label2id[lab])
            aligned_labels.append(lab_ids)

        enc["labels"] = torch.tensor(aligned_labels, dtype=torch.long)
        return enc

    return DataLoader(
        test_examples,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


# -----------------------------
# 轻量级评估
# -----------------------------
def evaluate_model(model, tokenizer, test_examples, label_list,
                   device="cuda", batch_size=32, max_length=128, num_workers=0):
    """
    避免 Trainer 开销的轻量级评估。
    空数据直接返回 0 指标；使用 -100 掩码忽略 padding 标签。
    """
    test_examples = _normalize_examples(test_examples)
    test_examples = _only_sentence_dicts(test_examples)
    if not test_examples:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    model.eval()
    id2label = {i: l for i, l in enumerate(label_list)}
    pin = (isinstance(device, str) and device.startswith("cuda")) or \
          (isinstance(device, torch.device) and device.type == "cuda")

    dataloader = build_eval_dataloader(
        test_examples, tokenizer, label_list,
        batch_size=batch_size, max_length=max_length,
        num_workers=num_workers, pin_memory=pin
    )

    # 空 DataLoader 情况
    if dataloader is None:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    true_labels, pred_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 0:  # 防御式判断
                continue
            input_ids = batch["input_ids"].to(device, non_blocking=pin)
            attention_mask = batch["attention_mask"].to(device, non_blocking=pin)
            labels = batch["labels"].to(device, non_blocking=pin)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            predictions = torch.argmax(logits, dim=-1)

            # 还原真实/预测序列（忽略 -100 的位置）
            for i in range(labels.size(0)):
                true_seq, pred_seq = [], []
                for p, l in zip(predictions[i], labels[i]):
                    l = l.item()
                    if l == -100:
                        continue
                    true_seq.append(id2label[l])
                    pred_seq.append(id2label[p.item()])
                true_labels.append(true_seq)
                pred_labels.append(pred_seq)

    return {
        "f1": f1_score(true_labels, pred_labels),
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
    }
