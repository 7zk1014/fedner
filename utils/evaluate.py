# utils/evaluate.py
import torch
from torch.utils.data import DataLoader
from seqeval.metrics import f1_score, precision_score, recall_score

# -----------------------------
# Normalization and filtering
# -----------------------------
def _normalize_examples(examples):
    """
    Normalize evaluation input to "indexable sample sequences" or HF Dataset:
      - Supports List[dict] / List[List[dict]] / Dict[Any, dict] / Dict[Any, List[dict]]
      - Automatically flattens nested structures
      - Returns datasets.Dataset as-is
    """
    if examples is None:
        return []

    # HF Dataset: return directly
    try:
        import datasets  # optional dependency
        if isinstance(examples, datasets.Dataset):
            return examples
    except Exception:
        pass

    # dict -> extract values
    if isinstance(examples, dict):
        vals = list(examples.values())
        # Dict[key, List[dict]] case: flatten
        if len(vals) > 0 and isinstance(vals[0], list):
            flat = []
            for v in vals:
                flat.extend(v)
            return flat
        return vals  # Dict[Any, dict]

    # List[List[dict]] -> flatten
    if isinstance(examples, list) and len(examples) > 0 and isinstance(examples[0], list):
        flat = []
        for sub in examples:
            flat.extend(sub)
        return flat

    return examples  # Already List[dict] or empty list


def _only_sentence_dicts(items):
    """
    Only keep sentence dictionaries of the form {'tokens': [...], 'labels': [...]}.
    Ignore all other types (str/int/dict without key fields, etc.).
    """
    out = []
    for x in items or []:
        if isinstance(x, dict) and "tokens" in x and "labels" in x:
            out.append(x)
    return out


# -----------------------------
# Single sample: tokenize + label alignment
# -----------------------------
def align_labels_with_tokens(tokenizer, tokens, labels, label2id, max_length=128, pad_to_max=True):
    """
    Align word-level labels to tokenizer's subword sequences.
    Returns dict containing input_ids/attention_mask/labels (tensor shape [1, L]).
    Requires fast tokenizer (supports .word_ids).
    """
    if "O" not in label2id:
        raise ValueError("label2id must contain 'O'.")

    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding=("max_length" if pad_to_max else True),
        max_length=max_length,
        return_tensors="pt",
    )

    # Align labels
    word_ids = enc.word_ids(batch_index=0)  # Requires fast tokenizer
    aligned = []
    for wi in word_ids:
        if wi is None:
            aligned.append(-100)
        else:
            lab = labels[wi]
            if lab not in label2id:
                raise KeyError(f"Unknown label: {lab}")
            aligned.append(label2id[lab])

    enc["labels"] = torch.tensor([aligned], dtype=torch.long)
    return enc


# -----------------------------
# Build evaluation DataLoader (reusable)
# -----------------------------
def build_eval_dataloader(test_examples, tokenizer, label_list, batch_size=32, max_length=128,
                          num_workers=0, pin_memory=False):
    """
    test_examples: List[{'tokens': [...], 'labels': [...]}] or compatible structure
    """
    test_examples = _normalize_examples(test_examples)
    test_examples = _only_sentence_dicts(test_examples)

    # Empty data: return empty DataLoader, evaluate_model will get 0 metrics
    if not test_examples:
        return DataLoader([], batch_size=batch_size)

    label2id = {l: i for i, l in enumerate(label_list)}
    if "O" not in label2id:
        raise ValueError("label_list must contain 'O'.")

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
                        raise KeyError(f"Unknown label: {lab}")
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
# Lightweight evaluation
# -----------------------------
def evaluate_model(model, tokenizer, test_examples, label_list,
                   device="cuda", batch_size=32, max_length=128, num_workers=0):
    """
    Lightweight evaluation avoiding Trainer overhead.
    Empty data returns 0 metrics; uses -100 mask to ignore padding labels.
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

    # Empty DataLoader case
    if dataloader is None:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    true_labels, pred_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 0:  # Defensive check
                continue
            input_ids = batch["input_ids"].to(device, non_blocking=pin)
            attention_mask = batch["attention_mask"].to(device, non_blocking=pin)
            labels = batch["labels"].to(device, non_blocking=pin)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            predictions = torch.argmax(logits, dim=-1)

            # Reconstruct true/predicted sequences (ignore -100 positions)
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
