import random
import torch
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from utils.evaluate import align_labels_with_tokens

def centralized_train(
    model,
    tokenizer,
    train_examples,
    label_list,
    device="auto",
    epochs=10,
    learning_rate=3e-5,
    scheduler_type="constant",
    batch_size=32,
    sample_size: int = 200,         # 新增：抽样大小
    train_last_n_layers: int = 4  # 新增：只训练后 N 层
):
    # 1) 抽样
    if sample_size is not None and sample_size < len(train_examples):
        train_examples = random.sample(train_examples, sample_size)

    # 2) 冻结参数
    if train_last_n_layers is not None:
        # 假设你用的是 Bert-like 模型，encoder 有 12 层
        # 只让最后 N 层和分类头可训练，其他全部 freeze
        total_layers = len(model.bert.encoder.layer)  # e.g. 12
        trainable_layers = set(range(total_layers - train_last_n_layers, total_layers))
        for name, param in model.named_parameters():
            # Bert encoder 层名称一般形如 bert.encoder.layer.{i}.*
            if name.startswith("bert.encoder.layer"):
                layer_num = int(name.split(".")[3])
                if layer_num not in trainable_layers:
                    param.requires_grad = False
            # 保证分类头可训练（名称可能是 classifier、head、tag_classifier 等）
            elif not name.startswith("classifier") and "classifier" in name:
                # 这里要根据你的实际 head 模块名称来写
                param.requires_grad = False
            # 其它参数默认保持原样

    label2id = {label: i for i, label in enumerate(label_list)}

    def preprocess(example):
        tokenized = tokenizer(
            example["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=128
        )
        tokenized["labels"] = align_labels_with_tokens(
            tokenized, [example["labels"]], label2id
        )[0]
        return tokenized

    dataset = Dataset.from_list(train_examples).map(preprocess, batched=False)

    args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        lr_scheduler_type=scheduler_type,
        logging_strategy="epoch",
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model.to(device),
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )
    trainer.train()
    return model