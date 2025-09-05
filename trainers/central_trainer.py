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
    learning_rate=5e-5,
    scheduler_type="constant",
    batch_size=32,
    sample_size: int = 200,
    train_last_n_layers: int = 4
):
    if sample_size is not None and sample_size < len(train_examples):
        train_examples = random.sample(train_examples, sample_size)

    if train_last_n_layers is not None:
        # Freeze layers for BERT-like models with 12 encoder layers
        # Only train last N layers and classification head, freeze all others
        total_layers = len(model.bert.encoder.layer)  # e.g. 12
        trainable_layers = set(range(total_layers - train_last_n_layers, total_layers))
        for name, param in model.named_parameters():
            # BERT encoder layer names follow pattern: bert.encoder.layer.{i}.*
            if name.startswith("bert.encoder.layer"):
                layer_num = int(name.split(".")[3])
                if layer_num not in trainable_layers:
                    param.requires_grad = False
            # Ensure classification head is trainable (name might be classifier, head, tag_classifier etc)
            elif not name.startswith("classifier") and "classifier" in name:
                param.requires_grad = False

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
