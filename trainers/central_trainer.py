import torch
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from utils.evaluate import align_labels_with_tokens

def centralized_train(model, tokenizer, train_examples, label_list, device="cpu", epochs=10):
    label2id = {label: i for i, label in enumerate(label_list)}

    def preprocess(example):
        tokenized = tokenizer(example["tokens"],
                              truncation=True,
                              is_split_into_words=True,
                              padding="max_length",
                              max_length=128)
        tokenized["labels"] = align_labels_with_tokens(tokenized, [example["labels"]], label2id)[0]
        return tokenized

    dataset = Dataset.from_list(train_examples).map(preprocess)

    args = TrainingArguments(
        output_dir="./central_tmp",
        per_device_train_batch_size=32,
        num_train_epochs=epochs,
        logging_strategy="epoch",
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer)
    trainer.train()
    return model
