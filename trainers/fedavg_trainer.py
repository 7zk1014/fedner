import torch
import copy
import random
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from utils.evaluate import align_labels_with_tokens
from trainers.base_trainer import BaseFederatedTrainer  

def freeze_bert_layers(model, train_last_n=4):
    """
    只训练 BERT 的最后 n 层（如最后4层），其余全部冻结。
    """
    num_layers = model.bert.config.num_hidden_layers
    for name, param in model.named_parameters():
        if name.startswith("bert.encoder.layer."):
            layer_num = int(name.split(".")[3])
            if layer_num < num_layers - train_last_n:
                param.requires_grad = False
            else:
                param.requires_grad = True
        elif name.startswith("bert.embeddings.") or name.startswith("bert.pooler."):
            param.requires_grad = False
        else:
            # 分类头和其它部分默认训练
            param.requires_grad = True

def subsample_data(examples, sample_size=200):
    return random.sample(examples, min(sample_size, len(examples)))

class FedAvgTrainer(BaseFederatedTrainer):
    def __init__(self, model_init, tokenizer, label_list, device="cpu",
                 epochs=1, learning_rate=3e-5, lr_scheduler_type="constant", batch_size=32):
        super().__init__(model_init, tokenizer, label_list, device)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lr_scheduler_type = lr_scheduler_type
        self.batch_size = batch_size
        self.label2id = {l: i for i, l in enumerate(label_list)}

    def preprocess(self, examples):
        def _preprocess(example):
            tokenized = self.tokenizer(
                example["tokens"],
                truncation=True,
                is_split_into_words=True,
                padding="max_length",
                max_length=128
            )
            tokenized["labels"] = align_labels_with_tokens(tokenized, [example["labels"]], self.label2id)[0]
            return tokenized
        dataset = Dataset.from_list(examples)
        return dataset.map(_preprocess)

    def train_on_client(self, model, train_examples):
        sampled_data = subsample_data(train_examples, sample_size=200)
        freeze_bert_layers(model, train_last_n=4)  # 只训练最后4层
        train_dataset = self.preprocess(sampled_data)
        args = TrainingArguments(
            per_device_train_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            logging_strategy="no",
            save_strategy="no",
            report_to="none",
            learning_rate=self.learning_rate,
            lr_scheduler_type=self.lr_scheduler_type,
            fp16=True  # 如果不支持可以设为 False 或去掉
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer
        )
        trainer.train()
        return model

    def train_round(self, global_model, clients_data):
        client_models = []
        for data in clients_data:
            model = self.model_init().to(self.device)
            model.load_state_dict(copy.deepcopy(global_model.state_dict()))
            trained_model = self.train_on_client(model, data)
            client_models.append(trained_model.cpu())
        return self.aggregate(client_models)
