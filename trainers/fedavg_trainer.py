import torch
import copy
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from utils.evaluate import align_labels_with_tokens
from aggregators.fedavg import average_weights
from trainers.base_trainer import BaseFederatedTrainer  

class FedAvgTrainer(BaseFederatedTrainer):
    def __init__(self, model_init, tokenizer, label_list, device="cpu",
                 epochs=1, learning_rate=3e-5, scheduler_type="constant", batch_size=32):
        super().__init__(model_init, tokenizer, label_list, device)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.scheduler_type = scheduler_type
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
        train_dataset = self.preprocess(train_examples)
        args = TrainingArguments(
            per_device_train_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            logging_strategy="no",
            save_strategy="no",
            report_to="none",
            learning_rate=self.learning_rate,
            lr_scheduler_type=self.scheduler_type
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
        return self.aggregate(client_models)  # 使用 BaseFederatedTrainer 提供的聚合方法
