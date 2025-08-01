import torch
import copy
import random
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from utils.evaluate import align_labels_with_tokens
from trainers.base_trainer import BaseFederatedTrainer


def freeze_bert_layers(model, train_last_n=2):
    """
    只训练 BERT 的最后 n 层（如最后2层），其余全部冻结。
    """
    num_layers = model.bert.config.num_hidden_layers
    for name, param in model.named_parameters():
        if name.startswith("bert.encoder.layer."):
            layer_num = int(name.split(".")[3])
            param.requires_grad = layer_num >= num_layers - train_last_n
        elif name.startswith("bert.embeddings.") or name.startswith("bert.pooler."):
            param.requires_grad = False
        else:
            param.requires_grad = True


def subsample_data(examples, sample_size=200):
    return random.sample(examples, min(sample_size, len(examples)))


class FedAdamTrainer(BaseFederatedTrainer):
    def __init__(self, model_init, tokenizer, label_list, device="cpu",
                 epochs=2, learning_rate=5e-5, scheduler_type="constant",
                 batch_size=32, server_lr=0.01, train_last_n=4, sample_size=200):
        super().__init__(model_init, tokenizer, label_list, device)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.scheduler_type = scheduler_type
        self.batch_size = batch_size
        self.server_lr = server_lr
        self.train_last_n = train_last_n
        self.sample_size = sample_size
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.epsilon = 1e-8
        self.label2id = {l: i for i, l in enumerate(label_list)}
        self.momentum = {}
        self.v = {}

    def preprocess(self, examples):
        def _preprocess(example):
            tokenized = self.tokenizer(
                example["tokens"],
                truncation=True,
                is_split_into_words=True,
                padding="max_length",
                max_length=128,
            )
            tokenized["labels"] = align_labels_with_tokens(tokenized, [example["labels"]], self.label2id)[0]
            return tokenized

        dataset = Dataset.from_list(examples)
        return dataset.map(_preprocess)

    def train_on_client(self, model, train_examples):
        sampled_data = subsample_data(train_examples, self.sample_size)
        freeze_bert_layers(model, train_last_n=self.train_last_n)
        train_dataset = self.preprocess(sampled_data)
        args = TrainingArguments(
            per_device_train_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            logging_strategy="no",
            save_strategy="no",
            report_to="none",
            learning_rate=self.learning_rate,
            lr_scheduler_type=self.scheduler_type,
            fp16=True
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
        global_model.eval()
        client_models = []
        global_weights = global_model.state_dict()

        if not self.momentum:
            for name, param in global_model.named_parameters():
                if param.requires_grad:
                    self.momentum[name] = torch.zeros_like(param.data)
                    self.v[name] = torch.zeros_like(param.data)

        for data in clients_data:
            model = self.model_init().to(self.device)
            model.load_state_dict(copy.deepcopy(global_weights))
            trained_model = self.train_on_client(model, data)
            client_models.append(copy.deepcopy(trained_model.state_dict()))

        new_state = copy.deepcopy(global_weights)
        delta = {}

        for name in new_state:
            if name in self.momentum:
                client_tensors = [cm[name].to(self.device) for cm in client_models]
                global_tensor = global_weights[name].to(self.device)
                stacked = torch.stack([ct - global_tensor for ct in client_tensors])
                delta[name] = stacked.mean(dim=0)

        with torch.no_grad():
            for name in delta:
                self.momentum[name] = self.beta1 * self.momentum[name] + (1 - self.beta1) * delta[name]
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (delta[name] ** 2)
                update = self.momentum[name] / (self.v[name].sqrt() + self.epsilon)
                new_state[name] = global_weights[name].to(self.device) + self.server_lr * update

        global_model.load_state_dict(new_state)
        return global_model
