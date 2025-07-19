import copy
import torch
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from trainers.base_trainer import BaseFederatedTrainer
from utils.evaluate import align_labels_with_tokens

class FedAdamTrainer(BaseFederatedTrainer):
    def __init__(self, model_init, tokenizer, label_list, device="cpu",
                 epochs=1, learning_rate=3e-5, scheduler_type="constant", batch_size=32, server_lr=0.01):
        super().__init__(model_init, tokenizer, label_list, device)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.scheduler_type = scheduler_type
        self.batch_size = batch_size
        self.server_lr = server_lr
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
        global_model.eval()  # 聚合时不进行 dropout 等操作
        client_models = []
        global_weights = global_model.state_dict()

        # 初始化 FedAdam 动量参数（仅第一次）
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

        # 平均每个参数的客户端更新值
        for name in new_state:
            if name in self.momentum:  # 只更新需要梯度的部分
                client_tensors = [cm[name].to(self.device) for cm in client_models]
                global_tensor = global_weights[name].to(self.device)
                stacked = torch.stack([ct - global_tensor for ct in client_tensors])
                delta[name] = stacked.mean(dim=0)

        # FedAdam 服务端更新
        with torch.no_grad():
            for name in delta:
                self.momentum[name] = self.beta1 * self.momentum[name] + (1 - self.beta1) * delta[name]
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (delta[name] ** 2)
                update = self.momentum[name] / (self.v[name].sqrt() + self.epsilon)
                new_state[name] = global_weights[name].to(self.device) + self.server_lr * update

        global_model.load_state_dict(new_state)
        return global_model

