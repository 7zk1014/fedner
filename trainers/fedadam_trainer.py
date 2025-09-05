# trainers/fedadam_trainer.py
import torch
import copy
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from utils.evaluate import align_labels_with_tokens
from trainers.base_trainer import BaseFederatedTrainer
from utils.training_utils import freeze_model_layers, sample_client_data


class FedAdamTrainer(BaseFederatedTrainer):
    def __init__(
        self,
        model_init,
        tokenizer,
        label_list,
        device="cpu",
        epochs=2,
        learning_rate=5e-5,
        scheduler_type="constant",
        batch_size=32,
        # FedAdam hyperparameters
        server_lr=0.0005,
        beta1=0.9,
        beta2=0.99,
        epsilon=1e-8,
        # Training control parameters
        train_last_n=4,
        sample_size=None,
        sample_strategy="random",
        max_seq_length=128
    ):
        super().__init__(model_init, tokenizer, label_list, device)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.scheduler_type = str(scheduler_type)
        self.batch_size = int(batch_size)

        self.server_lr = float(server_lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)

        self.train_last_n = int(train_last_n)
        self.sample_size = None if sample_size in (None, "None") else int(sample_size)
        self.sample_strategy = sample_strategy
        self.max_seq_length = int(max_seq_length)

        self.label2id = {l: i for i, l in enumerate(label_list)}
        # Server-side first and second moments for adaptive optimization
        self.momentum = {}
        self.v = {}

    # ---------- Data ----------
    def preprocess(self, examples):
        """(tokens, labels) -> HF Dataset"""
        def _preprocess(example):
            enc = align_labels_with_tokens(
                self.tokenizer,
                example["tokens"],
                example["labels"],
                self.label2id,
                max_length=self.max_seq_length,
                pad_to_max=True
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0).tolist(),
                "attention_mask": enc["attention_mask"].squeeze(0).tolist(),
                "labels": enc["labels"].squeeze(0).tolist(),
            }

        ds = Dataset.from_list(examples)
        ds = ds.map(_preprocess, remove_columns=ds.column_names)
        return ds

    # ---------- Local Train on a Single Client ----------
    def train_on_client(self, model, train_examples):
        # Sample client data
        sampled_data = sample_client_data(
            data=train_examples,
            sample_size=self.sample_size,
            strategy=self.sample_strategy,
            with_replacement=False,
            seed=None
        )

        # Freeze model layers (train only last N layers)
        freeze_model_layers(
            model,
            train_last_n_layers=self.train_last_n,
            freeze_embeddings=True
        )

        # Preprocess data
        train_dataset = self.preprocess(sampled_data)
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        use_fp16 = torch.cuda.is_available()
        args = TrainingArguments(
            per_device_train_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            logging_strategy="no",
            save_strategy="no",
            report_to="none",
            learning_rate=self.learning_rate,
            lr_scheduler_type=self.scheduler_type,
            fp16=use_fp16
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer
        )
        trainer.train()
        return model

    # ---------- One FedAdam Round ----------
    def train_round(self, global_model, clients_data):
        global_model.eval()
        client_models = []
        global_weights = global_model.state_dict()

        # Initialize first and second moments for FedAdam
        if not self.momentum:
            for name, param in global_model.named_parameters():
                if param.requires_grad:
                    self.momentum[name] = torch.zeros_like(param.data)
                    self.v[name] = torch.zeros_like(param.data)

        # Local client training
        for data in clients_data:
            model = self.model_init().to(self.device)
            model.load_state_dict(copy.deepcopy(global_weights))
            trained_model = self.train_on_client(model, data)
            client_models.append(copy.deepcopy(trained_model.state_dict()))

        # Compute average client gradients (delta)
        new_state = copy.deepcopy(global_weights)
        delta = {}
        for name in new_state:
            if name in self.momentum:  # Apply adaptive optimization to trainable parameters
                client_tensors = [cm[name].to(self.device) for cm in client_models]
                global_tensor = global_weights[name].to(self.device)
                stacked = torch.stack([ct - global_tensor for ct in client_tensors])
                delta[name] = stacked.mean(dim=0)

        # Server-side Adam update
        with torch.no_grad():
            for name in delta:
                self.momentum[name] = self.beta1 * self.momentum[name] + (1 - self.beta1) * delta[name]
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (delta[name] ** 2)
                update = self.momentum[name] / (self.v[name].sqrt() + self.epsilon)
                new_state[name] = global_weights[name].to(self.device) + self.server_lr * update

        global_model.load_state_dict(new_state)

        # Communication cost estimation (MB)
        uploaded_bytes = 0
        for name, param in global_weights.items():
            if name in self.momentum:
                uploaded_bytes += param.numel() * 4  # float32 bytes
        self.last_uploaded_size_mb = uploaded_bytes * len(clients_data) / (1024 ** 2)

        return global_model
