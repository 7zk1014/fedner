# trainers/fedavg_trainer.py
import torch
import copy
from transformers import Trainer, TrainingArguments, default_data_collator
from datasets import Dataset
from trainers.base_trainer import BaseFederatedTrainer
from utils.evaluate import align_labels_with_tokens
from utils.training_utils import freeze_model_layers, sample_client_data


class FedAvgTrainer(BaseFederatedTrainer):
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
        # Training control parameters
        train_last_n=4,
        sample_size=200,  # Number of samples per client (None=use all)
        sample_strategy="random",  # Sampling strategy
        max_seq_length=128  # Maximum sequence length
    ):
        super().__init__(model_init, tokenizer, label_list, device)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.scheduler_type = scheduler_type
        self.batch_size = int(batch_size)

        self.train_last_n = int(train_last_n)
        self.sample_size = None if sample_size in (None, "None") else int(sample_size)
        self.sample_strategy = sample_strategy
        self.max_seq_length = int(max_seq_length)

        self.label2id = {l: i for i, l in enumerate(label_list)}

    # ---------- Data ----------
    def preprocess(self, examples):
        """
        Convert (tokens, labels) to HuggingFace Dataset format.
        Uses fast tokenizer's word_ids for label alignment with padding to max_length.
        """
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

        # Preprocess data to torch tensors
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
            tokenizer=self.tokenizer,
            data_collator=default_data_collator
        )
        trainer.train()
        return model

    # ---------- One FedAvg Round ----------
    def train_round(self, global_model, clients_data):
        client_models = []
        for data in clients_data:
            model = self.model_init().to(self.device)
            model.load_state_dict(copy.deepcopy(global_model.state_dict()))
            trained_model = self.train_on_client(model, data)
            client_models.append(trained_model.cpu())

        aggregated_model = self.aggregate(client_models)

        # Communication cost estimation (MB, float32)
        total_uploaded = sum(
            sum(p.nelement() * p.element_size() for p in m.parameters())
            for m in client_models
        )
        self.last_uploaded_size_mb = total_uploaded / (1024 ** 2)

        return aggregated_model
