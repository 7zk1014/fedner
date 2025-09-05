# trainers/fedprox_trainer.py
import torch
import copy
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from utils.evaluate import align_labels_with_tokens
from trainers.base_trainer import BaseFederatedTrainer
from utils.training_utils import freeze_model_layers, sample_client_data


class FedProxTrainer(BaseFederatedTrainer):
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
        mu=0.01,
        train_last_n=4,
        sample_size=200,
        sample_strategy="random",
        max_seq_length=128
    ):
        super().__init__(model_init, tokenizer, label_list, device)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.scheduler_type = str(scheduler_type)
        self.batch_size = int(batch_size)
        self.mu = float(mu)

        self.train_last_n = int(train_last_n)
        self.sample_size = None if sample_size in (None, "None") else int(sample_size)
        self.sample_strategy = sample_strategy
        self.max_seq_length = int(max_seq_length)

        self.label2id = {l: i for i, l in enumerate(label_list)}

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

    def train_on_client(self, model, train_examples, global_params):
        sampled_data = sample_client_data(
            data=train_examples,
            sample_size=self.sample_size,
            strategy=self.sample_strategy,
            with_replacement=False,
            seed=None
        )

        freeze_model_layers(
            model,
            train_last_n_layers=self.train_last_n,
            freeze_embeddings=True
        )

        train_dataset = self.preprocess(sampled_data)
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        use_fp16 = torch.cuda.is_available()

        # Custom Trainer with FedProx regularization
        class FedProxHFTrainer(Trainer):
            def __init__(self, mu, global_params, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.mu = float(mu)
                # Store CPU version of global parameter snapshot to avoid GPU memory usage
                self.global_params = {k: v.detach().clone().cpu() for k, v in global_params.items()} if global_params else None
        
            def compute_loss(
                self,
                model,
                inputs,
                return_outputs: bool = False,
                num_items_in_batch=None,   # Compatible with new version transformers
                **kwargs,                  # Fallback for possible future parameters
            ):
                # Use parent class to calculate base loss (compatible with label_smoothing etc), pass new parameters through
                loss, outputs = super().compute_loss(
                    model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch, **kwargs
                )
        
                # FedProx proximal regularization: (μ/2) * Σ ||w - w_global||^2, applied only to trainable parameters
                if self.global_params is not None and self.mu > 0:
                    prox = 0.0
                    for name, p in model.named_parameters():
                        if not p.requires_grad:
                            continue
                        gp = self.global_params.get(name, None)
                        if gp is None:
                            continue
                        diff = (p - gp.to(p.device))
                        prox = prox + (diff * diff).sum()
                    loss = loss + 0.5 * self.mu * prox
        
                return (loss, outputs) if return_outputs else loss


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

        trainer = FedProxHFTrainer(
            mu=self.mu,
            global_params=global_params,
            model=model,
            args=args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer
        )
        trainer.train()
        return model

    def train_round(self, global_model, clients_data):
        global_model.eval()
        global_params = copy.deepcopy(global_model.state_dict())
        client_models = []

        for data in clients_data:
            model = self.model_init().to(self.device)
            model.load_state_dict(copy.deepcopy(global_params))
            trained_model = self.train_on_client(model, data, global_params)
            client_models.append(copy.deepcopy(trained_model.state_dict()))

        new_state = copy.deepcopy(global_params)
        with torch.no_grad():
            for key in new_state:
                client_tensors = [cm[key].to(self.device) for cm in client_models]
                new_state[key] = torch.stack(client_tensors).mean(dim=0)

        global_model.load_state_dict(new_state)

        # Communication cost estimation
        uploaded_bytes = 0
        for param in global_params.values():
            uploaded_bytes += param.numel() * 4  # float32
        self.last_uploaded_size_mb = uploaded_bytes * len(clients_data) / (1024 ** 2)

        return global_model
