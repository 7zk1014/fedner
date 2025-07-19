import copy
import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from trainers.base_trainer import BaseFederatedTrainer
from utils.evaluate import align_labels_with_tokens


def get_client_model(global_model, device):
    """Return a client copy of the global model."""
    return copy.deepcopy(global_model).to(device)


def train_local_model(model, tokenizer, train_examples, label_list, device, epochs, prox_mu=0.0, global_weights=None):
    """Train a model on the client's data with optional FedProx regularization."""
    label2id = {l: i for i, l in enumerate(label_list)}

    def preprocess(example):
        tokenized = tokenizer(
            example["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=128,
        )
        tokenized["labels"] = align_labels_with_tokens(tokenized, [example["labels"]], label2id)[0]
        return tokenized

    dataset = Dataset.from_list(train_examples).map(preprocess)

    class ProxTrainer(Trainer):
        def __init__(self, prox_mu, global_weights, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.prox_mu = prox_mu
            self.global_weights = global_weights

        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(**inputs)
            loss = outputs.loss
            if self.global_weights is not None and self.prox_mu > 0:
                prox_term = 0.0
                for name, param in model.named_parameters():
                    if name in self.global_weights:
                        prox_term += ((param - self.global_weights[name].to(param.device)) ** 2).sum()
                loss = loss + 0.5 * self.prox_mu * prox_term
            return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir="./tmp",
        per_device_train_batch_size=32,
        num_train_epochs=epochs,
        logging_strategy="no",
        save_strategy="no",
        report_to="none",
    )
    prox_trainer = ProxTrainer(prox_mu, global_weights, model=model, args=args, train_dataset=dataset, tokenizer=tokenizer)
    prox_trainer.train()
    return model

class FedProxTrainer(BaseFederatedTrainer):
    def __init__(self, model_init, tokenizer, label_list, device="cpu", epochs=1, mu=0.01):
        super().__init__(model_init, tokenizer, label_list, device)
        self.epochs = epochs
        self.mu = mu

    def train_round(self, global_model, clients_data):
        global_weights = copy.deepcopy(global_model.state_dict())
        client_models = []

        for i, client_data in enumerate(clients_data):
            client_model = get_client_model(global_model, self.device)
            trained_model = train_local_model(
                client_model,
                self.tokenizer,
                client_data,
                self.label_list,
                self.device,
                self.epochs,
                prox_mu=self.mu,
                global_weights=global_weights
            )
            client_models.append(trained_model.to("cpu"))

        # 聚合模型
        new_global = self.aggregate(client_models)
        return new_global
