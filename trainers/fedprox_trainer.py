import copy
import torch
import random
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from trainers.base_trainer import BaseFederatedTrainer
from utils.evaluate import align_labels_with_tokens

def freeze_bert_layers(model, trainable_layers=4):
    total_layers = 12
    freeze_up_to = total_layers - trainable_layers
    for name, param in model.named_parameters():
        if name.startswith("bert.encoder.layer."):
            layer_num = int(name.split(".")[3])
            if layer_num < freeze_up_to:
                param.requires_grad = False
        elif name.startswith("bert.embeddings.") or name.startswith("bert.pooler."):
            param.requires_grad = False

def subsample_data(examples, sample_size=200):
    return random.sample(examples, min(sample_size, len(examples)))

def get_client_model(global_model, device):
    return copy.deepcopy(global_model).to(device)

def train_local_model(
    model, tokenizer, train_examples, label_list, device,
    epochs, batch_size, learning_rate, scheduler_type,
    prox_mu, global_weights, trainable_layers=2, sample_size=200,
    max_seq_length=128
):
    label2id = {l: i for i, l in enumerate(label_list)}
    sampled_data = subsample_data(train_examples, sample_size)
    freeze_bert_layers(model, trainable_layers)

    def preprocess(example):
        tokenized = tokenizer(
            example["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=max_seq_length,
        )
        tokenized["labels"] = align_labels_with_tokens(tokenized, [example["labels"]], label2id)[0]
        return tokenized

    dataset = Dataset.from_list(sampled_data).map(preprocess, batched=False)

    class ProxTrainer(Trainer):
        def __init__(self, prox_mu, global_weights, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.prox_mu = prox_mu
            self.global_weights = {k: v.clone().detach() for k, v in global_weights.items()}
            self.global_weights.update({f"module.{k}": v.clone().detach() for k, v in global_weights.items()})

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            loss = outputs.loss
            if self.global_weights and self.prox_mu > 0:
                device = next(model.parameters()).device
                prox_term = torch.tensor(0.0, device=device)
                for name, param in model.named_parameters():
                    weight = self.global_weights.get(name)
                    if weight is not None:
                        prox_term += ((param - weight.to(param.device)) ** 2).sum()
                loss += 0.5 * self.prox_mu * prox_term
                # print(f"[FedProx] prox_term = {prox_term.item():.6f}")
            return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        lr_scheduler_type=scheduler_type,
        logging_strategy="no",
        save_strategy="no",
        report_to="none",
        fp16=True
    )

    trainer = ProxTrainer(
        prox_mu=prox_mu,
        global_weights=global_weights,
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )
    trainer.train()
    return model

class FedProxTrainer(BaseFederatedTrainer):
    def __init__(self, model_init, tokenizer, label_list, device="cpu",
                 epochs=1, mu=0.1, batch_size=32, learning_rate=3e-5,
                 scheduler_type="constant", trainable_layers=4, sample_size=200,
                 max_seq_length=128):
        super().__init__(model_init, tokenizer, label_list, device)
        self.epochs = epochs
        self.mu = mu
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.scheduler_type = scheduler_type
        self.trainable_layers = trainable_layers
        self.sample_size = sample_size
        self.max_seq_length = max_seq_length

    def train_round(self, global_model, clients_data):
        global_weights = global_model.state_dict()
        client_models = []
        for client_data in clients_data:
            client_model = get_client_model(global_model, self.device)
            trained = train_local_model(
                model=client_model,
                tokenizer=self.tokenizer,
                train_examples=client_data,
                label_list=self.label_list,
                device=self.device,
                epochs=self.epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                scheduler_type=self.scheduler_type,
                prox_mu=self.mu,
                global_weights=global_weights,
                trainable_layers=self.trainable_layers,
                sample_size=self.sample_size,
                max_seq_length=self.max_seq_length
            )
            client_models.append(trained.cpu())
        return self.aggregate(client_models)

