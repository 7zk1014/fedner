import torch
import copy
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from utils.evaluate import align_labels_with_tokens

class FedAvgTrainer:
    def __init__(self, model_init, tokenizer, label_list, device="cpu", epochs=1):
        self.model_init = model_init
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.label2id = {l: i for i, l in enumerate(label_list)}
        self.device = device
        self.epochs = epochs

    def preprocess(self, examples):
        def _preprocess(example):
            tokenized = self.tokenizer(example["tokens"],
                                       truncation=True,
                                       is_split_into_words=True,
                                       padding="max_length",
                                       max_length=128)
            tokenized["labels"] = align_labels_with_tokens(tokenized, [example["labels"]], self.label2id)[0]
            return tokenized
        dataset = Dataset.from_list(examples)
        return dataset.map(_preprocess)

    def train_on_client(self, model, train_examples):
        train_dataset = self.preprocess(train_examples)
        args = TrainingArguments(
            output_dir="./tmp",
            per_device_train_batch_size=32,
            num_train_epochs=self.epochs,
            logging_strategy="no",
            save_strategy="no",
            report_to="none"
        )
        trainer = Trainer(model=model, args=args, train_dataset=train_dataset, tokenizer=self.tokenizer)
        trainer.train()
        return model

    def average_models(self, models):
        new_model = copy.deepcopy(models[0])
        state_dicts = [m.state_dict() for m in models]
        for key in new_model.state_dict().keys():
            avg = torch.mean(torch.stack([sd[key] for sd in state_dicts], dim=0), dim=0)
            new_model.state_dict()[key].copy_(avg)
        return new_model

    def train_round(self, global_model, clients_data):
        client_models = []
        for data in clients_data:
            model = self.model_init().to(self.device)
            model.load_state_dict(copy.deepcopy(global_model.state_dict()))
            model = self.train_on_client(model, data)
            client_models.append(model.cpu())
        global_model = self.average_models(client_models).to(self.device)
        return global_model
