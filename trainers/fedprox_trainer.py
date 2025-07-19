import copy
import torch
from trainers.base_trainer import BaseFederatedTrainer
from trainer_utils import get_client_model, train_local_model

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
