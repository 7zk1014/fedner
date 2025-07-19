from abc import ABC, abstractmethod
import copy
import torch


class BaseTrainer(ABC):
    @abstractmethod
    def train_round(self, global_model, clients_data):
        pass


class BaseFederatedTrainer(BaseTrainer):
    """Simple base class for federated trainers."""

    def __init__(self, model_init, tokenizer, label_list, device="cpu"):
        self.model_init = model_init
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.label2id = {l: i for i, l in enumerate(label_list)}
        self.device = device

    def aggregate(self, client_models):
        """Average model weights from clients."""
        new_model = copy.deepcopy(client_models[0])
        state_dicts = [m.state_dict() for m in client_models]
        for key in new_model.state_dict().keys():
            avg = torch.mean(
                torch.stack([sd[key] for sd in state_dicts], dim=0), dim=0
            )
            new_model.state_dict()[key].copy_(avg)
        return new_model.to(self.device)

