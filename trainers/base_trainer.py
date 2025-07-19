from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    @abstractmethod
    def train_round(self, global_model, clients_data):
        pass
