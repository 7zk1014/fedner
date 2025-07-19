import copy
import torch
from trainers.base_trainer import BaseTrainer
from utils.evaluate import evaluate_model

class FedAdamTrainer(BaseTrainer):
    def __init__(self, model, tokenizer, label_list, train_data, dev_data, test_data, args):
        super().__init__(model, tokenizer, label_list, train_data, dev_data, test_data, args)
        self.global_model = copy.deepcopy(model)
        self.momentum = {name: torch.zeros_like(param.data) for name, param in self.global_model.named_parameters() if param.requires_grad}
        self.v = {name: torch.zeros_like(param.data) for name, param in self.global_model.named_parameters() if param.requires_grad}
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.epsilon = 1e-8

    def client_update(self, model, data):
        # 用 BaseTrainer 中的方法本地训练
        return self.train_locally(model, data)

    def aggregate(self, client_models):
        # 基于 FedAdam 方式聚合（模仿 Adam 动量）
        with torch.no_grad():
            new_state = copy.deepcopy(self.global_model.state_dict())
            delta = {}

            # 计算权重更新方向（梯度估计）
            for name in new_state:
                client_tensors = [m.state_dict()[name].cpu() for m in client_models]
                stacked = torch.stack(client_tensors)
                mean_diff = torch.mean(stacked - self.global_model.state_dict()[name].cpu(), dim=0)
                delta[name] = mean_diff

            # Adam-style update
            for name in new_state:
                self.momentum[name] = self.beta1 * self.momentum[name] + (1 - self.beta1) * delta[name]
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * delta[name]**2
                update = self.momentum[name] / (self.v[name].sqrt() + self.epsilon)
                new_state[name] = self.global_model.state_dict()[name].cpu() + self.args.get("server_lr", 1e-2) * update

            self.global_model.load_state_dict(new_state)

        return self.global_model