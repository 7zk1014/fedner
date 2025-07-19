# ⚠️ FedKD Trainer（联邦蒸馏）暂未实现
# 通常基于 soft logits（教师模型输出）进行客户端知识对齐
# 若需实现，请在本地评估蒸馏 loss（如 KL 散度），聚合 soft label 或 logits 平均

class FedKDTrainer:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("FedKDTrainer is not yet implemented.")