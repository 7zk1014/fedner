import yaml

class Config:
    def __init__(self, config_path="config/config.yaml"):
        # 读取 YAML 配置
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # 模型名称
        self.model_name = cfg.get("model_name")

        # 数据路径 (PubTator 格式)
        data_cfg = cfg.get("data", {})
        self.pubtator_path   = data_cfg.get("pubtator_path")
        self.trng_pmids_path = data_cfg.get("trng_pmids_path")
        self.dev_pmids_path  = data_cfg.get("dev_pmids_path")
        self.test_pmids_path = data_cfg.get("test_pmids_path")

        # 数据切分策略
        split_cfg = cfg.get("data_split", {})
        self.partition_strategy = split_cfg.get("partition_strategy", "iid")
        self.noniid_alpha       = float(split_cfg.get("noniid_alpha", 0.5))

        # 联邦学习设置
        federated_cfg = cfg.get("training", {})
        self.algorithm    = federated_cfg.get("algorithm")
        self.num_clients  = federated_cfg.get("num_clients")
        self.rounds       = federated_cfg.get("rounds")
        self.local_epochs = federated_cfg.get("local_epochs")

        # 超参数
        hp = cfg.get("hyperparameters", {})
        self.learning_rate     = float(hp.get("learning_rate", 0.0))
        self.lr_scheduler_type = hp.get("lr_scheduler_type", "linear")
        self.train_batch_size  = hp.get("train_batch_size")
        self.eval_batch_size   = hp.get("eval_batch_size")
        self.max_seq_length    = hp.get("max_seq_length")

        # 其他设置
        misc = cfg.get("misc", {})
        self.seed         = misc.get("seed", 42)
        self.device_mode  = misc.get("device", "auto")
        self.save_results = misc.get("save_results", True)
