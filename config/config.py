import yaml

class Config:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.model_name = cfg["model_name"]

        self.train_path = cfg["data"]["train_path"]
        self.dev_path = cfg["data"]["dev_path"]
        self.test_path = cfg["data"]["test_path"]

        self.algorithm = cfg["training"]["algorithm"]
        self.num_clients = cfg["training"]["num_clients"]
        self.rounds = cfg["training"]["rounds"]
        self.local_epochs = cfg["training"]["local_epochs"]

        self.learning_rate = float(cfg["hyperparameters"]["learning_rate"])
        self.lr_scheduler_type = cfg["hyperparameters"].get("lr_scheduler_type", "linear")
        self.train_batch_size = cfg["hyperparameters"]["train_batch_size"]
        self.eval_batch_size = cfg["hyperparameters"]["eval_batch_size"]
        self.max_seq_length = cfg["hyperparameters"]["max_seq_length"]
        
        self.seed = cfg["misc"].get("seed", 42)
        self.device_mode = cfg["misc"].get("device", "auto")
        self.save_results = cfg["misc"].get("save_results", True)
