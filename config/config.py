import os
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

def _abspath(p: Optional[str]) -> Optional[str]:
    """Expands user home directory and converts a path to an absolute path."""
    if p is None:
        return None
    return os.path.abspath(os.path.expanduser(p))

@dataclass
class Config:
    """
    Configuration class for Federated Learning experiments.
    Parameters are initialized with default values and then overridden
    by values read from a YAML configuration file.
    """
    # ==================== Model Parameters ====================
    model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

    # ==================== Data Parameters ====================
    pubtator_path: Optional[str] = None
    trng_pmids_path: Optional[str] = None
    dev_pmids_path: Optional[str] = None
    test_pmids_path: Optional[str] = None
    partition_strategy: str = "iid"
    noniid_alpha: float = 0.5

    # ==================== Training Parameters ====================
    algorithm: str = "FedAvg"
    num_clients: int = 5
    rounds: int = 10
    local_epochs: int = 1

    # ==================== Training Control Parameters ====================
    # Layer Freezing
    train_last_n_layers: int = 4  # -1 for all layers, 0 for only the classifier head
    freeze_embeddings: bool = True  # Whether to freeze the embedding layers

    # Data Sampling
    sample_size: Optional[int] = 200  # Number of samples per client, None for all data
    sample_strategy: str = "random"  # "random", "balanced", "stratified"
    sample_with_replacement: bool = False  # Sample with replacement

    # Client Selection
    client_fraction: float = 1.0  # Fraction of clients participating per round
    min_clients_per_round: int = 1  # Minimum number of clients per round

    # ==================== Hyperparameters ====================
    learning_rate: float = 5e-5
    lr_scheduler_type: str = "constant"
    train_batch_size: int = 32
    eval_batch_size: int = 64
    max_seq_length: int = 128

    # ==================== Misc Parameters ====================
    seed: int = 42
    device: str = "auto"
    save_results: bool = True
    verbose: bool = True  # Print detailed logs

    # ==================== Algorithm-specific Parameters ====================
    # FedProx
    mu: float = 0.1

    # FedAdam
    server_lr: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    
    # FedSD (Baseline)
    fedsd_alpha_ce: float = 1.0
    fedsd_alpha_kl: float = 1.0
    fedsd_alpha_hidden: float = 0.0
    fedsd_temperature: float = 2.0
    
    # FedSAD (Self-Adaptive Distillation)
    use_distillation: bool = True
    temperature_start: float = 5.0
    temperature_end: float = 1.0
    alpha_ce_start: float = 1.0
    alpha_ce_end: float = 0.3
    alpha_kd_start: float = 0.0
    alpha_kd_end: float = 0.4
    hidden_distill: bool = True
    alpha_hidden: float = 0.1
    adaptive_weighting: bool = True
    momentum_teacher: bool = True
    momentum_alpha: float = 0.99
    compress_gradients: bool = False
    compress_topk: int = 20
    compress_min_dim: int = 100
    # New FedSAD-specific parameters from the first code block
    kd_start_round: int = 2
    teacher_policy: str = "ema_first"
    first_ema_m0: float = 0.98
    ema_m_min: float = 0.98
    ema_m_max: float = 0.999
    o_downweight: float = 0.5


    # Store raw config for debugging or further use
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)
    
    # The `__init__` method will be overridden, so we don't need __post_init__ for these defaults.
    # The new parameters will be loaded directly from the YAML.

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initializes the Config object by loading from a YAML file.
        """
        # Load the YAML file
        if not os.path.exists(config_path):
            # If the file doesn't exist, use dataclass defaults.
            # This is a safe fallback for testing or manual setup.
            return
        
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        self._raw = cfg

        # Map YAML sections to object attributes
        self.model_name = cfg.get("model_name", self.model_name)
        if isinstance(self.model_name, str):
            expanded = os.path.expanduser(self.model_name)
            self.model_name = os.path.abspath(expanded) if os.path.isdir(expanded) else self.model_name

        data = cfg.get("data", {})
        self.pubtator_path = _abspath(data.get("pubtator_path", self.pubtator_path))
        self.trng_pmids_path = _abspath(data.get("trng_pmids_path", self.trng_pmids_path))
        self.dev_pmids_path = _abspath(data.get("dev_pmids_path", self.dev_pmids_path))
        self.test_pmids_path = _abspath(data.get("test_pmids_path", self.test_pmids_path))
        self.partition_strategy = data.get("partition_strategy", self.partition_strategy)
        self.noniid_alpha = float(data.get("noniid_alpha", self.noniid_alpha))

        train = cfg.get("training", {})
        self.algorithm = train.get("algorithm", self.algorithm)
        self.num_clients = int(train.get("num_clients", self.num_clients))
        self.rounds = int(train.get("rounds", self.rounds))
        self.local_epochs = int(train.get("local_epochs", self.local_epochs))

        control = cfg.get("training_control", {})
        self.train_last_n_layers = int(control.get("train_last_n_layers", self.train_last_n_layers))
        self.freeze_embeddings = bool(control.get("freeze_embeddings", self.freeze_embeddings))
        
        sample_size = control.get("sample_size", self.sample_size)
        self.sample_size = int(sample_size) if sample_size is not None else None
        self.sample_strategy = control.get("sample_strategy", self.sample_strategy)
        self.sample_with_replacement = bool(control.get("sample_with_replacement", self.sample_with_replacement))
        self.client_fraction = float(control.get("client_fraction", self.client_fraction))
        self.min_clients_per_round = int(control.get("min_clients_per_round", self.min_clients_per_round))

        hp = cfg.get("hyperparameters", {})
        self.learning_rate = float(hp.get("learning_rate", self.learning_rate))
        self.lr_scheduler_type = hp.get("lr_scheduler_type", self.lr_scheduler_type)
        self.train_batch_size = int(hp.get("train_batch_size", self.train_batch_size))
        self.eval_batch_size = int(hp.get("eval_batch_size", self.eval_batch_size))
        self.max_seq_length = int(hp.get("max_seq_length", self.max_seq_length))

        misc = cfg.get("misc", {})
        self.seed = int(misc.get("seed", self.seed))
        self.device = misc.get("device", self.device)
        self.save_results = bool(misc.get("save_results", self.save_results))
        self.verbose = bool(misc.get("verbose", self.verbose))

        prox = cfg.get("fedprox", {})
        self.mu = float(prox.get("mu", self.mu))

        fad = cfg.get("fedadam", {})
        self.server_lr = float(fad.get("server_lr", self.server_lr))
        self.beta1 = float(fad.get("beta1", self.beta1))
        self.beta2 = float(fad.get("beta2", self.beta2))

        fsd = cfg.get("fedsd", {})
        self.fedsd_alpha_ce = float(fsd.get("alpha_ce", self.fedsd_alpha_ce))
        self.fedsd_alpha_kl = float(fsd.get("alpha_kl", self.fedsd_alpha_kl))
        self.fedsd_alpha_hidden = float(fsd.get("alpha_hidden", self.fedsd_alpha_hidden))
        self.fedsd_temperature = float(fsd.get("temperature", self.fedsd_temperature))
        
        # FedSAD section
        fsad = cfg.get("fedsad", {})
        self.use_distillation = bool(fsad.get("use_distillation", self.use_distillation))
        self.kd_start_round = int(fsad.get("kd_start_round", self.kd_start_round))
        self.teacher_policy = str(fsad.get("teacher_policy", self.teacher_policy))
        self.first_ema_m0 = float(fsad.get("first_ema_m0", self.first_ema_m0))
        self.ema_m_min = float(fsad.get("ema_m_min", self.ema_m_min))
        self.ema_m_max = float(fsad.get("ema_m_max", self.ema_m_max))
        self.temperature_start = float(fsad.get("temperature_start", self.temperature_start))
        self.temperature_end = float(fsad.get("temperature_end", self.temperature_end))
        self.alpha_ce_start = float(fsad.get("alpha_ce_start", self.alpha_ce_start))
        self.alpha_ce_end = float(fsad.get("alpha_ce_end", self.alpha_ce_end))
        self.alpha_kd_start = float(fsad.get("alpha_kd_start", self.alpha_kd_start))
        self.alpha_kd_end = float(fsad.get("alpha_kd_end", self.alpha_kd_end))
        self.hidden_distill = bool(fsad.get("hidden_distill", self.hidden_distill))
        self.alpha_hidden = float(fsad.get("alpha_hidden", self.alpha_hidden))
        self.o_downweight = float(fsad.get("o_downweight", self.o_downweight))
        self.compress_gradients = bool(fsad.get("compress_gradients", self.compress_gradients))
        self.compress_topk = int(fsad.get("compress_topk", self.compress_topk))
        self.compress_min_dim = int(fsad.get("compress_min_dim", self.compress_min_dim))
        
        # The following parameters were present in the previous `fedkd` section of the code,
        # but are not in the new `fedsad_trainer.py` code. We'll retain them here
        # but with a note that they may not be used by the new trainer.
        self.adaptive_weighting = bool(fsad.get("adaptive_weighting", self.adaptive_weighting))
        self.momentum_teacher = bool(fsad.get("momentum_teacher", self.momentum_teacher))
        self.momentum_alpha = float(fsad.get("momentum_alpha", self.momentum_alpha))

        self._validate()

    def apply_overrides(self, **kwargs):
        """Applies command-line arguments to override configuration."""
        for k, v in kwargs.items():
            if v is None:
                continue
            if not hasattr(self, k):
                raise AttributeError(f"Unknown config key: {k}")
            
            if k == "sample_size":
                if v == "all" or v == "none":
                    setattr(self, k, None)
                else:
                    setattr(self, k, int(v))
            else:
                setattr(self, k, type(getattr(self, k))(v))
        
        self._validate()

    def _validate(self):
        """Validates the configuration parameters."""
        if self.partition_strategy not in {"iid", "noniid"}:
            raise ValueError("partition_strategy must be 'iid' or 'noniid'")
        
        if self.num_clients <= 0 or self.rounds <= 0 or self.local_epochs <= 0:
            raise ValueError("num_clients/rounds/local_epochs must > 0")
        
        if not (0.0 < self.learning_rate):
            raise ValueError("learning_rate must be > 0")
        
        if self.train_last_n_layers < -1:
            raise ValueError("train_last_n_layers must be >= -1 (-1 means train all layers)")
        
        if self.sample_size is not None and self.sample_size <= 0:
            raise ValueError("sample_size must be > 0 or None")
        
        if self.sample_strategy not in {"random", "balanced", "stratified"}:
            raise ValueError("sample_strategy must be one of: random, balanced, stratified")
        
        if not (0.0 < self.client_fraction <= 1.0):
            raise ValueError("client_fraction must be in (0, 1]")
        
        if self.min_clients_per_round < 1:
            raise ValueError("min_clients_per_round must be >= 1")
        
        if self.min_clients_per_round > self.num_clients:
            raise ValueError("min_clients_per_round cannot exceed num_clients")

    def get_training_info(self) -> str:
        """Returns a formatted string with key training information."""
        info = []
        info.append(f"Algorithm: {self.algorithm}")
        info.append(f"Clients: {self.num_clients} ({self.client_fraction*100:.0f}% per round)")
        info.append(f"Rounds: {self.rounds}, Local epochs: {self.local_epochs}")
        
        if self.train_last_n_layers == -1:
            info.append("Training: All layers")
        elif self.train_last_n_layers == 0:
            info.append("Training: Only classifier head")
        else:
            info.append(f"Training: Last {self.train_last_n_layers} layers")
        
        if self.sample_size is None:
            info.append("Sampling: Using all local data")
        else:
            info.append(f"Sampling: {self.sample_size} samples per client ({self.sample_strategy})")
        
        if self.partition_strategy == "noniid":
            info.append(f"Data: Non-IID (α={self.noniid_alpha})")
        else:
            info.append("Data: IID")
        
        return " | ".join(info)

    def as_dict(self) -> Dict[str, Any]:
        """Returns the full configuration as a serializable dictionary."""
        return {
            "model_name": self.model_name,
            "data": {
                "pubtator_path": self.pubtator_path,
                "trng_pmids_path": self.trng_pmids_path,
                "dev_pmids_path": self.dev_pmids_path,
                "test_pmids_path": self.test_pmids_path,
                "partition_strategy": self.partition_strategy,
                "noniid_alpha": self.noniid_alpha,
            },
            "training": {
                "algorithm": self.algorithm,
                "num_clients": self.num_clients,
                "rounds": self.rounds,
                "local_epochs": self.local_epochs,
            },
            "training_control": {
                "train_last_n_layers": self.train_last_n_layers,
                "freeze_embeddings": self.freeze_embeddings,
                "sample_size": self.sample_size,
                "sample_strategy": self.sample_strategy,
                "sample_with_replacement": self.sample_with_replacement,
                "client_fraction": self.client_fraction,
                "min_clients_per_round": self.min_clients_per_round,
            },
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "lr_scheduler_type": self.lr_scheduler_type,
                "train_batch_size": self.train_batch_size,
                "eval_batch_size": self.eval_batch_size,
                "max_seq_length": self.max_seq_length,
            },
            "misc": {
                "seed": self.seed,
                "device": self.device,
                "save_results": self.save_results,
                "verbose": self.verbose,
            },
            "fedprox": {
                "mu": self.mu,
            },
            "fedadam": {
                "server_lr": self.server_lr,
                "beta1": self.beta1,
                "beta2": self.beta2,
            },
            "fedsd": {
                "alpha_ce": self.fedsd_alpha_ce,
                "alpha_kl": self.fedsd_alpha_kl,
                "alpha_hidden": self.fedsd_alpha_hidden,
                "temperature": self.fedsd_temperature,
            },
            "fedsad": {
                "use_distillation": self.use_distillation,
                "temperature_start": self.temperature_start,
                "temperature_end": self.temperature_end,
                "alpha_ce_start": self.alpha_ce_start,
                "alpha_ce_end": self.alpha_ce_end,
                "alpha_kd_start": self.alpha_kd_start,
                "alpha_kd_end": self.alpha_kd_end,
                "hidden_distill": self.hidden_distill,
                "alpha_hidden": self.alpha_hidden,
                "adaptive_weighting": self.adaptive_weighting,
                "momentum_teacher": self.momentum_teacher,
                "momentum_alpha": self.momentum_alpha,
                "compress_gradients": self.compress_gradients,
                "compress_topk": self.compress_topk,
                "compress_min_dim": self.compress_min_dim,
                "kd_start_round": self.kd_start_round,
                "teacher_policy": self.teacher_policy,
                "first_ema_m0": self.first_ema_m0,
                "ema_m_min": self.ema_m_min,
                "ema_m_max": self.ema_m_max,
                "o_downweight": self.o_downweight,
            },
        }