import time
from collections import defaultdict

class MetricsLogger:
    def __init__(self, cfg=None):
        self.history = defaultdict(list)
        self.start_time = None
        self.cfg = cfg
        self.total_time = 0.0
        self.total_comm_mb = 0.0

    def start_timer(self):
        self.start_time = time.time()

    def stop_timer(self):
        elapsed = time.time() - self.start_time
        self.total_time += elapsed
        return elapsed

    def log_round_metrics(self, round_idx, metrics, elapsed_time, comm_mb=None):
        self.history["round"].append(round_idx)
        self.history["f1"].append(metrics.get("f1"))
        self.history["precision"].append(metrics.get("precision"))
        self.history["recall"].append(metrics.get("recall"))
        self.history["accuracy"].append(metrics.get("accuracy"))
        self.history["time"].append(elapsed_time)

        if comm_mb is not None:
            self.history["comm_mb"].append(comm_mb)
            self.total_comm_mb += comm_mb
        else:
            self.history["comm_mb"].append(None)

    def log(self, round_idx, metric_name, value):
        key = f"{metric_name}"
        self.history[key].append((round_idx, value))

    def get_logs(self):
        return {
            "history": dict(self.history),
            "total_time": self.total_time,
            "total_comm_mb": self.total_comm_mb
        }
