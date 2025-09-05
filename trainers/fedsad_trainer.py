import copy
import random
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from utils.streaming import compress_state_dict_diff
from trainers.base_trainer import BaseFederatedTrainer
from utils.training_utils import freeze_model_layers


class FedSADTrainer(BaseFederatedTrainer):
    """
    FedSAD: Improved FedKD (no adaptive weight), with unified KD start & teacher policy.

    Key:
    - kd_start_round (1-based): which global round to start KD.
    - teacher_policy in {"ema_first", "cold_then_ema", "prev_global"}:
        * "ema_first": first KD round teacher = EMA(G0, G_{t-1}) with m0=first_ema_m0; then EMA each round.
        * "cold_then_ema": first KD round teacher = G_{t-1}; then EMA each round.
        * "prev_global": every KD round teacher = G_{t-1} (no EMA at all).
    - Fixed KD composition (no double alpha):
        loss = alpha_ce * CE + alpha_kd * (T^2 * KL) + alpha_hidden * hid_MSE
    - Better masking: ignore -100/specials; downweight class "O".
    - Fairness unchanged: same backbone, train-last-4, sample_size=200.
    """

    def __init__(self, model_init, tokenizer, label_list, device,
                 epochs=2, batch_size=32, learning_rate=5e-5,
                 scheduler_type="constant", rounds=20,
                 # ====== schedules (rebased after KD starts) ======
                 use_distillation=True,
                 kd_start_round: int = 2,           # Which round to start knowledge distillation (1-based)
                 temperature_start=5.0,
                 temperature_end=1.5,
                 alpha_ce_start=1.0,
                 alpha_ce_end=0.7,
                 alpha_kd_start=0.0,
                 alpha_kd_end=0.3,
                 # ====== teacher policy ======
                 teacher_policy: str = "ema_first",    # {"ema_first","cold_then_ema","prev_global"}
                 first_ema_m0: float = 0.6,            # EMA momentum for first KD round (ema_first policy only)
                 ema_m_min: float = 0.6,              # Minimum EMA momentum for subsequent KD rounds
                 ema_m_max: float = 0.6,              # Maximum EMA momentum for subsequent KD rounds
                 # ====== hidden-state KD (optional) ======
                 hidden_distill: bool = False,
                 alpha_hidden: float = 0.0,
                 # KD mask: O downweight
                 o_downweight: float = 0.0,
                 # compression (optional)
                 compress_gradients: bool = False,
                 compress_topk: int = 0,
                 compress_min_dim: int = 100,
                 # fairness constants
                 train_last_n: int = 4,
                 sample_strategy: str = "random",
                 sample_size: int = 200,
                 max_seq_length: int = 128):

        super().__init__(model_init, tokenizer, label_list, device)

        # basic
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.scheduler_type = scheduler_type
        self.rounds = int(rounds) if rounds is not None else 20
        self.use_distillation = use_distillation

        # KD start & schedules
        self.kd_start_round = int(kd_start_round)      # 1-based round indexing
        self.temperature_start = float(temperature_start)
        self.temperature_end = float(temperature_end)
        self.alpha_ce_start = float(alpha_ce_start)
        self.alpha_ce_end = float(alpha_ce_end)
        self.alpha_kd_start = float(alpha_kd_start)
        self.alpha_kd_end = float(alpha_kd_end)

        # teacher policy
        assert teacher_policy in {"ema_first", "cold_then_ema", "prev_global"}
        self.teacher_policy = teacher_policy
        self.first_ema_m0 = float(first_ema_m0)
        self.ema_m_min = float(ema_m_min)
        self.ema_m_max = float(ema_m_max)

        # hidden KD
        self.hidden_distill = bool(hidden_distill)
        self.alpha_hidden = float(alpha_hidden)

        # mask
        self.o_downweight = float(o_downweight)

        # compression
        self.compress_gradients = bool(compress_gradients)
        self.compress_topk = int(compress_topk)
        self.compress_min_dim = int(compress_min_dim)

        # fairness constants
        self.train_last_n_layers = train_last_n
        self.sample_size = sample_size
        self.max_seq_length = max_seq_length

        # labels
        self.label2id = {l: i for i, l in enumerate(label_list)}
        self.o_id = self.label2id.get("O", None)

        # states
        self.round_idx = 0
        self.teacher_model = None
        self.init_global_snapshot = None  # Store initial global model (G0)

    # --------- KD schedules (rebased after KD starts) ----------
    def _progress_after_kd(self):
        # Progress ratio [0,1] counted from when KD starts
        t = max(0, self.round_idx - self.kd_start_round)
        denom = max(1, self.rounds - self.kd_start_round)
        return min(1.0, t / denom)

    def get_current_hyperparams(self):
        """KD-aware parameter scheduling: no KD before start round, then linear annealing."""
        if not self.use_distillation or self.round_idx < self.kd_start_round:
            return self.temperature_start, self.alpha_ce_start, 0.0

        p = self._progress_after_kd()  # Progress since KD started
        T  = self.temperature_start - (self.temperature_start - self.temperature_end) * p
        a_ce = self.alpha_ce_start    - (self.alpha_ce_start    - self.alpha_ce_end) * p
        a_kd = self.alpha_kd_start    + (self.alpha_kd_end      - self.alpha_kd_start) * p
        return T, a_ce, a_kd

    # --------- KD core & hidden KD (no alpha inside) ----------
    def _kd_and_hidden_core(self, student_logits, teacher_logits, attention_mask,
                             labels=None, student_hidden=None, teacher_hidden=None):
        T, _, _ = self.get_current_hyperparams()

        # Knowledge distillation on logits (maintains gradients for student)
        s_log = F.log_softmax(student_logits / T, dim=-1)
        t_soft = F.softmax(teacher_logits / T, dim=-1)
        kl_per_tok = F.kl_div(s_log, t_soft, reduction='none').sum(dim=-1)  # [B,L]

        # Create validity mask for tokens
        valid = attention_mask.float()
        if labels is not None:
            valid = valid * (labels != -100).float()
            if self.o_id is not None and 0.0 <= self.o_downweight < 1.0:
                is_o = (labels == self.o_id).float()
                valid = valid * ((1.0 - is_o) + self.o_downweight * is_o)

        denom = valid.sum().clamp_min(1e-8)
        kd_core = (T ** 2) * (kl_per_tok * valid).sum() / denom

        hid_raw = None
        if (self.hidden_distill and self.alpha_hidden > 0
                and student_hidden is not None and teacher_hidden is not None):
            s_last = student_hidden[-1]    # [B,L,H]
            t_last = teacher_hidden[-1]
            mse_tok = F.mse_loss(s_last, t_last, reduction='none').mean(dim=-1)  # [B,L]
            hid_raw = (mse_tok * valid).sum() / denom
        return kd_core, hid_raw

    # --------- Local training ----------
    def train_local_model(self, global_model, train_examples, client_id):
        sampled = random.sample(train_examples, min(self.sample_size, len(train_examples)))

        student_model = copy.deepcopy(global_model).to(self.device)
        freeze_model_layers(student_model, train_last_n_layers=self.train_last_n_layers)
        student_model.train()

        # Check if knowledge distillation is active
        kd_on = self.use_distillation and (self.round_idx >= self.kd_start_round) and (self.teacher_model is not None)
        teacher_model = None
        if kd_on:
            teacher_model = copy.deepcopy(self.teacher_model).to(self.device)
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad = False

        dataloader = self.prepare_dataloader(sampled)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, student_model.parameters()),
            lr=self.learning_rate
        )
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs * len(dataloader)
        ) if self.scheduler_type == "cosine" else None)

        total_loss = 0.0
        T, a_ce, a_kd = self.get_current_hyperparams()

        for _ in range(self.epochs):
            for input_ids, attention_mask, token_type_ids, labels in dataloader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                labels = labels.to(self.device)

                # student forward (ask hidden only if needed)
                student_out = student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                    output_hidden_states=(self.hidden_distill and kd_on)
                )

                loss = a_ce * student_out.loss

                if kd_on:
                    # teacher forward without grad
                    with torch.no_grad():
                        teacher_out = teacher_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            output_hidden_states=(self.hidden_distill and self.alpha_hidden > 0)
                        )
                    # KD/hidden computed WITH grad (for student)
                    kd_core, hid_raw = self._kd_and_hidden_core(
                        student_logits=student_out.logits,
                        teacher_logits=teacher_out.logits,
                        attention_mask=attention_mask,
                        labels=labels,
                        student_hidden=(student_out.hidden_states if self.hidden_distill else None),
                        teacher_hidden=(teacher_out.hidden_states if self.hidden_distill else None)
                    )
                    loss = loss + a_kd * kd_core
                    if (hid_raw is not None) and (self.alpha_hidden > 0):
                        loss = loss + self.alpha_hidden * hid_raw

                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, student_model.parameters()), 1.0)
                optimizer.step()
                if scheduler: scheduler.step()
                total_loss += loss.item()

        student_state = student_model.state_dict()
        if self.compress_gradients and self.compress_topk > 0:
            student_state = compress_state_dict_diff(global_model.state_dict(), student_state,
                                                     topk=self.compress_topk, min_dim=self.compress_min_dim)
        avg_loss = total_loss / max(1, (self.epochs * len(dataloader)))
        return student_state, len(sampled), avg_loss

    # --------- DataLoader ----------
    def prepare_dataloader(self, train_examples):
        texts = [e["tokens"] for e in train_examples]
        label_seqs = [e["labels"] for e in train_examples]

        enc = self.tokenizer(
            texts,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )

        all_label_ids = []
        for i in range(len(texts)):
            word_ids = enc.word_ids(batch_index=i)
            lab_ids = []
            for wi in word_ids:
                if wi is None:
                    lab_ids.append(-100)
                else:
                    if wi < len(label_seqs[i]):
                        lab_ids.append(self.label2id[label_seqs[i][wi]])
                    else:
                        lab_ids.append(-100)
            all_label_ids.append(lab_ids)

        labels_tensor = torch.tensor(all_label_ids, dtype=torch.long)
        dataset = TensorDataset(
            enc["input_ids"],
            enc["attention_mask"],
            enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])),
            labels_tensor
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    # --------- Teacher update (unified) ----------
    def _blend_models_(self, dst_model, src_model, m: float):
        """dst := m*dst + (1-m)*src (in-place)"""
        for dp, sp in zip(dst_model.parameters(), src_model.parameters()):
            dp.data.mul_(m).add_(sp.data, alpha=1 - m)

    def _update_teacher_policy(self, global_model):
        """
        Called at the BEGINNING of each round t.
        global_model at this moment is G_{t-1}.
        """
        if not self.use_distillation:
            self.teacher_model = None
            return

        # Save initial global model snapshot once
        if self.init_global_snapshot is None:
            self.init_global_snapshot = copy.deepcopy(global_model)  # G0

        if self.round_idx < self.kd_start_round:
            self.teacher_model = None
            return

        # Handle first knowledge distillation round
        if self.round_idx == self.kd_start_round:
            if self.teacher_policy == "ema_first":
                # Blend initial model with previous global model
                self.teacher_model = copy.deepcopy(self.init_global_snapshot)  # start from G0
                self._blend_models_(self.teacher_model, global_model, m=self.first_ema_m0)
            elif self.teacher_policy == "cold_then_ema":
                # Use previous global model as teacher
                self.teacher_model = copy.deepcopy(global_model)
            elif self.teacher_policy == "prev_global":
                # Always use previous global model (no EMA)
                self.teacher_model = copy.deepcopy(global_model)
            return

        # Handle subsequent knowledge distillation rounds
        if self.teacher_policy == "prev_global":
            # Always use previous global model, no EMA
            self.teacher_model = copy.deepcopy(global_model)
        else:
            # Apply EMA with annealed momentum
            p = self._progress_after_kd()  # Progress since KD started
            m = self.ema_m_max - (self.ema_m_max - self.ema_m_min) * p
            if self.teacher_model is None:
                self.teacher_model = copy.deepcopy(global_model)
            else:
                self._blend_models_(self.teacher_model, global_model, m=m)

        # Freeze teacher model parameters
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

    # --------- One federated round ----------
    def train_round(self, global_model, clients_data):
        self.round_idx += 1

        # Update teacher model at start of round
        self._update_teacher_policy(global_model)
        kd_on = self.use_distillation and (self.round_idx >= self.kd_start_round)
        print(f"[Round {self.round_idx}] KD={'ON' if kd_on else 'OFF'} | policy={self.teacher_policy}")

        # Local client training
        client_weights, client_samples = [], []
        for cid, cdata in enumerate(clients_data):
            sd, ns, _ = self.train_local_model(global_model, cdata, cid)
            client_weights.append(sd); client_samples.append(ns)

        # Aggregate client models
        new_global = self.weighted_aggregate(client_weights, client_samples)
        global_model.load_state_dict(new_global)

        T, a_ce, a_kd = self.get_current_hyperparams()
        print(f"[Round {self.round_idx}] T={T:.2f}  α_ce={a_ce:.2f}  α_kd={a_kd:.2f}")
        return global_model

    # --------- FedAvg weighted aggregate ----------
    def weighted_aggregate(self, client_weights, client_samples):
        total = sum(client_samples)
        avg = copy.deepcopy(client_weights[0])
        for k in avg.keys():
            avg[k] = torch.zeros_like(avg[k])
        for w, n in zip(client_weights, client_samples):
            coef = n / max(1, total)
            for k in avg.keys():
                if isinstance(w[k], torch.Tensor):
                    avg[k] += w[k].to(avg[k].device) * coef
        return avg

    # --------- (optional) comms estimate ----------
    def _estimate_communication_cost(self, client_weights):
        total_params = 0
        for sd in client_weights:
            for v in sd.values():
                if isinstance(v, torch.Tensor):
                    total_params += v.numel()
        return total_params * 4 / (1024 ** 2)  # Convert bytes to MB
