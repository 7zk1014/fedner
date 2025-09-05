"""
FedSD (baseline, no hidden-layer distillation)

What changed vs your snippet:
1) Turned OFF hidden-layer distillation completely (alpha_hidden=0 and no hidden states requested).
2) Fixed KD to token-level with proper masking: mask out positions where labels==-100
   (special tokens / padding / subword alignment gaps) in addition to attention_mask.
   -> KL is averaged over valid tokens only; uses the standard T^2 scaling.

Fairness constraints unchanged: same backbone, train_last_n=4, sample_size=200, same optimizer etc.
"""

import copy
import random
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from utils.streaming import compress_state_dict_diff
from trainers.base_trainer import BaseFederatedTrainer
from utils.training_utils import (
    freeze_model_layers, 
    sample_client_data, 
    print_data_statistics,
    get_trainable_params_info
)


class FedSDTrainer(BaseFederatedTrainer):
    def __init__(self, model_init, tokenizer, label_list, device,
                 epochs=2, batch_size=32, learning_rate=5e-5,
                 scheduler_type="constant",
                 alpha_ce=1.0, alpha_kl=1.0, alpha_hidden=0.0,   # Force hidden KD off
                 temperature=2.0,
                 compress_topk=0, compress_min_dim=50,
                 train_last_n: int = 4,
                 max_seq_length=128,
                 sample_strategy: str = "random",
                 sample_size: int = 200):
        super().__init__(model_init=model_init,
                         tokenizer=tokenizer,
                         label_list=label_list)
        self.device           = device
        self.epochs           = epochs
        self.batch_size       = batch_size
        self.learning_rate    = learning_rate
        self.scheduler_type   = scheduler_type
        self.alpha_ce         = alpha_ce
        self.alpha_kl         = alpha_kl
        self.alpha_hidden     = 0.0            # Disable hidden layer distillation
        self.temperature      = temperature
        self.compress_topk    = compress_topk
        self.compress_min_dim = compress_min_dim
        self.train_last_n     = train_last_n
        self.max_seq_length   = max_seq_length
        self.sample_size      = sample_size
        self.label2id         = {label: i for i, label in enumerate(label_list)}
        
    def train_round(self, global_model, clients_data, round_idx=1):
        client_weights = []
        total_num_samples = 0

        for i, client_data in enumerate(clients_data):
            use_kd = round_idx > 1  # No KD in first round
            state_dict, num_samples, _loss = self.train_local_model(
                global_model, client_data, use_kd, round_idx
            )
            client_weights.append(state_dict)
            total_num_samples += num_samples

        new_global_state = self.aggregate(client_weights)
        global_model.load_state_dict(new_global_state)
        return global_model

    def train_local_model(self, global_model, train_examples, use_kd=True, round_idx=1):
        print(f"Original training examples: {len(train_examples)}")
        
        # Client data sampling (consistent with FedAvg)
        sampled = sample_client_data(
            train_examples, 
            sample_size=self.sample_size,
            strategy="random",
        )

        # Teacher model (previous global model, no labels/hidden states needed)
        if use_kd:
            teacher_model = copy.deepcopy(global_model).to(self.device)
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad = False
        else:
            teacher_model = None

        # Student model (train only last N layers)
        student_model = copy.deepcopy(global_model).to(self.device)
        freeze_model_layers(
            student_model, 
            train_last_n_layers=self.train_last_n, 
            freeze_embeddings=True
        )
        student_model.train()

        # Data encoding
        texts = [e["tokens"] for e in sampled]
        label_seqs = [e["labels"] for e in sampled]

        enc = self.tokenizer(
            texts,
            is_split_into_words=True,
            truncation=True,
            padding=True,
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
                    lab_ids.append(self.label2id[label_seqs[i][wi]])
            all_label_ids.append(lab_ids)
        labels_tensor = torch.tensor(all_label_ids, dtype=torch.long)

        dataset = TensorDataset(
            enc["input_ids"],
            enc["attention_mask"],
            enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])),
            labels_tensor
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, student_model.parameters()),
            lr=self.learning_rate
        )

        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            
            for input_ids, attention_mask, token_type_ids, labels in dataloader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                labels = labels.to(self.device)

                # Student forward pass (no hidden states since hidden distillation is off)
                student_out = student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                    output_hidden_states=False
                )

                # Cross-entropy loss (using HF built-in loss, ignores -100 labels)
                loss_ce = student_out.loss

                # Knowledge distillation (only when teacher exists and KD enabled)
                loss_kl = 0.0
                if use_kd and teacher_model is not None:
                    with torch.no_grad():
                        teacher_out = teacher_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            output_hidden_states=False
                        )

                    # Token-level KL divergence with proper masking
                    T = self.temperature
                    s = F.log_softmax(student_out.logits / T, dim=-1)   # [B,L,C]
                    t = F.softmax(teacher_out.logits / T, dim=-1)       # [B,L,C]
                    kl_per_tok = F.kl_div(s, t, reduction="none").sum(dim=-1)  # [B,L]

                    valid = attention_mask.float() * (labels != -100).float()
                    denom = valid.sum().clamp_min(1e-8)
                    kl = (kl_per_tok * valid).sum() / denom
                    loss_kl = (T * T) * kl

                # Total loss (hidden distillation disabled)
                loss = self.alpha_ce * loss_ce + self.alpha_kl * loss_kl

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                epoch_loss += loss.item()
                num_batches += 1
                epoch_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Get final state dictionary
        student_sd = student_model.state_dict()
        
        # Optional compression if enabled
        original_size = self._estimate_size(student_sd)
        if self.compress_topk > 0:
            print(f" Applying compression with topk={self.compress_topk}")
            global_sd = global_model.state_dict()
            student_sd = compress_state_dict_diff(
                global_sd, student_sd,
                topk=self.compress_topk,
                min_dim=self.compress_min_dim
            )
            compressed_size = self._estimate_size(student_sd)
            print(f" Compression: {original_size:.2f}MB → {compressed_size:.2f}MB "
                  f"({compressed_size/original_size*100:.1f}%)")
        else:
            print(f" No compression applied (topk={self.compress_topk})")
            print(f" Model size: {original_size:.2f}MB (uncompressed)")

        return student_sd, len(sampled), avg_loss

    def aggregate(self, weights_list):
        """Aggregate client weights (simple averaging; can be changed to sample-weighted for standard FedAvg)."""
        avg = copy.deepcopy(weights_list[0])
        for k in avg:
            for w in weights_list[1:]:
                avg[k] += w[k]
            avg[k] /= len(weights_list)

        # Upload cost estimation (does not consider actual compressed differential size)
        single_model_size = self._estimate_size(avg)
        self.last_uploaded_size_mb = single_model_size * len(weights_list)
        
        print(f" Single model: {single_model_size:.2f} MB")
        print(f" Total uploaded: {self.last_uploaded_size_mb:.2f} MB ({len(weights_list)} clients)")
        return avg

    def _estimate_size(self, state_dict):
        total = sum(p.numel() * p.element_size() for p in state_dict.values())
        return total / (1024 ** 2)

    # Compatibility method
    def train_on_client(self, global_model, train_examples):
        """Compatibility method that calls train_local_model."""
        use_kd = hasattr(self, '_current_round') and self._current_round > 1
        state_dict, num_samples, loss = self.train_local_model(
            global_model, train_examples, use_kd, 
            getattr(self, '_current_round', 1)
        )
        model = copy.deepcopy(global_model)
        model.load_state_dict(state_dict)
        return model.cpu()
    
    def set_current_round(self, round_idx):
        """Set current round index for compatibility."""
        self._current_round = round_idx
