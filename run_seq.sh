#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

# 生成带时间戳的总日志文件
ts="$(date +'%Y%m%d_%H%M%S')"
master_log="logs/run_${ts}.log"

# 把整个脚本的 stdout+stderr 同时输出到屏幕和 master_log
exec > >(tee -a "$master_log") 2>&1

echo "日志文件：$master_log"
echo

# echo "== niFedAvg 0.1  =="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedAvg \
#   --partition_strategy noniid --noniid_alpha 0.1 \
#   --num_clients 10 --local_epochs 2 --rounds 50 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract

# echo "== niFedAvg 0.01  =="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedAvg \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract
# echo "== FedProx 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedProx \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract
# echo "== FedAdam 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedAdam \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract

# echo "== FedSAD 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract
# echo "== iFedAvg =="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedAvg \
#   --partition_strategy iid \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract




# echo "all layers sample200"
# echo "== niFedAvg 0.01  =="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedAvg \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --train_layers -1
# echo "== FedAdam 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedAdam \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --train_layers -1

echo "== FedProx 0.01=="
PYTHONUNBUFFERED=1 python run_experiment.py \
  --alg FedProx \
  --partition_strategy noniid --noniid_alpha 0.01 \
  --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --mu 0.005

echo "== FedProx 0.01=="
PYTHONUNBUFFERED=1 python run_experiment.py \
  --alg FedProx \
  --partition_strategy noniid --noniid_alpha 0.01 \
  --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --mu 0.0005
  
# echo "== FedSD 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --temperature_start 2 --temperature_end 2 --alpha_ce_start 1 --alpha_ce_end  1 --alpha_kd_start 1 --alpha_kd_end 1 --teacher_policy 'prev_global' --train_layers -1

# echo "== FedSAD 0.01== "
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --train_layers -1


# echo "8 layers smaple200"
# echo "== niFedAvg 0.01  =="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedAvg \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --train_layers 8
  
# echo "== FedAdam 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedAdam \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --train_layers 8

# echo "== FedProx 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedProx \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --train_layers 8
  
# echo "== FedSD 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --temperature_start 2 --temperature_end 2 --alpha_ce_start 1 --alpha_ce_end  1 --alpha_kd_start 1 --alpha_kd_end 1 --teacher_policy 'prev_global' --train_layers 8
  
# echo "== FedSAD 0.01== "
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --train_layers 8

  
# echo "== niFedAvg 0.01  12layers=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedAvg \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 50 --train_last_n_layers=12 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract

# echo "== niFedAvg 0.01  alldata=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedAvg \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 30 --local_sample_size="all" --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract

# echo "== niFedAvg 0.01  alldata all layers"
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedAvg \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 30 --local_sample_size="all" --train_last_n_layers=12 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract

# echo "== niFedAvg 0.001  =="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedAvg \
#   --partition_strategy noniid --noniid_alpha 0.001 \
#   --num_clients 10 --local_epochs 2 --rounds 50 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract

# echo "== FedProx 0.1=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedProx \
#   --partition_strategy noniid --noniid_alpha 0.1 \
#   --num_clients 10 --local_epochs 2 --rounds 50 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract


# echo "== FedProx 0.001=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedProx \
#   --partition_strategy noniid --noniid_alpha 0.001 \
#   --num_clients 10 --local_epochs 2 --rounds 50 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract

# echo "== FedAdam 0.1=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedAdam \
#   --partition_strategy noniid --noniid_alpha 0.1 \
#   --num_clients 10 --local_epochs 2 --rounds 50 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract


  
# echo "== FedAdam 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedAdam \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 50 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract

# echo "== FedAdam 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedAdam \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 50 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract

# echo "== FedAdam 0.001=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedAdam \
#   --partition_strategy noniid --noniid_alpha 0.001 \
#   --num_clients 10 --local_epochs 2 --rounds 50 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract



# echo "== FedSD 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --train_layers -1

# echo "== FedSD 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract 
  
# echo "== FedSAD 0.01== "
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract 
# echo "== FedSAD 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --teacher_policy 'prev_global'
  
# echo "== FedSAD 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 50 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --temperature_start 2 --temperature_end 2 --alpha_ce_start 1 --alpha_ce_end  1 --alpha_kd_start 1 --alpha_kd_end 1
  
# echo "== FedSAD 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 50 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --temperature_start 2 --temperature_end 2 --alpha_ce_start 1 --alpha_ce_end  1 --alpha_kd_start 1 --alpha_kd_end 1 --teacher_policy 'prev_global'
  

  
# echo "== FedSAD 0.1=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.1 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --temperature_start 2 --temperature_end 2 --alpha_ce_start 1 --alpha_ce_end  1 --alpha_kd_start 1 --alpha_kd_end 1 --teacher_policy 'prev_global'



  
  




# echo "== niFedAvg 0.01  =="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedAvg \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --train_layers 8
# echo "== FedSD 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --train_layers 8

# echo "== FedSAD 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --train_layers 8
  
# echo "== FedAdam 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedAdam \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --train_layers 8

# echo "== FedProx 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedProx \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --train_layers 8

  

# echo "== niFedAvg 0.01  =="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedAvg \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --sample_size 500
# echo "== FedSD 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --sample_size 500

# echo "== FedSAD 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --sample_size 500
  
# echo "== FedAdam 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedAdam \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --sample_size 500

# echo "== FedProx 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedProx \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --sample_size 500



# echo "== niFedAvg 0.001  =="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedAvg \
#   --partition_strategy noniid --noniid_alpha 0.001 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract
# echo "== FedSD 0.001=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSD \
#   --partition_strategy noniid --noniid_alpha 0.001 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract

# echo "== FedSAD 0.001=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.001 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract
  
# echo "== FedAdam 0.001=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedAdam \
#   --partition_strategy noniid --noniid_alpha 0.001 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract

# echo "== FedProx 0.001=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedProx \
#   --partition_strategy noniid --noniid_alpha 0.001 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract


# echo "== niFedAvg 0.05  =="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedAvg \
#   --partition_strategy noniid --noniid_alpha 0.05 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract
# echo "== FedSD 0.05=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSD \
#   --partition_strategy noniid --noniid_alpha 0.05 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract

# echo "== FedSAD 0.05=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.05 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract
  
# echo "== FedAdam 0.05=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedAdam \
#   --partition_strategy noniid --noniid_alpha 0.05 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract

# echo "== FedProx 0.05=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedProx \
#   --partition_strategy noniid --noniid_alpha 0.05 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract

# echo "== FedSD 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --temperature_start 3.25 --temperature_end 3.25 --alpha_ce_start 0.85 --alpha_ce_end  0.85 --alpha_kd_start 0.15 --alpha_kd_end 0.15 --teacher_policy 'prev_global'
# echo "== FedSAD 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --first_ema_m0 0.8 --ema_m_min 0.8 --ema_m_max 0.8
  
# echo "== FedSAD 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --first_ema_m0 0.6 --ema_m_min 0.6 --ema_m_max 0.6
  
# echo "== FedSAD 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --first_ema_m0 0.5 --ema_m_min 0.5 --ema_m_max 0.5
  
# echo "== FedSAD 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --first_ema_m0 0.4 --ema_m_min 0.4 --ema_m_max 0.4
  
# echo "== FedSAD 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --first_ema_m0 0.8 --ema_m_min 0.05 --ema_m_max 0.8

# echo "== FedSAD 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --first_ema_m0 0.6 --ema_m_min 0.05 --ema_m_max 0.6
# echo "== FedAdam 0.001=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedAdam \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 30 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --sample_size all --train_layers -1
  
# echo "== FedProx 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedProx \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --mu 0.01
  
# echo "== FedProx 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedProx \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --mu 0.005

# echo "== FedProx 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedProx \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --mu 0.05
  
# echo "== FedProx 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedProx \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --mu 0.1
  

# echo "== niFedAvg 0.01  =="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedAvg \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract  --train_layers -1 --sample_size all
# echo "== FedSAD 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --first_ema_m0 0.6 --ema_m_min 0.6 --ema_m_max 0.6 --sample_size 500 --temperature_start 2 --temperature_end 2 --alpha_ce_start 1 --alpha_ce_end  1 --alpha_kd_start 1 --alpha_kd_end 1
  
# echo "== FedSAD 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --first_ema_m0 0.6 --ema_m_min 0.6 --ema_m_max 0.6 --sample_size 300 --temperature_start 2 --temperature_end 2 --alpha_ce_start 1 --alpha_ce_end  1 --alpha_kd_start 1 --alpha_kd_end 1
  
# echo "== FedSAD 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --first_ema_m0 0.6 --ema_m_min 0.6 --ema_m_max 0.6 --train_layers 8 --temperature_start 2 --temperature_end 2 --alpha_ce_start 1 --alpha_ce_end  1 --alpha_kd_start 1 --alpha_kd_end 1

# echo "== FedSAD 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract  --first_ema_m0 0.6 --ema_m_min 0.6 --ema_m_max 0.6  --train_layers -1 --temperature_start 2 --temperature_end 2 --alpha_ce_start 1 --alpha_ce_end  1 --alpha_kd_start 1 --alpha_kd_end 1






# echo "== FedSAD 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --temperature_start 3.25 --temperature_end 3.25 --alpha_ce_start 0.85 --alpha_ce_end  0.85 --alpha_kd_start 0.15 --alpha_kd_end 0.15

# echo "== FedSAD 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --sample_size 500
  
# echo "== FedSAD 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --sample_size 300

# echo "== FedSAD 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --train_layers -1
  
# echo "== FedSAD 0.01=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.01 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --train_layers 8


  

  
# echo "== FedSAD 0.1=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.1 \
#   --num_clients 10 --local_epochs 2 --rounds 100 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract --temperature_start 2 --temperature_end 2 --alpha_ce_start 1 --alpha_ce_end  1 --alpha_kd_start 1 --alpha_kd_end 1 --teacher_policy 'prev_global'
# echo "== FedKD 0.001=="
# PYTHONUNBUFFERED=1 python run_experiment.py \
#   --alg FedSAD \
#   --partition_strategy noniid --noniid_alpha 0.001 \
#   --num_clients 10 --local_epochs 2 --rounds 50 --model_name ./models/microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract

echo "全部实验完成！"
