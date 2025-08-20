# model_loader.py
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification


def _local_model_ready(model_dir: Path):
    """
    判断本地模型目录是否“可直接加载”（文件齐全）。
    返回: (ok, missing_files, has_weights)
    """
    must = ["config.json", "tokenizer_config.json", "special_tokens_map.json"]
    has_weights = (model_dir / "pytorch_model.bin").exists() or (model_dir / "model.safetensors").exists()
    missing = [f for f in must if not (model_dir / f).exists()]
    ok = (len(missing) == 0) and has_weights
    return ok, missing, has_weights


def load_pubmedbert_model(model_name_or_path, label_list):
    """
    加载 PubMedBERT（若传入本地目录且文件齐全 -> 用本地；否则回退到 HuggingFace Hub）
    """
    if "O" not in label_list:
        raise ValueError("label_list 必须包含 'O'。")

    # 规范化路径并检测是否可用本地
    resolved = Path(os.path.expanduser(model_name_or_path)).resolve()
    is_dir = resolved.is_dir()
    use_local = False
    model_source = model_name_or_path  # 默认认为是 Hub 名称

    if is_dir:
        ok, missing, has_w = _local_model_ready(resolved)
        if ok:
            use_local = True
            model_source = str(resolved)
            print(f"📁 从本地加载模型: {model_source}")
        else:
            print(
                "⚠️ 本地目录存在但文件不完整，将回退到 HuggingFace Hub：\n"
                f"   目录: {resolved}\n"
                f"   缺少: {missing} | 有权重: {has_w}"
            )
            print(f"🌐 从 HuggingFace Hub 加载模型: {model_name_or_path}")
    else:
        print(f"🌐 从 HuggingFace Hub 加载模型: {model_name_or_path}")

    # 必须使用 fast tokenizer 才有 .word_ids()
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        use_fast=True,
        local_files_only=use_local  # 本地加载时禁止联网；回退 Hub 时允许联网
    )

    num_labels = len(label_list)
    model = AutoModelForTokenClassification.from_pretrained(
        model_source,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        local_files_only=use_local
    )

    # 配置标签映射，确保评估/训练一致
    id2label = {i: l for i, l in enumerate(label_list)}
    model.config.id2label = id2label
    model.config.label2id = {l: i for i, l in id2label.items()}

    # FedKD 会用到
    model.config.output_hidden_states = True
    # 若需要注意力： model.config.output_attentions = True

    return tokenizer, model


def check_model_availability(model_name_or_path):
    """
    返回 True 表示可用（本地可用或可从网络访问）。
    """
    resolved = Path(os.path.expanduser(model_name_or_path)).resolve()
    if resolved.is_dir():
        ok, _, _ = _local_model_ready(resolved)
        return ok
    # 远端一律视为可用（是否能联网由运行环境决定）
    return True


def get_local_model_info(model_path):
    model_dir = Path(model_path)
    if not model_dir.exists():
        return None

    size_mb = sum(
        (model_dir / f).stat().st_size
        for f in model_dir.iterdir()
        if f.is_file()
    ) / (1024 * 1024)

    info = {"path": str(model_dir), "exists": True, "size_mb": size_mb}

    info_file = model_dir / "download_info.json"
    if info_file.exists():
        import json
        try:
            info.update(json.loads(info_file.read_text(encoding="utf-8")))
        except Exception:
            pass

    return info
