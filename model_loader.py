# model_loader.py
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification


def _local_model_ready(model_dir: Path):
    """
    Check if local model directory is "directly loadable" (complete files).
    Returns: (ok, missing_files, has_weights)
    """
    must = ["config.json", "tokenizer_config.json", "special_tokens_map.json"]
    has_weights = (model_dir / "pytorch_model.bin").exists() or (model_dir / "model.safetensors").exists()
    missing = [f for f in must if not (model_dir / f).exists()]
    ok = (len(missing) == 0) and has_weights
    return ok, missing, has_weights


def load_pubmedbert_model(model_name_or_path, label_list):
    """
    Load PubMedBERT (if passed local directory and files complete -> use local; otherwise fallback to HuggingFace Hub)
    """
    if "O" not in label_list:
        raise ValueError("label_list must contain 'O'.")

    resolved = Path(os.path.expanduser(model_name_or_path)).resolve()
    is_dir = resolved.is_dir()
    use_local = False
    model_source = model_name_or_path

    if is_dir:
        ok, missing, has_w = _local_model_ready(resolved)
        if ok:
            use_local = True
            model_source = str(resolved)
            print(f" Loading model from local: {model_source}")
        else:
            print(
                " Local directory exists but files incomplete, falling back to HuggingFace Hub:\n"
                f"   Directory: {resolved}\n"
                f"   Missing: {missing} | Has weights: {has_w}"
            )
            print(f" Loading model from HuggingFace Hub: {model_name_or_path}")
    else:
        print(f" Loading model from HuggingFace Hub: {model_name_or_path}")

    # Must use fast tokenizer to have .word_ids()
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        use_fast=True,
        local_files_only=use_local  # Disable network when loading locally; allow network when falling back to Hub
    )

    num_labels = len(label_list)
    model = AutoModelForTokenClassification.from_pretrained(
        model_source,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        local_files_only=use_local
    )

    # Configure label mapping to ensure evaluation/training consistency
    id2label = {i: l for i, l in enumerate(label_list)}
    model.config.id2label = id2label
    model.config.label2id = {l: i for i, l in id2label.items()}

    # FedKD will use this
    model.config.output_hidden_states = True
    # If attention needed: model.config.output_attentions = True

    return tokenizer, model


def check_model_availability(model_name_or_path):
    """
    Return True means available (locally available or accessible from network).
    """
    resolved = Path(os.path.expanduser(model_name_or_path)).resolve()
    if resolved.is_dir():
        ok, _, _ = _local_model_ready(resolved)
        return ok
    # Remote models are always considered available (network connectivity depends on runtime environment)
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
