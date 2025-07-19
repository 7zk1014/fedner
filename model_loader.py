from transformers import AutoTokenizer, AutoModelForTokenClassification
def load_pubmedbert_model(model_name, label_list):
    """
    加载预训练 PubMedBERT 模型，用于命名实体识别任务
    """
    num_labels = len(label_list)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model
