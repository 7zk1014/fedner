from transformers import Trainer, TrainingArguments
from datasets import Dataset
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

def convert_to_dataset(examples):
    return Dataset.from_list(examples)

def align_labels_with_tokens(tokenized_inputs, labels, label2id):
    aligned_labels = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id.get(label[word_idx], 0))
            else:
                label_ids.append(label2id.get(label[word_idx], 0))
            previous_word_idx = word_idx
        aligned_labels.append(label_ids)
    return aligned_labels

def evaluate_model(model, tokenizer, test_examples, label_list):
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    test_ds = convert_to_dataset(test_examples)

    def preprocess(example):
        tokenized = tokenizer(example["tokens"],
                              truncation=True,
                              is_split_into_words=True,
                              padding="max_length",
                              max_length=128)
        tokenized["labels"] = align_labels_with_tokens(tokenized, [example["labels"]], label2id)[0]
        return tokenized

    test_ds = test_ds.map(preprocess)

    args = TrainingArguments(
        output_dir="./eval_tmp",
        per_device_eval_batch_size=8
    )
    trainer = Trainer(model=model, args=args, tokenizer=tokenizer)
    predictions = trainer.predict(test_ds)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids

    true_labels = []
    pred_labels = []

    for true_seq, pred_seq in zip(labels, preds):
        true_tags = []
        pred_tags = []
        for t, p in zip(true_seq, pred_seq):
            if t != -100:
                true_tags.append(id2label[t])
                pred_tags.append(id2label[p])
        true_labels.append(true_tags)
        pred_labels.append(pred_tags)

    return {
        "f1": f1_score(true_labels, pred_labels),
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "accuracy": accuracy_score(true_labels, pred_labels)
    }
