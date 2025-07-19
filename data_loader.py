import os

def read_bio_file(path):
    sentences = []
    sentence = {"tokens": [], "labels": []}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if sentence["tokens"]:
                    sentences.append(sentence)
                    sentence = {"tokens": [], "labels": []}
                continue
            if len(line.split()) == 2:
                token, label = line.split()
                sentence["tokens"].append(token)
                sentence["labels"].append(label)
    if sentence["tokens"]:
        sentences.append(sentence)
    return sentences

def split_clients(data, num_clients=3):
    total = len(data)
    size = total // num_clients
    return [data[i*size:(i+1)*size] for i in range(num_clients)]

def load_medmentions_bio(train_path, dev_path, test_path, num_clients=3):
    train_data = read_bio_file(train_path)
    dev_data = read_bio_file(dev_path)
    test_data = read_bio_file(test_path)
    client_train_sets = split_clients(train_data, num_clients=num_clients)
    return client_train_sets, dev_data, test_data

def extract_label_list_from_bio(bio_path):
    label_set = set()
    with open(bio_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and len(line.split()) == 2:
                _, label = line.split()
                label_set.add(label)
    return sorted(label_set)