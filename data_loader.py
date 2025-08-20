import re
import random
import pickle
import numpy as np
from collections import defaultdict, Counter
from scipy.stats import dirichlet
import os
import pandas as pd

def load_pubtator(path):
    docs = []
    cur = {"pmid": None, "text": "", "annotations": []}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line == "":
                if cur["pmid"]:
                    docs.append(cur)
                cur = {"pmid": None, "text": "", "annotations": []}
                continue
            if "|t|" in line or "|a|" in line:
                pmid, tag, txt = line.split("|", 2)
                if tag == "t":
                    cur["pmid"], cur["text"] = pmid, txt
                else:
                    cur["text"] += " " + txt
            else:
                parts = line.split("\t")
                if len(parts) >= 6:
                    _, start, end, mention, semtypes, _ = parts[:6]
                    cur["annotations"].append({
                        "start": int(start),
                        "end": int(end),
                        "mention": mention,
                        "semtype": semtypes.split(",")[0]
                    })
    return docs

def read_pmids(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def docs_to_sentences(docs):
    pattern = re.compile(r"[^.?!]+[.?!]?")
    sents = []
    for d in docs:
        for m in pattern.finditer(d["text"]):
            sent = m.group().strip()
            if not sent:
                continue
            toks = sent.split()
            tags = ["O"] * len(toks)
            for ann in d["annotations"]:
                ment = ann["mention"].split()
                for i in range(len(toks) - len(ment) + 1):
                    if toks[i:i+len(ment)] == ment:
                        tags[i] = "B-" + ann["semtype"]
                        for j in range(1, len(ment)):
                            tags[i+j] = "I-" + ann["semtype"]
            sents.append({"tokens": toks, "labels": tags})
    return sents

def multi_label_dirichlet_routing_partition(sentences, num_clients, alpha=0.1, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    # 1. 为每个标签生成 Dirichlet 分布概率向量
    all_labels = sorted({lab[2:] for s in sentences for lab in s["labels"] if lab != "O"})
    label2client_probs = {
        l: np.random.dirichlet([alpha] * num_clients) for l in all_labels
    }

    # 2. 遍历每个句子，根据其标签决定客户端归属
    client_sent_ids = {i: [] for i in range(num_clients)}
    for idx, s in enumerate(sentences):
        labels = [lab[2:] for lab in s["labels"] if lab != "O"]
        if not labels:
            assigned_client = random.randint(0, num_clients - 1)
        else:
            scores = np.zeros(num_clients)
            for l in labels:
                scores += label2client_probs[l]
            assigned_client = int(np.argmax(scores))
        client_sent_ids[assigned_client].append(idx)

    return {cid: [sentences[i] for i in sent_ids] for cid, sent_ids in client_sent_ids.items()}

def load_and_split_pubtator(
    pubtator_path,
    trng_pmids_path,
    dev_pmids_path,
    test_pmids_path,
    num_clients=5,
    partition_strategy="iid",
    alpha=0.1,
    seed=42,
    cache_path=None
):
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    docs = load_pubtator(pubtator_path)
    pmid_map = {d["pmid"]: d for d in docs}

    # 读取各阶段文档
    train_docs = [pmid_map[p] for p in read_pmids(trng_pmids_path) if p in pmid_map]
    dev_docs   = [pmid_map[p] for p in read_pmids(dev_pmids_path)   if p in pmid_map]
    test_docs  = [pmid_map[p] for p in read_pmids(test_pmids_path)  if p in pmid_map]

    # 转成句子
    train_sents = docs_to_sentences(train_docs)
    dev_sents   = docs_to_sentences(dev_docs)
    test_sents  = docs_to_sentences(test_docs)

    # 根据策略分配训练集
    if partition_strategy == "iid":
        # 训练集 IID 划分
        idxs = list(range(len(train_sents)))
        random.seed(seed); random.shuffle(idxs)
        parts = [idxs[i::num_clients] for i in range(num_clients)]
        client_train = {i: [train_sents[j] for j in parts[i]] for i in range(num_clients)}

        # 验证集 IID 划分
        idxs_dev = list(range(len(dev_sents)))
        random.seed(seed)
        random.shuffle(idxs_dev)
        parts_dev = [idxs_dev[i::num_clients] for i in range(num_clients)]
        client_dev = {i: [dev_sents[j] for j in parts_dev[i]] for i in range(num_clients)}

        # 测试集 IID 划分
        idxs_test = list(range(len(test_sents)))
        random.seed(seed)
        random.shuffle(idxs_test)
        parts_test = [idxs_test[i::num_clients] for i in range(num_clients)]
        client_test = {i: [test_sents[j] for j in parts_test[i]] for i in range(num_clients)}

    elif partition_strategy == "noniid":
        # 训练集非 IID
        client_train = multi_label_dirichlet_routing_partition(train_sents, num_clients, alpha, seed)
        # 验证集、测试集也用同样的 Dirichlet 分配
        client_dev   = multi_label_dirichlet_routing_partition(dev_sents,   num_clients, alpha, seed)
        client_test  = multi_label_dirichlet_routing_partition(test_sents,  num_clients, alpha, seed)
    else:
        raise ValueError(f"Unknown partition strategy: {partition_strategy}")

    # 转成 list of lists
    client_train_sents = [client_train[i] for i in range(num_clients)]
    client_dev_sents   = [client_dev[i]   for i in range(num_clients)]
    client_test_sents  = [client_test[i]  for i in range(num_clients)]

    result = (client_train_sents, client_dev_sents, client_test_sents)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)

    return result

def print_client_label_distribution(client_sents):
    for i, sents in enumerate(client_sents):
        labs = [lab[2:] for s in sents for lab in s["labels"] if lab != "O"]
        dist = Counter(labs)
        df = pd.DataFrame(dist.items(), columns=["Label", "Count"]).sort_values("Count", ascending=False)
        print(f"\nClient {i} - Total Sentences: {len(sents)}")
        print(df.head(10))
        print("Unique labels:", len(dist))
