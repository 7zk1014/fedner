import os
import random
import numpy as np
from collections import Counter

# 1. Parse PubTator format file into document-level dicts
# Each document: {'pmid', 'text', 'annotations': [{'start','end','mention','semtype','entity'}, ...]}
def load_pubtator(path):
    docs = []
    cur = {"pmid": None, "text": "", "annotations": []}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                if cur["pmid"]:
                    docs.append(cur)
                cur = {"pmid": None, "text": "", "annotations": []}
                continue
            if "|t|" in line or "|a|" in line:
                pmid, tag, txt = line.split("|", 2)
                if tag == "t":
                    cur["pmid"] = pmid
                    cur["text"] = txt
                else:
                    cur["text"] += " " + txt
            else:
                parts = line.split("\t")
                _, start, end, mention, semtypes, entity = parts
                cur["annotations"].append({
                    "start":    int(start),
                    "end":      int(end),
                    "mention":  mention,
                    "semtype":  semtypes.split(",")[0],
                    "entity":   entity
                })
        if cur["pmid"]:
            docs.append(cur)
    return docs

# 2. Read official PMID split files (one PMID per line)
def read_pmids(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

# 3. Count labels in each document
def add_label_counts(docs):
    for d in docs:
        labels = [ann['semtype'] for ann in d['annotations']]
        d['label_counts'] = Counter(labels)
    return docs

# 4. Split documents to clients: IID or Dirichlet Non-IID
def split_data_docs(dataset, num_clients, strategy='iid', alpha=0.5):
    if strategy == 'iid':
        random.shuffle(dataset)
        shards = np.array_split(dataset, num_clients)
        return {i: list(shards[i]) for i in range(num_clients)}
    # Non-IID partition via Dirichlet distribution. Each document should
    # appear in exactly one client's list. For every label we sample a
    # Dirichlet proportion across clients and allocate the corresponding
    # number of documents with that label. Documents that have already been
    # assigned by a previous label are skipped to avoid duplicates.
    all_labels = sorted({l for d in dataset for l in d['label_counts']})
    client_docs = {i: [] for i in range(num_clients)}
    assigned = set()

    for label in all_labels:
        # Indices of docs containing this label
        idxs = [i for i, d in enumerate(dataset) if d['label_counts'].get(label, 0) > 0]
        if not idxs:
            continue
        random.shuffle(idxs)

        # Dirichlet draw to determine how many documents per client
        props = np.random.dirichlet([alpha] * num_clients)
        counts = (props * len(idxs)).astype(int)
        while counts.sum() < len(idxs):
            counts[np.argmax(props)] += 1

        start = 0
        for cid, cnt in enumerate(counts):
            taken = 0
            while taken < cnt and start < len(idxs):
                idx = idxs[start]
                start += 1
                if idx in assigned:
                    continue
                client_docs[cid].append(dataset[idx])
                assigned.add(idx)
                taken += 1

    # Any documents not assigned by the above procedure (e.g., without labels)
    # are distributed round-robin to keep dataset sizes consistent.
    unassigned = [dataset[i] for i in range(len(dataset)) if i not in assigned]
    for i, doc in enumerate(unassigned):
        client_docs[i % num_clients].append(doc)

    return client_docs

# 5. Flatten document-level to sentence-level samples via annotations
# Each sentence dict: {'tokens': [...], 'labels': [...]} 
def docs_to_sentences(doc_list):
    sents = []
    for doc in doc_list:
        toks = doc['text'].split()
        tags = ['O'] * len(toks)
        for ann in doc['annotations']:
            ment = ann['mention'].split()
            for i in range(len(toks)-len(ment)+1):
                if toks[i:i+len(ment)] == ment:
                    tags[i] = 'B-'+ann['semtype']
                    for j in range(1, len(ment)):
                        tags[i+j] = 'I-'+ann['semtype']
        sents.append({'tokens': toks, 'labels': tags})
    return sents

# 6. Main API: load PubTator docs, apply official splits, split train, flatten to sentences
# Returns: client_train_sents (list of lists), dev_sents, test_sents
def load_and_split_pubtator(
    pubtator_path,
    trng_pmids_path,
    dev_pmids_path,
    test_pmids_path,
    num_clients=5,
    partition_strategy='iid',
    noniid_alpha=0.5
):
    # load documents
    all_docs = load_pubtator(pubtator_path)
    doc_map  = {d['pmid']: d for d in all_docs}

    # official splits
    train_pmids = read_pmids(trng_pmids_path)
    dev_pmids   = read_pmids(dev_pmids_path)
    test_pmids  = read_pmids(test_pmids_path)

    train_docs = [doc_map[p] for p in train_pmids if p in doc_map]
    dev_docs   = [doc_map[p] for p in dev_pmids   if p in doc_map]
    test_docs  = [doc_map[p] for p in test_pmids  if p in doc_map]

    # count labels & split train docs
    train_docs = add_label_counts(train_docs)
    client_docs = split_data_docs(
        train_docs, num_clients,
        strategy=partition_strategy,
        alpha=noniid_alpha
    )

    # flatten to sentence-level
    client_train_sents = []
    for cid in range(num_clients):
        docs = client_docs.get(cid, [])
        client_train_sents.append(docs_to_sentences(docs))
    dev_sents  = docs_to_sentences(dev_docs)
    test_sents = docs_to_sentences(test_docs)
    return client_train_sents, dev_sents, test_sents
