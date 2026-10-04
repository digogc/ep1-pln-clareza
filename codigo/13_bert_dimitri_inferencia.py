"""Inferencia (so CPU, sem retreinar) do BERTimbau ajustado pelo Dimitri (resposta_dimitri/).

Ordem: teste (900) -> 200 linhas dos 80% de treino do holdout (diagnostico de memorizacao)
-> linhas do holdout de 20% (semente 123), em ordem aleatoria. Grava parciais a cada lote em
resultados/13_bert_logits.npz, para que a avaliacao possa usar o que ja estiver pronto.
Uso:  python codigo/13_bert_dimitri_inferencia.py
"""
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import LABELS, ROOT

SEED, MAXLEN, LOTE = 123, 256, 16
tr, te = pd.read_excel(ROOT / "train.xlsx"), pd.read_excel(ROOT / "test1.xlsx")
y = tr["clarity"].map({l: i for i, l in enumerate(LABELS)}).values
n = len(tr)
textos = [str(t) for t in tr["resp_text"]] + [str(t) for t in te["resp_text"]]
a, b = train_test_split(np.arange(n), test_size=0.2, random_state=SEED, stratify=y)
rng = np.random.RandomState(SEED)
ordem = np.r_[np.arange(n, n + len(te)), rng.choice(a, 200, replace=False), rng.permutation(b)]

torch.set_num_threads(4)
tok = AutoTokenizer.from_pretrained(ROOT / "resposta_dimitri" / "bertimbau_model")
mod = AutoModelForSequenceClassification.from_pretrained(ROOT / "resposta_dimitri" / "bertimbau_model").eval()
meio = MAXLEN // 2
ids_feitos, logits, t0 = [], [], time.time()
for i in range(0, len(ordem), LOTE):
    lote = ordem[i:i + LOTE]
    enc = tok([textos[j] for j in lote], add_special_tokens=False)["input_ids"]
    seqs = []
    for t in enc:                                   # truncamento cabeca+cauda, igual ao codigo do Dimitri
        if len(t) > MAXLEN - 2:
            t = t[:meio] + t[-(meio - 2):]
        seqs.append([tok.cls_token_id] + t + [tok.sep_token_id])
    m = max(len(s) for s in seqs)
    ii = torch.tensor([s + [tok.pad_token_id] * (m - len(s)) for s in seqs])
    am = torch.tensor([[1] * len(s) + [0] * (m - len(s)) for s in seqs])
    with torch.no_grad():
        logits.append(mod(input_ids=ii, attention_mask=am).logits.numpy())
    ids_feitos.extend(lote)
    if (i // LOTE) % 10 == 0 or i + LOTE >= len(ordem):
        np.savez(ROOT / "resultados" / "13_bert_logits.npz", ids=np.array(ids_feitos), logits=np.vstack(logits))
        print(f"{len(ids_feitos)}/{len(ordem)} textos em {time.time()-t0:.0f}s", flush=True)
