"""Abordagem 3b - rede neural convolucional (TextCNN, Kim 2014) sobre a sequencia de palavras.

Cada palavra entra como o seu vetor pre-treinado do spaCy (pt_core_news_lg, 300d reduzido a
100d por PCA); convolucoes
de 2 a 5 palavras + max-pooling detectam expressoes; uma camada linear decide a classe.
Treino em CPU (PyTorch). O numero de epocas e escolhido pela validacao cruzada.
Uso:  python codigo/03_textcnn.py            (grade completa)
      python codigo/03_textcnn.py --cronometro (so mede o tempo por epoca)
"""
import re
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed

from common import load_data, save_test_xlsx, ROOT, N_FOLDS

RES, DADOS = ROOT / "resultados", ROOT / "dados"
TOKEN = re.compile(r"[A-Za-zÀ-ÿ]+|\d+")
MAXLEN, EPOCAS, LOTE, DIM = 200, 5, 64, 100
CONFIGS = [
    {"nome": "congelado_f100", "congelar": True, "filtros": 100, "dropout": 0.5},
    {"nome": "ajustado_f100", "congelar": False, "filtros": 100, "dropout": 0.5},
]


def preparar(T):
    ax, ae = DADOS / "cnn_ids.npy", DADOS / "cnn_emb.npy"
    if ax.exists() and ae.exists():
        return np.load(ax), np.load(ae)
    toks = [TOKEN.findall(t.lower()) for t in T]
    cont = Counter(p for t in toks for p in t)
    vocab = ["<pad>", "<unk>"] + [p for p, c in cont.most_common() if c >= 2]
    pos = {p: i for i, p in enumerate(vocab)}
    X = np.zeros((len(T), MAXLEN), dtype=np.int64)
    for i, t in enumerate(toks):
        ids = [pos.get(p, 1) for p in t[:MAXLEN]]
        X[i, :len(ids)] = ids
    import pt_core_news_lg
    nlp = pt_core_news_lg.load(disable=["tagger", "parser", "ner", "lemmatizer", "morphologizer",
                                        "attribute_ruler"])
    E = np.random.default_rng(42).normal(0, 0.1, (len(vocab), 300)).astype(np.float32)
    achou = 0
    for p, i in pos.items():
        if nlp.vocab.has_vector(p):
            v = nlp.vocab.get_vector(p)
            E[i] = v / (np.linalg.norm(v) + 1e-8)
            achou += 1
    # reduz os vetores de 300 para DIM dimensoes por PCA (deixa a convolucao ~3x mais rapida em CPU)
    from sklearn.decomposition import PCA
    tem = np.array([nlp.vocab.has_vector(p) for p in vocab])
    pca = PCA(DIM, random_state=42).fit(E[tem])
    R = np.random.default_rng(42).normal(0, 0.05, (len(vocab), DIM)).astype(np.float32)
    R[tem] = pca.transform(E[tem]).astype(np.float32)
    print(f"PCA 300->{DIM}: {100 * pca.explained_variance_ratio_.sum():.0f}% da variancia mantida", flush=True)
    E = R
    E[0] = 0
    print(f"vocabulario {len(vocab)} palavras, {achou} com vetor pre-treinado", flush=True)
    np.save(ax, X)
    np.save(ae, E)
    return X, E


class TextCNN(nn.Module):
    def __init__(self, E, congelar, filtros, dropout, **_):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.tensor(E), freeze=congelar, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(E.shape[1], filtros, k, padding=k // 2) for k in (2, 3, 4, 5)])
        self.drop = nn.Dropout(dropout)
        self.saida = nn.Linear(4 * filtros, 3)

    def forward(self, x):
        x = x[:, : max(int((x > 0).sum(1).max()), 5)]          # corta o padding do lote
        e = self.drop(self.emb(x)).transpose(1, 2)
        h = torch.cat([F.relu(c(e)).max(dim=2).values for c in self.convs], dim=1)
        return self.saida(self.drop(h))


def lotes(idx, comp, rng):
    """Lotes com textos de tamanho parecido (menos padding = treino mais rapido)."""
    idx = rng.permutation(idx)
    out = []
    for i in range(0, len(idx), LOTE * 20):
        bloco = idx[i:i + LOTE * 20]
        bloco = bloco[np.argsort(comp[bloco], kind="stable")]
        out += [bloco[j:j + LOTE] for j in range(0, len(bloco), LOTE)]
    return [out[i] for i in rng.permutation(len(out))]


def prever(modelo, X, idx):
    modelo.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), 256):
            out.append(torch.softmax(modelo(torch.tensor(X[idx[i:i + 256]])), dim=1).numpy())
    return np.vstack(out)


def treinar_fold(cfg, X, E, y, a, b, teste, cronometro=False):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    comp = (X > 0).sum(1)
    modelo = TextCNN(E, **cfg)
    opt = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    amostra = rng.choice(a, 3000, replace=False)
    hist = []
    for ep in range(EPOCAS):
        modelo.train()
        t0 = time.time()
        for n, lote in enumerate(lotes(a, comp, rng)):
            opt.zero_grad()
            perda = F.cross_entropy(modelo(torch.tensor(X[lote])), torch.tensor(y[lote]), label_smoothing=0.05)
            perda.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 2.0)
            opt.step()
            if cronometro and n == 30:
                dt = (time.time() - t0) / 31
                return f"{cfg['nome']}: {dt:.3f}s/lote -> {dt * len(a) / LOTE:.0f}s/epoca"
        pb = prever(modelo, X, b)
        hist.append((pb, prever(modelo, X, teste),
                     float((prever(modelo, X, amostra).argmax(1) == y[amostra]).mean())))
    return hist


def main():
    tr, te = load_data()
    y, fold, N = tr["y"].values, tr["fold"].values, len(tr)
    T = list(tr["text"]) + list(te["text"])
    X, E = preparar(T)
    idx, teste = np.arange(N), np.arange(N, len(T))
    if "--cronometro" in sys.argv:
        for cfg in CONFIGS:
            print(treinar_fold(cfg, X, E, y, idx[fold != 0], idx[fold == 0], teste, cronometro=True), flush=True)
        return
    linhas, melhor = [], None
    for cfg in CONFIGS:
        t0 = time.time()
        res = Parallel(n_jobs=2)(delayed(treinar_fold)(cfg, X, E, y, idx[fold != k], idx[fold == k], teste)
                                 for k in range(N_FOLDS))
        for ep in range(EPOCAS):
            oof = np.zeros((N, 3), dtype=np.float32)
            for k in range(N_FOLDS):
                oof[fold == k] = res[k][ep][0]
            va = [float((res[k][ep][0].argmax(1) == y[fold == k]).mean()) for k in range(N_FOLDS)]
            pt = np.mean([res[k][ep][1] for k in range(N_FOLDS)], axis=0)   # media dos 5 modelos
            l = dict(config=cfg["nome"], embeddings="congelados" if cfg["congelar"] else "ajustados",
                     filtros=cfg["filtros"], dropout=cfg["dropout"], epocas=ep + 1,
                     acc_val_media=np.mean(va), acc_val_desvio=np.std(va),
                     acc_treino_media=np.mean([res[k][ep][2] for k in range(N_FOLDS)]),
                     **{f"acc_dobra{k}": v for k, v in enumerate(va)})
            linhas.append(l)
            print(f"{cfg['nome']:16s} epoca {ep+1}: val={l['acc_val_media']:.4f} "
                  f"treino={l['acc_treino_media']:.4f}", flush=True)
            if melhor is None or l["acc_val_media"] > melhor[0]["acc_val_media"]:
                melhor = (l, oof, pt)
        pd.DataFrame(linhas).to_csv(RES / "03_textcnn_grid.csv", index=False)
        print(f"{cfg['nome']} ok em {time.time()-t0:.0f}s", flush=True)
    l, oof, pt = melhor
    print(f"\nMELHOR: {l['config']} com {l['epocas']} epocas val={l['acc_val_media']:.4f}", flush=True)
    np.save(RES / "03_textcnn_oof_melhor.npy", oof)
    np.save(RES / "03_textcnn_teste_melhor.npy", pt)
    out = save_test_xlsx(pt.argmax(1), "test1_textcnn.xlsx")
    print("teste rotulado em", out, "| distribuicao:",
          pd.Series(pt.argmax(1)).value_counts().sort_index().to_dict(), flush=True)


if __name__ == "__main__":
    main()
