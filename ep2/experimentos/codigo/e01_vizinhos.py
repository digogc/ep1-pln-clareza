"""Experimento 1: atributos de vizinhos mais proximos (kNN) como bloco do stacking.

Para cada resposta, procura as respostas mais parecidas do treino (cosseno em TF-IDF e nos embeddings MiniLM)
e resume os rotulos delas: distribuicao ponderada pela similaridade nos k vizinhos (k = 5, 25, 100),
similaridade maxima e rotulos das copias quase identicas (cosseno > 0,98).
Fora da dobra: os vizinhos de uma resposta da dobra k vem so das outras dobras.
"""
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from comum import (EXP, N_DOBRAS, V2, blocos_v2, carregar, cv_stacking, dobras, fmt, salvar_bloco)

KS = (5, 25, 100)


def atributos(S, yb):
    """S: similaridades (consultas x base), yb: rotulos da base -> atributos por consulta."""
    Y = np.eye(3, dtype=np.float32)[yb]
    ordem = np.argpartition(-S, max(KS), axis=1)[:, :max(KS)]
    sims = np.take_along_axis(S, ordem, 1)
    o2 = np.argsort(-sims, 1)
    ordem, sims = np.take_along_axis(ordem, o2, 1), np.take_along_axis(sims, o2, 1)
    feats = []
    for k in KS:
        w = np.clip(sims[:, :k], 0, None)[:, :, None]
        dist = (w * Y[ordem[:, :k]]).sum(1) / (w.sum(1) + 1e-6)
        feats.append(dist)
    quase = (sims > 0.98)[:, :, None]
    n_quase = quase.sum(1)
    dist_q = (quase * Y[ordem]).sum(1) / np.maximum(n_quase, 1)
    feats += [sims[:, :1], np.log1p(n_quase), dist_q]
    return np.hstack(feats).astype(np.float32)


def blocos_knn(R, y, dobra, n, lote=2000):
    fora = np.zeros((n, 3 * len(KS) + 5), np.float32)
    for k in range(N_DOBRAS):
        base, cons = np.where(dobra != k)[0], np.where(dobra == k)[0]
        for i in range(0, len(cons), lote):
            c = cons[i:i + lote]
            S = R[c] @ R[base].T
            S = S.toarray() if hasattr(S, "toarray") else S
            fora[c] = atributos(np.asarray(S, np.float32), y[base])
    teste = []
    base = np.arange(n)
    for i in range(n, R.shape[0], lote):
        S = R[i:i + lote] @ R[base].T
        S = S.toarray() if hasattr(S, "toarray") else S
        teste.append(atributos(np.asarray(S, np.float32), y))
    return fora, np.vstack(teste)


def main():
    t0 = time.time()
    d = carregar()
    y, n = d["y"], len(d["treino"])
    textos = np.r_[d["treino"], d["test1"], d["test2"]]
    dobra = dobras(y)
    R_tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(textos).astype(np.float32).tocsr()
    R_emb = np.load(EXP.parent / "cache" / "emb_minilm.npy").astype(np.float32)
    for nome, R in (("knn_tfidf", R_tfidf), ("knn_minilm", R_emb)):
        t1 = time.time()
        fora, teste = blocos_knn(R, y, dobra, n)
        salvar_bloco(nome, fora, teste)
        print(f"{nome} pronto ({time.time()-t1:.0f}s); acerto do voto k=25 sozinho: {(fora[:, 3:6].argmax(1) == y).mean()*100:.2f}", flush=True)

    fora, _, _ = blocos_v2()
    from comum import carregar_bloco
    base = np.hstack([fora[b] for b in V2])
    print("\nstacking (media | dobras 1-5):")
    print("v2                       ", fmt(cv_stacking(base, y, dobra)))
    for extra in (["knn_tfidf"], ["knn_minilm"], ["knn_tfidf", "knn_minilm"]):
        X = np.hstack([base] + [carregar_bloco(e)[0] for e in extra])
        print(f"v2 + {'+'.join(extra):20s}", fmt(cv_stacking(X, y, dobra)))
    print(f"tempo {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
