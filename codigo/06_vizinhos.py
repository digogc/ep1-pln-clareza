"""Abordagem 6 - "memoria difusa": rotulos das respostas MAIS PARECIDAS do treino.

Generaliza a memoria de respostas repetidas: em vez de exigir texto identico, procura os
vizinhos mais proximos (cosseno) e resume como eles foram rotulados. Rodado sozinho, este script
usa leave-one-out (a propria linha nunca entra entre os vizinhos), o que VAZA rotulo em textos
repetidos; o pipeline final (pipeline.py) reutiliza topk/resumo, mas fora da dobra. Espacos de vizinhanca: TF-IDF de palavras e, se existirem, os embeddings
de sentenca de dados/emb_*.npy.
Uso:  python codigo/06_vizinhos.py
"""
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import load_data, ROOT, N_FOLDS

RES = ROOT / "resultados"
KS = [5, 20, 100]
LIMIARES = [0.6, 0.8, 0.95]


def topk(A, B, kmax, excluir_diagonal, bloco=1000):
    """Para cada linha de A, indices e similaridades dos kmax vizinhos em B (produto interno)."""
    idx = np.zeros((A.shape[0], kmax), dtype=np.int32)
    sim = np.zeros((A.shape[0], kmax), dtype=np.float32)
    for i in range(0, A.shape[0], bloco):
        S = A[i:i + bloco] @ B.T
        S = np.asarray(S.todense()) if hasattr(S, "todense") else np.array(S)
        if excluir_diagonal:
            S[np.arange(S.shape[0]), np.arange(i, i + S.shape[0])] = -1
        p = np.argpartition(-S, kmax, axis=1)[:, :kmax]
        s = np.take_along_axis(S, p, 1)
        o = np.argsort(-s, axis=1)
        idx[i:i + bloco], sim[i:i + bloco] = np.take_along_axis(p, o, 1), np.take_along_axis(s, o, 1)
    return idx, sim


def resumo(idx, sim, y):
    """Atributos: proporcao de rotulos (ponderada pela similaridade) nos k vizinhos e acima de limiares."""
    Y = np.eye(3)[y[idx]]                      # (n, kmax, 3)
    cols = []
    for k in KS:
        w = np.clip(sim[:, :k], 0, None)[:, :, None]
        cols.append((Y[:, :k] * w).sum(1) / (w.sum(1) + 1e-9))
    for lim in LIMIARES:
        m = (sim >= lim)[:, :, None]
        n = m.sum(1)
        cols.append((Y * m).sum(1) / np.maximum(n, 1))
        cols.append(np.log1p(n))
    cols.append(sim[:, [0, 4, 19]])
    return np.hstack(cols).astype(np.float32)


def espacos(tr, te):
    textos = pd.concat([tr["text"], te["text"]])
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(textos)
    yield "tfidf", v.transform(tr["text"]).tocsr(), v.transform(te["text"]).tocsr()
    for f in sorted((ROOT / "dados").glob("emb_*.npy")):
        E = np.load(f)
        yield f.stem[4:], E[:len(tr)], E[len(tr):]


def main():
    tr, te = load_data()
    y, fold = tr["y"].values, tr["fold"].values
    for nome, A, B in espacos(tr, te):
        t0 = time.time()
        Ftr = resumo(*topk(A, A, max(KS), True), y)
        Fte = resumo(*topk(B, A, max(KS), False), y)
        np.save(RES / f"06_vizinhos_{nome}_treino.npy", Ftr)
        np.save(RES / f"06_vizinhos_{nome}_teste.npy", Fte)
        # qualidade isolada: regressao logistica sobre os atributos de vizinhanca, nas dobras do projeto
        accs = []
        for k in range(N_FOLDS):
            a, b = fold != k, fold == k
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(Ftr[a], y[a])
            accs.append(float((m.predict(Ftr[b]) == y[b]).mean()))
        print(f"{nome}: {Ftr.shape[1]} atributos | acuracia isolada {np.mean(accs):.4f} +- {np.std(accs):.4f}"
              f" | voto dos 20 vizinhos {(Ftr[:, 3:6].argmax(1) == y).mean():.4f} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
