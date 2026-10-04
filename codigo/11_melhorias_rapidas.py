"""Ultima busca: variacoes rapidas em torno do stacking v2 (semente 123).

Cada variacao troca ou acrescenta UM bloco na v2 e e medida na validacao cruzada de 5 dobras.
Nada aqui altera a entrega. Uso:  python codigo/11_melhorias_rapidas.py
"""
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pipeline as P

V2 = ["lr_pal", "svc", "emb_e5small", "extremos"]


def main():
    c = P.Corpus(embeddings=("e5small", "minilm"))
    n, y = c.n_treino, c.y
    idx = np.arange(n)
    z = np.load(P.RES / "v2_blocos_completo.npz")
    Z = {k[2:]: z[k] for k in z.files if k.startswith("a_")}
    fold = np.zeros(n, dtype=int)
    for k, (_, v) in enumerate(P.dobras(y)):
        fold[v] = k

    def oof(f):
        partes = [f(idx[fold != k], y[fold != k], idx[fold == k]) for k in range(P.N_FOLDS)]
        o = np.zeros((n, partes[0].shape[1]))
        for k in range(P.N_FOLDS):
            o[fold == k] = partes[k]
        return o

    def lr(M, C, **kw):
        return lambda a, ya, b: LogisticRegression(C=C, max_iter=300, **kw).fit(M[a], ya).predict_log_proba(M[b])

    def lr_emb(E, C):
        return lambda a, ya, b: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=300)).fit(
            E[a], ya).predict_log_proba(E[b])

    e5 = c.emb["e5small"]
    # (nome, bloco substituido na v2 ou None para acrescentar, funcao)
    variacoes = [
        ("lr_pal C=0.5", "lr_pal", lr(c.tfidf_pal, 0.5)),
        ("lr_pal C=2", "lr_pal", lr(c.tfidf_pal, 2.0)),
        ("lr_pal balanceada", "lr_pal", lr(c.tfidf_pal, 1.0, class_weight="balanced")),
        ("e5 C=0.003", "emb_e5small", lr_emb(e5, 0.003)),
        ("e5 C=0.1", "emb_e5small", lr_emb(e5, 0.1)),
    ]
    if "minilm" in c.emb:
        mi = c.emb["minilm"]
        variacoes += [
            ("+ minilm C=0.01", None, lr_emb(mi, 0.01)),
            ("minilm no lugar do e5", "emb_e5small", lr_emb(mi, 0.01)),
            ("e5+minilm concatenados", "emb_e5small", lr_emb(np.hstack([e5, mi]), 0.003)),
        ]

    ref, pred_ref = P.cv_meta(Z, V2, y, fold, 0.1)
    linhas = [dict(variacao="v2 (referencia)", cv=np.mean(ref), dobras_melhores_que_v2=0, so_ela_acerta=0, so_v2_acerta=0)]
    print(f"v2 (referencia): {np.mean(ref):.4f}", flush=True)
    for nome, troca, f in variacoes:
        t0 = time.time()
        Z[nome] = oof(f)
        blocos = [nome if b == troca else b for b in V2] + ([nome] if troca is None else [])
        accs, pred = P.cv_meta(Z, blocos, y, fold, 0.1)
        linhas.append(dict(variacao=nome, cv=np.mean(accs),
                           dobras_melhores_que_v2=int(sum(a > b for a, b in zip(accs, ref))),
                           so_ela_acerta=int(((pred == y) & (pred_ref != y)).sum()),
                           so_v2_acerta=int(((pred != y) & (pred_ref == y)).sum())))
        print(f"{nome}: {np.mean(accs):.4f} ({np.mean(accs)-np.mean(ref):+.4f}) | melhor em {linhas[-1]['dobras_melhores_que_v2']}/5 dobras"
              f" | so ela acerta {linhas[-1]['so_ela_acerta']} x so v2 {linhas[-1]['so_v2_acerta']} ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(linhas).to_csv(P.RES / "11_melhorias_rapidas.csv", index=False)


if __name__ == "__main__":
    main()
