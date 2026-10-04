"""Busca final em torno do stacking com MiniLM (semente 123), sempre por validacao cruzada de 5 dobras.

Testa acrescentar/trocar blocos ja calculados e novos "mini modelos" de embeddings (cada um vira
uma regressao logistica sobre o vetor). Mostra a acuracia em cada dobra. Nao altera a entrega.
Uso:  python codigo/15_busca_final.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pipeline as P

ATUAL = ["lr_pal", "svc", "emb_minilm", "extremos"]


def main():
    c = P.Corpus(embeddings=("e5small", "minilm", "distiluse"))
    n, y = c.n_treino, c.y
    idx = np.arange(n)
    z = np.load(P.RES / "v2_blocos_completo.npz")
    Z = {k[2:]: z[k] for k in z.files if k.startswith("a_")}
    fold = np.zeros(n, dtype=int)
    for k, (_, v) in enumerate(P.dobras(y)):
        fold[v] = k

    def emb(E, C):
        o = np.zeros((n, 3))
        for k in range(P.N_FOLDS):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=300)).fit(E[idx[fold != k]], y[fold != k])
            o[fold == k] = m.predict_log_proba(E[idx[fold == k]])
        return o

    Z["minilm C=0.003"] = emb(c.emb["minilm"], 0.003)
    Z["minilm C=0.03"] = emb(c.emb["minilm"], 0.03)
    Z["minilm+e5 concat"] = emb(np.hstack([c.emb["minilm"], c.emb["e5small"]]), 0.003)
    opcoes = {
        "ATUAL (MiniLM)": ATUAL,
        "MiniLM com C=0.003": ["lr_pal", "svc", "minilm C=0.003", "extremos"],
        "MiniLM com C=0.03": ["lr_pal", "svc", "minilm C=0.03", "extremos"],
        "+ e5-small": ATUAL + ["emb_e5small"],
        "MiniLM e e5 concatenados": ["lr_pal", "svc", "minilm+e5 concat", "extremos"],
        "+ SVM de palavras": ATUAL + ["svc_pal"],
        "+ eixo ordinal": ATUAL + ["ordinal"],
        "+ LightGBM de atributos": ATUAL + ["lgbm"],
        "sem o SVM": ["lr_pal", "emb_minilm", "extremos"],
        "sem 'extremos'": ["lr_pal", "svc", "emb_minilm"],
    }
    if "distiluse" in c.emb:
        Z["emb_distiluse"] = emb(c.emb["distiluse"], 0.01)
        Z["tres embeddings concat"] = emb(np.hstack([c.emb[k] for k in ("minilm", "e5small", "distiluse")]), 0.003)
        opcoes.update({
            "+ distiluse": ATUAL + ["emb_distiluse"],
            "distiluse no lugar do MiniLM": ["lr_pal", "svc", "emb_distiluse", "extremos"],
            "+ distiluse + e5-small": ATUAL + ["emb_distiluse", "emb_e5small"],
            "tres embeddings concatenados": ["lr_pal", "svc", "tres embeddings concat", "extremos"],
        })
        np.save(P.RES / "15_oof_emb_distiluse.npy", Z["emb_distiluse"])

    ref, pref = P.cv_meta(Z, ATUAL, y, fold, 0.1)
    linhas = []
    for nome, blocos in opcoes.items():
        for C in ((0.1,) if nome != "ATUAL (MiniLM)" else (0.1, 0.01, 1.0)):
            accs, pred = P.cv_meta(Z, blocos, y, fold, C)
            linhas.append(dict(versao=nome + ("" if C == 0.1 else f" [meta C={C}]"), media=np.mean(accs),
                               **{f"dobra{k+1}": v for k, v in enumerate(accs)},
                               dobras_melhores=int(sum(a > b for a, b in zip(accs, ref))),
                               so_ela=int(((pred == y) & (pref != y)).sum()), so_atual=int(((pred != y) & (pref == y)).sum())))
    df = pd.DataFrame(linhas)
    df.to_csv(P.RES / "15_busca_final.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 50)
    print((df.set_index("versao") * [100] * 6 + [1, 1, 1]).round(2).to_string() if False else
          df.assign(**{k: (df[k] * 100).round(2) for k in ["media"] + [f"dobra{i}" for i in range(1, 6)]}).to_string(index=False))


if __name__ == "__main__":
    main()
