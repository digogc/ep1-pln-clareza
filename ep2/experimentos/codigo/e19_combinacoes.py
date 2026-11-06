"""Experimento 19 (03/11, pedido do usuario): novas combinacoes no stacking, so com blocos ja prontos (sem retreino).

Compara com a v3 (v2 + BERT 256), nas mesmas 5 dobras: outros blocos (BERT 128/512, snapshots da epoca 1, Qwen3,
atributos, kNN), outras formas de entrar com o BERT (escore x 3 logits x media dos BERTs), a forca da regularizacao
do meta-classificador, um meta nao linear (gradient boosting) e ajuste dos vieses de classe para a acuracia.

Como sao muitas configuracoes, a melhor media na validacao cruzada e otimista (escolha pelo maximo). Por isso o
script tambem estima, com validacao cruzada aninhada, quanto rende o PROCEDIMENTO "escolher a melhor configuracao":
em cada dobra externa, a escolha e feita so com as outras 4 dobras e avaliada na dobra que ficou de fora.
Regra fixada antes de rodar: troca a v3 so se a nova configuracao (a) ganhar pelo menos 0,2 ponto na media,
(b) ganhar em pelo menos 4 das 5 dobras e (c) o aninhado nao ficar abaixo da v3.
So usa o treino; nada do test1/test2.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")              # nao atrapalhar o BERT que treina ao lado

import numpy as np                                          # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import StratifiedKFold        # noqa: E402

from comum import SAIDA, SEED, V2, blocos_v2, carregar, carregar_bloco, dobras, fmt, meta  # noqa: E402
from e05_avaliar import bloco_bert                          # noqa: E402

y = carregar()["y"]
n, dobra = len(y), dobras(y)
F, _, _ = blocos_v2()
esc = lambda o: (o[:, 2] - o[:, 0])[:, None]               # noqa: E731
LOG = {}
for nome, chave in [("bert5d", "b128"), ("bert256_5d", "b256"), ("bert512_5d", "b512"),
                    ("bert5d_ep1", "b128e1"), ("bert256_5d_ep1", "b256e1")]:
    LOG[chave] = bloco_bert(nome, n)[0]
    F[chave] = esc(LOG[chave])
    F[chave + "_log"] = LOG[chave]
F["b256+512"] = esc(LOG["b256"] + LOG["b512"])            # media dos logits de dois BERTs, um bloco so
F["b128+256+512"] = esc(LOG["b128"] + LOG["b256"] + LOG["b512"])
for nome in ["emb_qwen3", "atributos", "knn_minilm", "knn_tfidf"]:
    F[nome] = carregar_bloco(nome)[0]

V3 = V2 + ["b256"]
sem = lambda *tira: [b for b in V3 if b not in tira]      # noqa: E731
CONFIGS = {
    "v3 (referencia)": (V3, "lr", 0.1),
    "v3, C=0,01": (V3, "lr", 0.01), "v3, C=0,03": (V3, "lr", 0.03),
    "v3, C=0,3": (V3, "lr", 0.3), "v3, C=1": (V3, "lr", 1.0),
    "v3 + BERT 512": (V3 + ["b512"], "lr", 0.1),
    "v3 + BERT 128": (V3 + ["b128"], "lr", 0.1),
    "v3 + BERT 128 + 512": (V3 + ["b128", "b512"], "lr", 0.1),
    "v3 + epoca 1 do 256": (V3 + ["b256e1"], "lr", 0.1),
    "v3 + epocas 1 do 128 e 256": (V3 + ["b128e1", "b256e1"], "lr", 0.1),
    "v3, BERT 256 com 3 logits": (sem("b256") + ["b256_log"], "lr", 0.1),
    "v2 + media 256/512": (V2 + ["b256+512"], "lr", 0.1),
    "v2 + media 128/256/512": (V2 + ["b128+256+512"], "lr", 0.1),
    "v3 sem o BERT da v2": (sem("bert"), "lr", 0.1),
    "v3 sem o MiniLM": (sem("emb"), "lr", 0.1),
    "v3 com Qwen3 no lugar do MiniLM": (sem("emb") + ["emb_qwen3"], "lr", 0.1),
    "v3 + Qwen3": (V3 + ["emb_qwen3"], "lr", 0.1),
    "v3 + atributos": (V3 + ["atributos"], "lr", 0.1),
    "v3 + kNN MiniLM": (V3 + ["knn_minilm"], "lr", 0.1),
    "v3 + kNN TF-IDF": (V3 + ["knn_tfidf"], "lr", 0.1),
    "v3 + todos os extras": (V3 + ["b128", "b512", "emb_qwen3", "atributos", "knn_minilm", "knn_tfidf"], "lr", 0.1),
    "v3, vieses de classe ajustados": (V3, "lr_vies", 0.1),
    "v3, gradient boosting": (V3, "hgb", 0.1),
    "todos os blocos, gradient boosting": (V3 + ["b128", "b512", "emb_qwen3", "atributos", "knn_minilm", "knn_tfidf"], "hgb", 0.1),
}


def ajustar(tipo, C, X, yy):
    if tipo == "hgb":
        return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=100,
                                              l2_regularization=1.0, random_state=SEED).fit(X, yy)
    m = meta(C).fit(X, yy)
    if tipo == "lr_vies":                     # vieses escolhidos em validacao cruzada interna, so no treino da dobra
        P = np.zeros((len(yy), 3))
        for a, b in StratifiedKFold(4, shuffle=True, random_state=SEED).split(X, yy):
            P[b] = meta(C).fit(X[a], yy[a]).predict_log_proba(X[b])
        grade = np.arange(-0.3, 0.31, 0.02)
        melhor = max(((v0, v2) for v0 in grade for v2 in grade),
                     key=lambda v: ((P + [v[0], 0, v[1]]).argmax(1) == yy).mean())
        m.vies = np.array([melhor[0], 0, melhor[1]])
    return m


def prever(m, X):
    if hasattr(m, "vies"):
        return (m.predict_log_proba(X) + m.vies).argmax(1)
    return m.predict(X)


def cv(nome, idx=None, nd=None):
    """Acuracia por dobra. Com idx/nd: validacao cruzada so nas linhas idx, com as dobras nd."""
    blocos, tipo, C = CONFIGS[nome]
    X = np.hstack([F[b] for b in blocos])
    idx = np.arange(n) if idx is None else idx
    nd = dobra if nd is None else nd
    X, yy = X[idx], y[idx]
    return np.array([(prever(ajustar(tipo, C, X[nd != k], yy[nd != k]), X[nd == k]) == yy[nd == k]).mean()
                     for k in np.unique(nd)])


def main():
    res = {}
    print(f"{'configuracao':38s} media | dobras 1-5   (dobras melhores que a v3)")
    for nome in CONFIGS:
        res[nome] = cv(nome)
        ganha = int((res[nome] > res["v3 (referencia)"]).sum())
        print(f"{nome:38s} {fmt(res[nome])}   ({ganha}/5)", flush=True)
    # validacao cruzada aninhada do procedimento "escolher a configuracao com a maior media"
    acc, escolhas = [], []
    for k in range(5):
        fora = np.where(dobra != k)[0]
        medias = {nome: cv(nome, fora, dobra[fora]).mean() for nome in CONFIGS}
        melhor = max(medias, key=medias.get)
        blocos, tipo, C = CONFIGS[melhor]
        X = np.hstack([F[b] for b in blocos])
        m = ajustar(tipo, C, X[dobra != k], y[dobra != k])
        acc.append((prever(m, X[dobra == k]) == y[dobra == k]).mean())
        escolhas.append(melhor)
        print(f"  aninhado, dobra {k+1}: escolheu '{melhor}' -> {acc[-1]*100:.2f}", flush=True)
    acc = np.array(acc)
    ref = res["v3 (referencia)"]
    print(f"\nprocedimento 'escolher a melhor' (aninhado): {fmt(acc)}  | v3: {fmt(ref)}")
    melhor = max(res, key=lambda k: res[k].mean())
    ok = (res[melhor].mean() - ref.mean() >= 0.002 and (res[melhor] > ref).sum() >= 4 and acc.mean() >= ref.mean())
    print(f"melhor na validacao cruzada: '{melhor}' ({res[melhor].mean()*100:.2f})")
    print(f"decisao pela regra fixada: {'trocar a v3 por ' + repr(melhor) if ok else 'manter a v3'}")


if __name__ == "__main__":
    main()
