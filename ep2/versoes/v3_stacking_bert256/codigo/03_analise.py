"""EP2 - item 9 do relatorio: analise do modelo.

(i)  interpretacao global: atributos mais importantes de cada classe;
(ii) interpretacao local: por que um exemplo recebeu certa classe;
     ambos com eli5 sobre um classificador simples (regressao logistica em TF-IDF de palavras),
     como pede o modelo de relatorio;
(iii) analise de erros: casos do treino que o modelo final (stacking) errou, com previsoes fora da dobra.

Saidas em ep2/resultados/analise/: tabelas .csv, explicacoes .html (abrir no navegador) e um .txt com os casos.
Uso:  python ep2/codigo/03_analise.py   (depois de 02_stacking_ep2.py)
"""
import re
import warnings

import eli5
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from dados import RES, ROTULOS, SEED, carregar

warnings.filterwarnings("ignore")
OUT = RES / "analise"


def main():
    OUT.mkdir(exist_ok=True)
    d = carregar()
    X, y = d["treino"], d["y"]
    # classificador simples e interpretavel: palavras 1-2-gramas, minusculas, min_df=20 (tira nomes proprios raros)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=20, sublinear_tf=True)
    clf = LogisticRegression(C=1.0, max_iter=1000)

    # ---------- (i) global: pesos por classe, treinado em todo o treino ----------
    Xv = vec.fit_transform(X)
    clf.fit(Xv, y)
    nomes = np.array(vec.get_feature_names_out())
    linhas = []
    for j, r in enumerate(ROTULOS):
        o = np.argsort(clf.coef_[j])
        for pos, i in enumerate(o[::-1][:25]):
            linhas.append(dict(classe=r, sinal="a favor", posicao=pos + 1, termo=nomes[i], peso=round(float(clf.coef_[j][i]), 3)))
        for pos, i in enumerate(o[:15]):
            linhas.append(dict(classe=r, sinal="contra", posicao=pos + 1, termo=nomes[i], peso=round(float(clf.coef_[j][i]), 3)))
    glob = pd.DataFrame(linhas)
    glob.to_csv(OUT / "global_pesos_por_classe.csv", index=False)
    html = eli5.format_as_html(eli5.explain_weights(clf, vec=vec, target_names=ROTULOS, top=25))
    (OUT / "global_eli5.html").write_text(html, encoding="utf-8")
    print("(i) termos com maior peso a favor de cada classe:")
    for r in ROTULOS:
        print(f"  {r}: " + ", ".join(glob[(glob.classe == r) & (glob.sinal == "a favor")].termo.head(15)))

    # ---------- (ii) local: modelo treinado sem a dobra 1, explicando respostas da dobra 1 ----------
    dobra = np.zeros(len(y), dtype=int)
    for k, (_, v) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(y, y)):
        dobra[v] = k
    a, b = np.where(dobra != 0)[0], np.where(dobra == 0)[0]
    vec_l = TfidfVectorizer(ngram_range=(1, 2), min_df=20, sublinear_tf=True)
    clf_l = LogisticRegression(C=1.0, max_iter=1000).fit(vec_l.fit_transform(X[a]), y[a])
    proba = clf_l.predict_proba(vec_l.transform(X[b]))
    pred = proba.argmax(1)
    rng = np.random.RandomState(SEED)
    escolhidos = []
    for j in range(3):                                      # um acerto confiante de cada classe
        cand = b[(pred == j) & (y[b] == j) & (proba.max(1) > 0.6) & (np.char.str_len(X[b].astype(str)) < 900)]
        if len(cand):
            escolhidos.append(("acerto", rng.choice(cand)))
    for j in range(3):                                      # um erro confiante de cada classe prevista
        cand = b[(pred == j) & (y[b] != j) & (proba.max(1) > 0.6) & (np.char.str_len(X[b].astype(str)) < 900)]
        if len(cand):
            escolhidos.append(("erro", rng.choice(cand)))
    partes, resumo = [], []
    pos = {i: k for k, i in enumerate(b)}
    for tipo, i in escolhidos:
        e = eli5.explain_prediction(clf_l, X[i], vec=vec_l, target_names=ROTULOS, top=12)
        partes.append(f"<h3>{tipo}: real {ROTULOS[y[i]]}, previsto {ROTULOS[pred[pos[i]]]} "
                      f"(prob. {proba[pos[i]].max():.2f})</h3>" + eli5.format_as_html(e))
        pesos = eli5.formatters.as_dataframe.format_as_dataframe(e)
        top = pesos[pesos.target == ROTULOS[pred[pos[i]]]].sort_values("weight", ascending=False).head(6)
        resumo.append(dict(linha_treino=int(i), tipo=tipo, real=ROTULOS[y[i]], previsto=ROTULOS[pred[pos[i]]],
                           prob=round(float(proba[pos[i]].max()), 3),
                           termos_que_mais_pesaram=", ".join(f"{t} ({w:+.2f})" for t, w in zip(top.feature, top.weight)),
                           texto=X[i][:400]))
    (OUT / "local_eli5.html").write_text("<meta charset='utf-8'>" + "".join(partes), encoding="utf-8")
    pd.DataFrame(resumo).to_csv(OUT / "local_exemplos.csv", index=False)
    print("\n(ii) exemplos de interpretacao local:")
    for r in resumo:
        print(f"  [{r['tipo']}] real {r['real']} -> previsto {r['previsto']} ({r['prob']}): {r['termos_que_mais_pesaram']}")

    # ---------- (iii) analise de erros do modelo final (previsoes fora da dobra) ----------
    arq = RES / "02_oof_pred_final.npy"
    if not arq.exists():
        print("\n(iii) rode 02_stacking_ep2.py antes para ter as previsoes do modelo final"); return
    pf = np.load(arq)
    limpo = pd.Series(X)
    repet = limpo.map(limpo.value_counts())
    cm = pd.crosstab(pd.Series([ROTULOS[v] for v in y], name="real"), pd.Series([ROTULOS[v] for v in pf], name="previsto"))
    cm.to_csv(OUT / "erros_matriz_confusao.csv")
    print("\n(iii) matriz de confusao do modelo final (fora da dobra):\n" + cm.to_string())
    # textos repetidos com rotulos diferentes: erro inevitavel para qualquer modelo que so ve o texto
    grupos = pd.DataFrame({"t": X, "y": y}).groupby("t").y.agg(["nunique", "size"])
    conflito = grupos[(grupos["nunique"] > 1)]
    lin_conf = limpo.isin(conflito.index).values
    print(f"linhas com texto repetido e rotulos conflitantes: {lin_conf.sum()} | acerto nelas: {(pf[lin_conf] == y[lin_conf]).mean():.4f}"
          f" | nas demais: {(pf[~lin_conf] == y[~lin_conf]).mean():.4f}")
    casos = []
    rng = np.random.RandomState(SEED)
    for real, prev, k in ((0, 2, 3), (2, 0, 3), (1, 2, 3), (1, 0, 2), (0, 1, 2)):
        cand = np.where((y == real) & (pf == prev) & (np.char.str_len(X.astype(str)) < 1200) & ~lin_conf)[0]
        for i in rng.choice(cand, min(k, len(cand)), replace=False):
            casos.append(dict(linha_treino=int(i), real=ROTULOS[real], previsto=ROTULOS[prev], repeticoes=int(repet[i]), texto=X[i]))
    pd.DataFrame(casos).to_csv(OUT / "erros_casos.csv", index=False)
    with open(OUT / "erros_casos.txt", "w", encoding="utf-8") as f:
        for c in casos:
            f.write(f"### linha {c['linha_treino']} | real {c['real']} | previsto {c['previsto']} | repeticoes {c['repeticoes']}\n{c['texto']}\n\n")
    print(f"{len(casos)} casos de erro gravados em resultados/analise/erros_casos.txt")


if __name__ == "__main__":
    main()
