"""Junta o BERTimbau do Dimitri ao stacking e testa trocar as arvores de legibilidade por LightGBM.

O BERT nao e retreinado. Como ele ja viu parte do treino, tudo e medido so nas linhas do holdout
de 20% (semente 123) em que a inferencia ja terminou (resultados/13_bert_logits.npz):
  - modelos-base do stacking e arvores: treinados nos 80%, aplicados ao holdout;
  - meta-classificador com BERT: validacao cruzada de 5 dobras DENTRO do holdout.
Grava a planilha candidata em predicoes/test1_stacking_bert.xlsx (nao mexe na entrega).
Uso:  python codigo/14_stacking_com_bert.py
"""
import importlib
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

import pipeline as P
from common import LABELS, save_test_xlsx

sys.path.insert(0, str(P.ROOT / "resposta_dimitri"))
FINAL = importlib.import_module("08_final").FINAL


def atributos_dimitri(textos):
    from clarity_classifier import ClarityHybridClassifier
    return ClarityHybridClassifier.extract_linguistic_features(textos)


def acc(p, y):
    return float((p.argmax(1) == y).mean())


def cv_dentro(X, y, C=0.1):
    """Validacao cruzada de 5 dobras do meta-classificador dentro do proprio conjunto."""
    pred = np.zeros(len(y), dtype=int)
    for i, j in StratifiedKFold(5, shuffle=True, random_state=P.SEED).split(X, y):
        pred[j] = P.meta(C).fit(X[i], y[i]).predict(X[j])
    return pred


def main():
    c = P.Corpus()
    n, y = c.n_treino, c.y
    todos, teste = np.arange(n), np.arange(n, len(c.texto))
    a, b = train_test_split(todos, test_size=0.2, random_state=P.SEED, stratify=y)
    z = np.load(P.RES / "13_bert_logits.npz")
    logit = dict(zip(z["ids"].tolist(), z["logits"]))
    S = np.array([i for i in b if i in logit])                    # holdout ja processado pelo BERT
    D = np.array([i for i in a if i in logit])                    # amostra dos 80% (diagnostico)
    pos = {i: k for k, i in enumerate(b)}
    sb = np.array([pos[i] for i in S])                            # posicao de S dentro de b
    yS = y[S]
    LS = np.array([logit[i] for i in S])
    print(f"holdout com BERT pronto: {len(S)} de {len(b)} linhas | teste pronto: {sum(i in logit for i in teste)} de {len(teste)}")
    if len(D):
        LD = np.array([logit[i] for i in D])
        print(f"BERT sozinho: {acc(LD, y[D]):.4f} em {len(D)} linhas dos 80% de treino x {acc(LS, yS):.4f} no holdout"
              " (se o primeiro for muito maior, o BERT foi treinado nos 80% e o holdout esta limpo)")
    pb = softmax(LS / 1.5, axis=1)                                # mesma temperatura do codigo do Dimitri

    # ---------- arvores de legibilidade: original x LightGBM ----------
    F = atributos_dimitri(list(c.bruto))
    Fa, FS = F.iloc[a], F.iloc[S]
    hgb = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=31, min_samples_leaf=20,
                                         class_weight="balanced", random_state=P.SEED).fit(Fa, y[a])
    kw = dict(objective="multiclass", learning_rate=0.03, num_leaves=31, min_child_samples=60, subsample=0.7,
              subsample_freq=1, reg_lambda=1.0, verbose=-1, n_jobs=4, random_state=P.SEED)
    lg14 = lgb.LGBMClassifier(n_estimators=150, colsample_bytree=0.6, **kw).fit(Fa, y[a])
    Za, Zb, _ = P.empilhar(c, a, b, cache="holdout", verbose=False)
    arvores = {
        "HistGradientBoosting, 14 atributos (original do Dimitri)": hgb.predict_proba(FS),
        "LightGBM, 14 atributos": lg14.predict_proba(FS),
        "LightGBM, 87 atributos (nossos)": np.exp(Zb["lgbm"][sb]),
    }
    linhas = [dict(modelo="BERTimbau sozinho", acc=acc(pb, yS))]
    for nome, pt in arvores.items():
        linhas.append(dict(modelo=f"arvore sozinha: {nome}", acc=acc(pt, yS)))
        linhas.append(dict(modelo=f"60% BERT + 40% {nome}", acc=acc(0.6 * pb + 0.4 * pt, yS)))

    # ---------- stacking ----------
    nosso = P.juntar(Zb, FINAL)[sb]
    p_atual = P.meta(0.1).fit(P.juntar(Za, FINAL), y[a]).predict(P.juntar(Zb, FINAL))[sb]
    linhas.append(dict(modelo="NOSSO STACKING (entrega atual, meta treinado nos 80%)", acc=float((p_atual == yS).mean())))
    lt = np.log(arvores["LightGBM, 87 atributos (nossos)"] + 1e-6)
    lh = np.log(arvores["HistGradientBoosting, 14 atributos (original do Dimitri)"] + 1e-6)
    opcoes = {
        "stacking so com os nossos blocos (meta por CV dentro do holdout)": nosso,
        "stacking: nossos blocos + BERT": np.hstack([nosso, LS]),
        "stacking: nossos blocos + BERT + arvores originais": np.hstack([nosso, LS, lh]),
        "stacking: nossos blocos + BERT + LightGBM 87": np.hstack([nosso, LS, lt]),
        "stacking: so BERT + LightGBM 87": np.hstack([LS, lt]),
    }
    preds = {}
    for nome, X in opcoes.items():
        preds[nome] = cv_dentro(X, yS)
        linhas.append(dict(modelo=nome, acc=float((preds[nome] == yS).mean())))
    df = pd.DataFrame(linhas)
    df["erro_padrao"] = np.sqrt(df.acc * (1 - df.acc) / len(S))
    df.to_csv(P.RES / "14_stacking_com_bert.csv", index=False)
    pd.set_option("display.width", 200, "display.max_colwidth", 90)
    print(df.round(4).to_string(index=False))
    ref, com = preds["stacking so com os nossos blocos (meta por CV dentro do holdout)"], preds["stacking: nossos blocos + BERT"]
    print(f"\nrespostas em que so um acerta: com BERT {int(((com == yS) & (ref != yS)).sum())} x sem BERT {int(((com != yS) & (ref == yS)).sum())}")
    print(f"com BERT x entrega atual: {int(((com == yS) & (p_atual != yS)).sum())} x {int(((com != yS) & (p_atual == yS)).sum())}")

    # ---------- planilha candidata: nossos blocos + BERT ----------
    if all(i in logit for i in teste):
        Zt, Zte, _ = P.empilhar(c, todos, teste, cache="completo", verbose=False)
        Lte = np.array([logit[i] for i in teste])
        m = P.meta(0.1).fit(np.hstack([nosso, LS]), yS)
        pt = m.predict(np.hstack([P.juntar(Zte, FINAL), Lte]))
        out = save_test_xlsx(pt, "test1_stacking_bert.xlsx")
        atual = pd.read_excel(P.ROOT / "entrega" / "test1.xlsx")["clarity"].map({l: i for i, l in enumerate(LABELS)}).values
        print(f"planilha candidata em predicoes/{out.name} | distribuicao:",
              {LABELS[i]: int(q) for i, q in enumerate(np.bincount(pt, minlength=3))},
              f"| concordancia com a entrega atual: {(pt == atual).mean():.3f}")


if __name__ == "__main__":
    main()
