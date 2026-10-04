"""Modelo final - combinacao (stacking) dos melhores modelos das abordagens 1, 2 e 3.

Cada modelo-base ja tem previsoes "fora da dobra" para todo o treino (feitas sem ver a propria
linha). Uma regressao logistica (meta-classificador) aprende a combinar essas previsoes.
Opcionalmente entra tambem a "memoria de respostas repetidas": como as outras linhas do treino
com texto identico foram rotuladas.
Uso:  python codigo/04_ensemble.py   (depois de rodar 01, 02, 03_embeddings e 03_textcnn)
"""
import itertools
import shutil
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import load_data, save_test_xlsx, ROOT, N_FOLDS, LABELS

RES = ROOT / "resultados"
BASES = {"tfidf": "01_tfidf", "atributos": "02_atributos", "lsa": "03_embeddings", "textcnn": "03_textcnn"}
SAO_PROB = {"atributos", "lsa", "textcnn"}          # tfidf (LinearSVC) devolve pontuacoes, nao probabilidades
GRADE_C = [0.01, 0.1, 1.0]


def carregar_bases():
    oof, tst = {}, {}
    for nome, pref in BASES.items():
        a, b = np.load(RES / f"{pref}_oof_melhor.npy"), np.load(RES / f"{pref}_teste_melhor.npy")
        if nome in SAO_PROB:
            a, b = np.log(a + 1e-6), np.log(b + 1e-6)
        oof[nome], tst[nome] = a.astype(np.float64), b.astype(np.float64)
    return oof, tst


def memoria(tr, te):
    """Proporcao de cada rotulo entre as OUTRAS linhas do treino com texto identico (+ quantas sao)."""
    cont = defaultdict(lambda: np.zeros(3))
    for t, c in zip(tr["text"], tr["y"]):
        cont[t][c] += 1

    def linha(v):
        n = v.sum()
        return np.r_[v / n if n > 0 else np.zeros(3), np.log1p(n)]

    Mtr = np.array([linha(cont[t] - np.eye(3)[c]) for t, c in zip(tr["text"], tr["y"])])
    Mte = np.array([linha(cont[t]) if t in cont else linha(np.zeros(3)) for t in te["text"]])
    return Mtr, Mte


def montar(nomes, usa_mem, B, M):
    return np.hstack([B[n] for n in nomes] + ([M] if usa_mem else []))


def meta(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def cv(X, y, fold, C, dobras):
    """Acuracia por dobra: treina o meta-classificador nas outras dobras da lista e avalia em cada uma."""
    accs, pred = [], np.zeros(len(y), dtype=int)
    usar = np.isin(fold, dobras)
    for k in dobras:
        a, b = usar & (fold != k), fold == k
        pred[b] = meta(C).fit(X[a], y[a]).predict(X[b])
        accs.append(float((pred[b] == y[b]).mean()))
    return accs, pred


def configs():
    nomes = list(BASES)
    for r in range(1, len(nomes) + 1):
        for sub in itertools.combinations(nomes, r):
            for mem in (False, True):
                for C in GRADE_C:
                    yield sub, mem, C


def aninhada(k, oof, Mtr, y, fold):
    """Dobra externa k: escolhe a configuracao so com as outras 4 dobras e avalia na k."""
    internas = [j for j in range(N_FOLDS) if j != k]
    melhor = None
    for sub, mem, C in configs():
        X = montar(sub, mem, oof, Mtr)
        a = np.mean(cv(X, y, fold, C, internas)[0])
        if melhor is None or a > melhor[0]:
            melhor = (a, sub, mem, C)
    _, sub, mem, C = melhor
    X = montar(sub, mem, oof, Mtr)
    tr_, va_ = fold != k, fold == k
    acc = float((meta(C).fit(X[tr_], y[tr_]).predict(X[va_]) == y[va_]).mean())
    return dict(dobra=k, modelos=" + ".join(sub), memoria=mem, C=C, acc=acc)


def main():
    t0 = time.time()
    tr, te = load_data()
    y, fold = tr["y"].values, tr["fold"].values
    oof, tst = carregar_bases()
    Mtr, Mte = memoria(tr, te)
    todas = list(range(N_FOLDS))
    print("linhas do treino com outra linha de texto identico:", int((Mtr[:, 3] > 0).sum()),
          "| linhas do teste com texto identico no treino:", int((Mte[:, 3] > 0).sum()), flush=True)

    # ---------- 1) todas as combinacoes ----------
    linhas, preds = [], {}
    for sub, mem, C in configs():
        accs, pred = cv(montar(sub, mem, oof, Mtr), y, fold, C, todas)
        linhas.append(dict(modelos=" + ".join(sub), n_modelos=len(sub), memoria=mem, C=C,
                           acc_val_media=np.mean(accs), acc_val_desvio=np.std(accs),
                           **{f"acc_dobra{k}": a for k, a in enumerate(accs)}))
        preds[(sub, mem, C)] = pred
    df = pd.DataFrame(linhas).sort_values("acc_val_media", ascending=False)
    df.to_csv(RES / "04_ensemble_grid.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.head(12)[["modelos", "memoria", "C", "acc_val_media", "acc_val_desvio"]].round(4).to_string(index=False))
    print("\nmelhor por (n_modelos, memoria):")
    print(df.groupby(["n_modelos", "memoria"]).acc_val_media.max().round(4).to_string(), flush=True)

    # ---------- 2) estimativa honesta: validacao cruzada aninhada ----------
    an = pd.DataFrame(Parallel(n_jobs=2)(delayed(aninhada)(k, oof, Mtr, y, fold) for k in todas))
    an.to_csv(RES / "04_ensemble_aninhada.csv", index=False)
    print("\nvalidacao aninhada:\n", an.round(4).to_string(index=False))
    print(f"acuracia aninhada: {an.acc.mean():.4f} +- {an.acc.std(ddof=0):.4f}", flush=True)

    # ---------- 3) modelo final ----------
    b = df.iloc[0]
    sub, mem, C = tuple(b.modelos.split(" + ")), bool(b.memoria), float(b.C)
    pred = preds[(sub, mem, C)]
    np.save(RES / "04_ensemble_oof_pred.npy", pred)
    rep = Mtr[:, 3] > 0
    print(f"\nFINAL: {sub} memoria={mem} C={C} -> {b.acc_val_media:.4f}")
    print(f"  acuracia em linhas com texto repetido ({rep.sum()}): {(pred[rep] == y[rep]).mean():.4f}"
          f" | demais ({(~rep).sum()}): {(pred[~rep] == y[~rep]).mean():.4f}")
    m = meta(C).fit(montar(sub, mem, oof, Mtr), y)
    Xte = montar(sub, mem, tst, Mte)
    pte = m.predict_proba(Xte)
    np.save(RES / "04_ensemble_teste_proba.npy", pte)
    out = save_test_xlsx(pte.argmax(1), "test1_ensemble.xlsx")
    (ROOT / "entrega").mkdir(exist_ok=True)
    shutil.copy(out, ROOT / "entrega" / "test1.xlsx")
    print("teste rotulado em entrega/test1.xlsx | distribuicao:",
          {LABELS[i]: int(n) for i, n in enumerate(np.bincount(pte.argmax(1), minlength=3))})
    coef = pd.DataFrame(m[-1].coef_, index=LABELS,
                        columns=[f"{n}_{l}" for n in sub for l in LABELS] +
                                ([f"mem_{l}" for l in LABELS] + ["mem_qtd"] if mem else []))
    coef.round(3).T.to_csv(RES / "04_ensemble_coeficientes.csv")
    print(coef.round(2).T.to_string())
    print(f"tempo total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
