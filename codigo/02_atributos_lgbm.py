"""Abordagem 2 - atributos manuais + gradient boosting (LightGBM).

Nenhuma palavra entra como atributo direto: o modelo so ve medidas de tamanho,
estrutura, legibilidade, marcadores de conteudo (negativa, anexo, link, lei...),
orgao/ano do protocolo e repeticao de modelos de resposta (ver atributos.py).
Uso:  python codigo/02_atributos_lgbm.py
"""
import itertools
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from atributos import CATEGORICAS, tabela_atributos
from common import load_data, save_test_xlsx, ROOT, N_FOLDS

warnings.filterwarnings("ignore")
RES = ROOT / "resultados"
ARVORES = [100, 200, 300, 400, 600]
GRADE = {
    "learning_rate": [0.03],
    "num_leaves": [7, 15, 31, 63],
    "min_child_samples": [20, 60],
    "colsample_bytree": [0.3, 0.6],
}
FIXOS = dict(subsample=0.7, subsample_freq=1, reg_lambda=1.0)


def grupo_de(col):
    if col.startswith("m_") or col in ("saudacao_com_nome", "comeca_com_prezado"):
        return "marcadores de conteudo"
    if col in ("palavras_por_frase", "letras_por_palavra", "silabas_por_palavra", "flesch_pt",
               "frac_palavras_longas", "riqueza_lexical", "zipf_medio", "zipf_p10",
               "frac_palavras_raras", "frac_palavras_desconhecidas"):
        return "legibilidade e vocabulario"
    if col in ("tem_protocolo", "orgao_protocolo", "ano_protocolo", "ano_max", "ano_min",
               "n_anos_citados", "n_siglas", "sigla_principal", "repeticoes_do_texto",
               "repeticoes_do_inicio", "repeticoes_do_final"):
        return "orgao, datas e repeticao"
    return "tamanho e estrutura"


def modelo(params, n_estimators=max(ARVORES)):
    return lgb.LGBMClassifier(objective="multiclass", n_estimators=n_estimators, verbose=-1,
                              n_jobs=1, random_state=42, **FIXOS, **params)


def fold_lgbm(params, Xa, ya, Xb, yb, cats):
    m = modelo(params).fit(Xa, ya, categorical_feature=cats)
    out = {}
    for k in ARVORES:
        p = m.predict_proba(Xb, num_iteration=k)
        pa = m.predict_proba(Xa, num_iteration=k)
        out[k] = (float((p.argmax(1) == yb).mean()), float((pa.argmax(1) == ya).mean()), p)
    return out


def cv_lgbm(params, A, y, fold, cols=None):
    cols = list(A.columns) if cols is None else cols
    cats = [c for c in CATEGORICAS if c in cols]
    X = A[cols]
    res = Parallel(n_jobs=2)(delayed(fold_lgbm)(params, X[fold != k], y[fold != k],
                                                X[fold == k], y[fold == k], cats)
                             for k in range(N_FOLDS))
    saida = {}
    for n in ARVORES:
        oof = np.zeros((len(y), 3), dtype=np.float32)
        for k in range(N_FOLDS):
            oof[fold == k] = res[k][n][2]
        saida[n] = ([r[n][0] for r in res], [r[n][1] for r in res], oof)
    return saida


def main():
    tr, te = load_data()
    y, fold = tr["y"].values, tr["fold"].values
    N = len(tr)
    # os atributos usam o texto ORIGINAL (com os espacos duplos que marcam quebras de linha)
    bruto = [str(t) for t in pd.read_excel(ROOT / "train.xlsx")["resp_text"]] + \
            [str(t) for t in pd.read_excel(ROOT / "test1.xlsx")["resp_text"]]
    t0 = time.time()
    T = tabela_atributos(bruto)
    A, Ate = T.iloc[:N].reset_index(drop=True), T.iloc[N:].reset_index(drop=True)
    print(f"{A.shape[1]} atributos calculados em {time.time()-t0:.0f}s", flush=True)

    # ---------- 1) busca em grade do LightGBM ----------
    linhas, melhor = [], None
    for vals in itertools.product(*GRADE.values()):
        params = dict(zip(GRADE, vals))
        t0 = time.time()
        for n, (va, trn, oof) in cv_lgbm(params, A, y, fold).items():
            l = dict(**params, n_estimators=n, acc_val_media=np.mean(va), acc_val_desvio=np.std(va),
                     acc_treino_media=np.mean(trn), **{f"acc_dobra{k}": v for k, v in enumerate(va)})
            linhas.append(l)
            if melhor is None or l["acc_val_media"] > melhor[0]["acc_val_media"]:
                melhor = (l, params, n, oof)
        pd.DataFrame(linhas).to_csv(RES / "02_atributos_grid.csv", index=False)
        b = max(linhas[-len(ARVORES):], key=lambda r: r["acc_val_media"])
        print(f"{params} -> melhor n={b['n_estimators']} val={b['acc_val_media']:.4f} "
              f"treino={b['acc_treino_media']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(linhas).sort_values("acc_val_media", ascending=False).to_csv(
        RES / "02_atributos_grid.csv", index=False)
    l, params, n, oof = melhor
    print(f"\nMELHOR LightGBM: {params} n_estimators={n} val={l['acc_val_media']:.4f}", flush=True)
    np.save(RES / "02_atributos_oof_melhor.npy", oof)

    # ---------- 2) quanto cada grupo de atributos consegue sozinho ----------
    grupos = {}
    for c in A.columns:
        grupos.setdefault(grupo_de(c), []).append(c)
    gl = [dict(grupo="todos os atributos", n_atributos=A.shape[1],
               acc_val_media=l["acc_val_media"], acc_val_desvio=l["acc_val_desvio"])]
    for g, cols in grupos.items():
        va = cv_lgbm(params, A, y, fold, cols)[n][0]
        gl.append(dict(grupo="so " + g, n_atributos=len(cols), acc_val_media=np.mean(va),
                       acc_val_desvio=np.std(va)))
        print(gl[-1], flush=True)

    # ---------- 3) referencia linear: regressao logistica nos mesmos atributos ----------
    num = [c for c in A.columns if c not in CATEGORICAS]
    for C in (0.01, 0.1, 1.0):
        va = []
        for k in range(N_FOLDS):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
            m.fit(A.loc[fold != k, num], y[fold != k])
            va.append(float((m.predict(A.loc[fold == k, num]) == y[fold == k]).mean()))
        gl.append(dict(grupo=f"regressao logistica (C={C}) nos atributos numericos",
                       n_atributos=len(num), acc_val_media=np.mean(va), acc_val_desvio=np.std(va)))
        print(gl[-1], flush=True)
    pd.DataFrame(gl).to_csv(RES / "02_atributos_grupos.csv", index=False)

    # ---------- 4) modelo final: treino completo, importancia e teste ----------
    m = modelo(params, n).fit(A, y, categorical_feature=CATEGORICAS)
    imp = pd.DataFrame({"atributo": A.columns, "grupo": [grupo_de(c) for c in A.columns],
                        "ganho": m.booster_.feature_importance(importance_type="gain")})
    imp["ganho_%"] = (100 * imp["ganho"] / imp["ganho"].sum()).round(2)
    imp = imp.sort_values("ganho", ascending=False).drop(columns="ganho")
    imp.to_csv(RES / "02_atributos_importancia.csv", index=False)
    print(imp.head(20).to_string(index=False))
    print(imp.groupby("grupo")["ganho_%"].sum().round(1).to_string(), flush=True)
    p = m.predict_proba(Ate)
    np.save(RES / "02_atributos_teste_melhor.npy", p)
    out = save_test_xlsx(p.argmax(1), "test1_atributos.xlsx")
    print("teste rotulado em", out, "| distribuicao:",
          pd.Series(p.argmax(1)).value_counts().sort_index().to_dict(), flush=True)
    A.assign(y=y, fold=fold).to_pickle(ROOT / "dados" / "atributos_treino.pkl")


if __name__ == "__main__":
    main()
