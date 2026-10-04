"""Abordagem 1 - modelos lexicais: TF-IDF + classificadores lineares.

Inclui o baseline oficial (TF-IDF padrao + regressao logistica padrao) e uma
busca em grade sobre representacao (palavras, n-gramas, caracteres) e classificador.
Uso:  python codigo/01_tfidf.py
"""
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from common import load_data, save_test_xlsx, ROOT, N_FOLDS

W1 = dict(ngram_range=(1, 1), min_df=2, sublinear_tf=True)
W2 = dict(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
CH = dict(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)
VECS = {
    "baseline_padrao": [dict()],
    "palavra_1gram": [W1],
    "palavra_1-2gram": [W2],
    "char_2-5gram": [CH],
    "palavra_1-2gram+char_2-5gram": [W2, CH],
}


def make_clfs(vec_name):
    if vec_name == "baseline_padrao":
        return {"LR|C=1": LogisticRegression(max_iter=1000)}
    d = {}
    for C in (0.3, 1, 3):
        d[f"LR|C={C}"] = LogisticRegression(C=C, max_iter=2000)
    for C in (0.03, 0.1, 0.3):
        d[f"LinearSVC|C={C}"] = LinearSVC(C=C)
    for a in (0.1, 0.5):
        d[f"ComplementNB|alpha={a}"] = ComplementNB(alpha=a)
    return d


def featurize(vec_name, fit_texts, other_texts_list):
    vs = [TfidfVectorizer(**kw) for kw in VECS[vec_name]]
    Xfit = sp.hstack([v.fit_transform(fit_texts) for v in vs]).tocsr()
    others = [sp.hstack([v.transform(t) for v in vs]).tocsr() for t in other_texts_list]
    return Xfit, others


def scores(clf, X):
    return clf.predict_proba(X) if hasattr(clf, "predict_proba") else clf.decision_function(X)


def run_fold(vec_name, k, tr):
    t0 = time.time()
    m = (tr["fold"] != k).values
    Xtr, (Xva,) = featurize(vec_name, tr["text"][m], [tr["text"][~m]])
    ytr, yva = tr["y"].values[m], tr["y"].values[~m]
    out = {}
    for name, clf in make_clfs(vec_name).items():
        clf.fit(Xtr, ytr)
        s = scores(clf, Xva)
        out[name] = (float((s.argmax(1) == yva).mean()),
                     float((clf.predict(Xtr) == ytr).mean()), s)
    print(f"  {vec_name} dobra {k}: {Xtr.shape[1]} atributos, {time.time()-t0:.0f}s", flush=True)
    return out, Xtr.shape[1]


def main():
    tr, te = load_data()
    (ROOT / "resultados").mkdir(exist_ok=True)
    rows, oofs = [], {}
    for vec_name in VECS:
        t0 = time.time()
        res = Parallel(n_jobs=2)(delayed(run_fold)(vec_name, k, tr) for k in range(N_FOLDS))
        for name in res[0][0]:
            va = [r[0][name][0] for r in res]
            trn = [r[0][name][1] for r in res]
            oof = np.zeros((len(tr), 3), dtype=np.float32)
            for k, r in enumerate(res):
                oof[(tr["fold"] == k).values] = r[0][name][2]
            oofs[(vec_name, name)] = oof
            rows.append(dict(representacao=vec_name, classificador=name.split("|")[0],
                             parametro=name.split("|")[1],
                             n_atributos=int(np.mean([r[1] for r in res])),
                             acc_val_media=np.mean(va), acc_val_desvio=np.std(va),
                             acc_treino_media=np.mean(trn),
                             **{f"acc_dobra{k}": v for k, v in enumerate(va)}))
        df = pd.DataFrame(rows)
        df.to_csv(ROOT / "resultados" / "01_tfidf_grid.csv", index=False)
        print(f"{vec_name} ok em {time.time()-t0:.0f}s", flush=True)
        print(df[df.representacao == vec_name][["classificador", "parametro", "acc_val_media",
              "acc_val_desvio", "acc_treino_media"]].round(4).to_string(index=False), flush=True)

    df = pd.DataFrame(rows).sort_values("acc_val_media", ascending=False)
    df.to_csv(ROOT / "resultados" / "01_tfidf_grid.csv", index=False)
    best = df.iloc[0]
    key = (best.representacao, f"{best.classificador}|{best.parametro}")
    print("\nMELHOR:", key, round(best.acc_val_media, 4), flush=True)
    np.save(ROOT / "resultados" / "01_tfidf_oof_melhor.npy", oofs[key])

    # Modelo final: retreina a melhor configuracao em todo o treino e rotula o teste
    Xtr, (Xte,) = featurize(best.representacao, tr["text"], [te["text"]])
    clf = make_clfs(best.representacao)[key[1]].fit(Xtr, tr["y"].values)
    s = scores(clf, Xte)
    np.save(ROOT / "resultados" / "01_tfidf_teste_melhor.npy", s)
    out = save_test_xlsx(s.argmax(1), "test1_tfidf.xlsx")
    print("teste rotulado em", out, "| distribuicao:",
          pd.Series(s.argmax(1)).value_counts().sort_index().to_dict(), flush=True)


if __name__ == "__main__":
    main()
