"""Comparacao: so TF-IDF + SVM (com grade de C e de representacao) x modelo final, semente 123.

Mesmas divisoes do baseline oficial: holdout 80/20 e 5 dobras estratificadas.
Uso:  python codigo/09_so_tfidf_svm.py   (depois de 07_avaliar.py e 08_final.py)
"""
import importlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

import pipeline as P

FINAL = importlib.import_module("08_final").FINAL


def main():
    c = P.Corpus()
    n, y = c.n_treino, c.y
    todos = np.arange(n)
    a, b = train_test_split(todos, test_size=0.2, random_state=P.SEED, stratify=y)
    fold = np.zeros(n, dtype=int)
    for k, (_, v) in enumerate(P.dobras(y)):
        fold[v] = k

    linhas, oof_svm = [], {}
    for rep, M in (("palavras 1-2", c.tfidf_pal), ("palavras 1-2 + caracteres 2-5", c.tfidf)):
        for C in (0.01, 0.03, 0.1, 0.3, 1.0):
            pred = np.zeros(n, dtype=int)
            for k in range(P.N_FOLDS):
                m = LinearSVC(C=C, random_state=P.SEED).fit(M[todos[fold != k]], y[fold != k])
                pred[fold == k] = m.predict(M[todos[fold == k]])
            accs = [float((pred[fold == k] == y[fold == k]).mean()) for k in range(P.N_FOLDS)]
            ph = LinearSVC(C=C, random_state=P.SEED).fit(M[a], y[a]).predict(M[b])
            oof_svm[(rep, C)] = pred
            linhas.append(dict(modelo=f"so TF-IDF + SVM | {rep} | C={C}", cv=np.mean(accs), cv_desvio=np.std(accs),
                               holdout=float((ph == y[b]).mean())))
            print(linhas[-1], flush=True)

    # modelo final, a partir dos blocos ja calculados
    Zt, _, _ = P.empilhar(c, todos, np.arange(n, len(c.texto)), cache="completo", verbose=False)
    Za, Zb, _ = P.empilhar(c, a, b, cache="holdout", verbose=False)
    accs, pred_final = P.cv_meta(Zt, FINAL, y, fold, 0.1)
    ph = P.meta(0.1).fit(P.juntar(Za, FINAL), y[a]).predict(P.juntar(Zb, FINAL))
    linhas.append(dict(modelo="MODELO FINAL (stacking)", cv=np.mean(accs), cv_desvio=np.std(accs),
                       holdout=float((ph == y[b]).mean())))
    df = pd.DataFrame(linhas)
    df.to_csv(P.RES / "09_so_tfidf_svm.csv", index=False)
    pd.set_option("display.width", 200, "display.max_colwidth", 70)
    print(df.round(4).to_string(index=False))

    melhor = max(oof_svm, key=lambda k: (oof_svm[k] == y).mean())
    ps = oof_svm[melhor]
    print(f"\nmelhor SVM na validacao cruzada: {melhor}")
    for k in range(P.N_FOLDS):
        f = fold == k
        print(f"  dobra {k}: SVM {(ps[f] == y[f]).mean():.4f} | final {(pred_final[f] == y[f]).mean():.4f}")
    print(f"  respostas em que so um acerta: final {int(((pred_final == y) & (ps != y)).sum())} x "
          f"SVM {int(((pred_final != y) & (ps == y)).sum())}")
    for j, l in enumerate(P.LABELS):
        print(f"  acerto na classe {l}: SVM {(ps[y == j] == j).mean():.4f} | final {(pred_final[y == j] == j).mean():.4f}")


if __name__ == "__main__":
    main()
