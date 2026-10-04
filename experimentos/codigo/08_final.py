"""Modelo final: stacking de quatro modelos-base, semente 123. Rotula o teste.

    lr_pal       regressao logistica sobre TF-IDF de palavras (1-2-gramas)
    svc          LinearSVC sobre TF-IDF de palavras (1-2) + caracteres (2-5)
    emb_minilm   regressao logistica sobre embeddings de sentenca (paraphrase-multilingual-MiniLM-L12-v2)
    extremos     classificador c1 x c5 treinado so nos extremos (eixo "ruim-bom")
Uso:  python codigo/08_final.py     (depois de 05_embeddings_sentenca.py)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import pipeline as P
from common import LABELS, save_test_xlsx

FINAL = ["lr_pal", "svc", "emb_minilm", "extremos"]      # versao anterior (v2) usava emb_e5small
C_META = 0.1


def main():
    c = P.Corpus()
    todos, teste = np.arange(c.n_treino), np.arange(c.n_treino, len(c.texto))
    Za, Zb, fold = P.empilhar(c, todos, teste, cache="completo")
    accs, pred = P.cv_meta(Za, FINAL, c.y, fold, C_META)
    base = np.zeros(c.n_treino, dtype=int)
    for k in range(P.N_FOLDS):
        base[fold == k] = P.baseline_oficial(c, todos[fold != k], todos[fold == k])
    acc_b = [float((base[fold == k] == c.y[fold == k]).mean()) for k in range(P.N_FOLDS)]
    print(f"\nMODELO FINAL {FINAL}, meta C={C_META}")
    print(f"  validacao cruzada: {np.mean(accs):.4f} +- {np.std(accs):.4f} | por dobra {np.round(accs, 4)}")
    print(f"  baseline oficial : {np.mean(acc_b):.4f} +- {np.std(acc_b):.4f} | por dobra {np.round(acc_b, 4)}")
    print(f"  diferenca por dobra: {np.round(np.array(accs) - np.array(acc_b), 4)}")
    print(f"  linhas em que so um acerta: modelo {int(((pred == c.y) & (base != c.y)).sum())} x "
          f"baseline {int(((pred != c.y) & (base == c.y)).sum())}")
    print("  matriz de confusao (linhas = real, colunas = previsto):")
    print(pd.DataFrame(confusion_matrix(c.y, pred), index=LABELS, columns=LABELS).to_string())
    pd.DataFrame(dict(dobra=range(P.N_FOLDS), modelo_final=accs, baseline_oficial=acc_b)).to_csv(
        P.RES / "08_final_dobras.csv", index=False)
    np.save(P.RES / "08_final_oof_pred.npy", pred)

    m = P.meta(C_META).fit(P.juntar(Za, FINAL), c.y)
    proba = m.predict_proba(P.juntar(Zb, FINAL))
    np.save(P.RES / "08_final_teste_proba.npy", proba)
    out = save_test_xlsx(proba.argmax(1), "test1_final_minilm.xlsx")
    print(f"teste rotulado em predicoes/{out.name} | distribuicao:",
          {LABELS[i]: int(v) for i, v in enumerate(np.bincount(proba.argmax(1), minlength=3))})
    cols = [f"{n}_{l}" if Za[n].shape[1] == 3 else n for n in FINAL for l in LABELS[:Za[n].shape[1]]]
    pd.DataFrame(m[-1].coef_, index=LABELS, columns=cols).round(3).T.to_csv(P.RES / "08_final_coeficientes.csv")


if __name__ == "__main__":
    main()
