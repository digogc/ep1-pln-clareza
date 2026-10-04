"""Compara duas versoes do stacking (semente 123), sem alterar a entrega:
    v2 (atual): lr_pal + svc (palavras + caracteres) + emb_e5small + extremos
    v3 (nova) : lr_pal + svc_pal (so palavras)       + emb_e5small + extremos
Grava a planilha de teste da v3 em predicoes/test1_v3.xlsx.
Uso:  python codigo/10_comparar_versoes.py
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import pipeline as P
from common import LABELS, save_test_xlsx

VERSOES = {
    "v2 (atual): SVM palavras+caracteres": ["lr_pal", "svc", "emb_e5small", "extremos"],
    "v3 (nova): SVM so palavras": ["lr_pal", "svc_pal", "emb_e5small", "extremos"],
    "extra: os dois SVMs": ["lr_pal", "svc", "svc_pal", "emb_e5small", "extremos"],
}
C_META = 0.1


def main():
    c = P.Corpus()
    n, y = c.n_treino, c.y
    todos, teste = np.arange(n), np.arange(n, len(c.texto))
    a, b = train_test_split(todos, test_size=0.2, random_state=P.SEED, stratify=y)
    Za, Zb, _ = P.empilhar(c, a, b, cache="holdout")
    Zt, Zte, fold = P.empilhar(c, todos, teste, cache="completo")
    base = np.zeros(n, dtype=int)
    for k in range(P.N_FOLDS):
        base[fold == k] = P.baseline_oficial(c, todos[fold != k], todos[fold == k])
    base_h = P.baseline_oficial(c, a, b)
    acc_b = [float((base[fold == k] == y[fold == k]).mean()) for k in range(P.N_FOLDS)]

    linhas = [dict(versao="baseline oficial", cv=np.mean(acc_b), cv_desvio=np.std(acc_b),
                   holdout=float((base_h == y[b]).mean()), **{f"dobra{k}": v for k, v in enumerate(acc_b)})]
    oof, tst = {"baseline oficial": base}, {}
    for nome, blocos in VERSOES.items():
        accs, pred = P.cv_meta(Zt, blocos, y, fold, C_META)
        ph = P.meta(C_META).fit(P.juntar(Za, blocos), y[a]).predict(P.juntar(Zb, blocos))
        oof[nome] = pred
        tst[nome] = P.meta(C_META).fit(P.juntar(Zt, blocos), y).predict(P.juntar(Zte, blocos))
        linhas.append(dict(versao=nome, cv=np.mean(accs), cv_desvio=np.std(accs),
                           holdout=float((ph == y[b]).mean()), **{f"dobra{k}": v for k, v in enumerate(accs)},
                           **{f"acerto_{l}": float((pred[y == j] == j).mean()) for j, l in enumerate(LABELS)},
                           dobras_acima_do_baseline=int(sum(x > z for x, z in zip(accs, acc_b)))))
    df = pd.DataFrame(linhas)
    df.to_csv(P.RES / "10_comparar_versoes.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(4).to_string(index=False))

    v2, v3 = list(VERSOES)[:2]
    print(f"\nrespostas do treino em que so uma versao acerta: v3 {int(((oof[v3] == y) & (oof[v2] != y)).sum())}"
          f" x v2 {int(((oof[v2] == y) & (oof[v3] != y)).sum())}")
    for v in (v2, v3):
        print(f"  {v} x baseline: {int(((oof[v] == y) & (base != y)).sum())} x {int(((oof[v] != y) & (base == y)).sum())}")
    out = save_test_xlsx(tst[v3], "test1_v3.xlsx")
    print(f"teste da v3 em {out.name} | distribuicao:",
          {LABELS[i]: int(q) for i, q in enumerate(np.bincount(tst[v3], minlength=3))},
          f"| concordancia com a v2 no teste: {(tst[v2] == tst[v3]).mean():.3f}")
    atual = pd.read_excel(P.ROOT / "entrega" / "test1.xlsx")["clarity"].map({l: i for i, l in enumerate(LABELS)}).values
    print("entrega/test1.xlsx igual a v2:", bool((atual == tst[v2]).all()))


if __name__ == "__main__":
    main()
