"""Avaliacao pareada do pipeline v2 contra o baseline oficial, nas MESMAS divisoes (semente 123).

  (A) holdout oficial: train_test_split(test_size=0.2, random_state=123, stratify=Y)
  (B) validacao cruzada de 5 dobras estratificadas (semente 123) sobre todo o treino
Uso:  python codigo/07_avaliar.py
"""
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import pipeline as P


def blocos(Z):
    return [n for n in Z if n in ("mem", "viz") or not n.startswith(("mem", "viz"))]


def main():
    c = P.Corpus()
    todos = np.arange(c.n_treino)

    print("\n(A) holdout oficial 80/20, semente 123", flush=True)
    a, b = train_test_split(todos, test_size=0.2, random_state=P.SEED, stratify=c.y)
    base_b = P.baseline_oficial(c, a, b)
    print(f"  baseline oficial: {(base_b == c.y[b]).mean():.4f}", flush=True)
    Za, Zb, fold_a = P.empilhar(c, a, b, cache="holdout")

    print("\n(B) 5 dobras estratificadas, semente 123, treino completo", flush=True)
    teste = np.arange(c.n_treino, len(c.texto))
    Zt, Zte, fold = P.empilhar(c, todos, teste, cache="completo")
    base_cv = np.zeros(c.n_treino, dtype=int)
    for k in range(P.N_FOLDS):
        base_cv[fold == k] = P.baseline_oficial(c, todos[fold != k], todos[fold == k])
    acc_base = [float((base_cv[fold == k] == c.y[fold == k]).mean()) for k in range(P.N_FOLDS)]
    print(f"  baseline oficial: {np.mean(acc_base):.4f} +- {np.std(acc_base):.4f}", flush=True)

    def medir(nomes, C=0.1):
        accs, _ = P.cv_meta(Zt, nomes, c.y, fold, C)
        ph = P.meta(C).fit(P.juntar(Za, nomes), c.y[a]).predict(P.juntar(Zb, nomes))
        return dict(entradas=" + ".join(nomes), C=C, cv=np.mean(accs), cv_desvio=np.std(accs),
                    cv_menos_baseline=np.mean(accs) - np.mean(acc_base),
                    dobras_vencidas=int(sum(x > y for x, y in zip(accs, acc_base))),
                    holdout=float((ph == c.y[b]).mean()),
                    holdout_menos_baseline=float((ph == c.y[b]).mean() - (base_b == c.y[b]).mean()))

    # selecao gulosa: comeca vazio e acrescenta, a cada passo, o bloco que mais aumenta a validacao cruzada
    linhas, escolhidos, resto = [], [], blocos(Zt)
    while resto:
        cand = [medir(escolhidos + [n]) for n in resto]
        linhas += cand
        m = max(cand, key=lambda r: r["cv"])
        novo = m["entradas"].split(" + ")[-1]
        escolhidos.append(novo)
        resto.remove(novo)
        print(f"  + {novo:14s} cv {m['cv']:.4f} ({m['cv_menos_baseline']:+.4f}, vence {m['dobras_vencidas']}/5)"
              f" | holdout {m['holdout']:.4f} ({m['holdout_menos_baseline']:+.4f})", flush=True)
    for C in (0.01, 1.0):
        linhas.append(medir(blocos(Zt), C))
    df = pd.DataFrame(linhas).sort_values("cv", ascending=False)
    df.to_csv(P.RES / "07_avaliacao.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 70)
    print(df.head(12).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
