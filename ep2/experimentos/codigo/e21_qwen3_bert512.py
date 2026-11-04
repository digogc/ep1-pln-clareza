"""Experimento 21 (03/11, pedido do usuario): Qwen3 com o BERT de 512 tokens, contra a v4 (v3 com Qwen3).

1. validacao cruzada (5 dobras, so treino): acuracia por dobra e McNemar entre as duas, nas 20.092 previsoes fora
   da dobra;
2. test1 (rotulos do professor): acuracia de cada uma, McNemar de cada uma contra o EP1 e entre as duas.
O test2 nao tem rotulos e nao e usado. Nao grava planilhas de entrega.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402
from scipy.stats import binom, chi2                         # noqa: E402

from comum import SAIDA, V2, blocos_v2, carregar, carregar_bloco, dobras, fmt, meta  # noqa: E402
from dados import DADOS, EP2, ROTULOS                       # noqa: E402
from e05_avaliar import bloco_bert                          # noqa: E402


def mcnemar(a1, a2):
    """a1, a2: acertos (bool) de dois modelos nos mesmos exemplos. Devolve (so 1 acerta, so 2 acerta, p qui2, p exato)."""
    n01, n10 = int((a1 & ~a2).sum()), int((~a1 & a2).sum())
    chi = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return n01, n10, chi2.sf(chi, 1), min(1.0, 2 * binom.cdf(min(n01, n10), n01 + n10, 0.5))


d = carregar()
y, n, n1 = d["y"], len(d["treino"]), len(d["test1"])
dobra = dobras(y)
F, T, _ = blocos_v2()
_, F["b256"], T["b256"] = bloco_bert("bert256_5d", n)
_, F["b512"], T["b512"] = bloco_bert("bert512_5d", n)
F["qwen3"], T["qwen3"] = carregar_bloco("emb_qwen3")
Q = [("qwen3" if b == "emb" else b) for b in V2]
MODELOS = {"v4 (Qwen3 + BERT 256)": Q + ["b256"],
           "Qwen3 + BERT 512": Q + ["b512"],
           "Qwen3 + BERT 256 + 512": Q + ["b256", "b512"],
           "v2 + Qwen3 + BERT 512": V2 + ["qwen3", "b512"]}

linhas, oof, t1 = [], {}, {}
for nome, blocos in MODELOS.items():
    X, Xt = np.hstack([F[b] for b in blocos]), np.hstack([T[b] for b in blocos])
    p = np.zeros(n, int)
    for k in range(5):
        p[dobra == k] = meta().fit(X[dobra != k], y[dobra != k]).predict(X[dobra == k])
    oof[nome] = p
    t1[nome] = meta().fit(X, y).predict(Xt)[:n1]
    acc = np.array([(p[dobra == k] == y[dobra == k]).mean() for k in range(5)])
    linhas.append(f"{nome:24s} blocos {blocos}\n  validacao cruzada {fmt(acc)}")

ref = list(MODELOS)[0]
linhas.append("")
for nome in list(MODELOS)[1:]:
    n01, n10, pc, pe = mcnemar(oof[ref] == y, oof[nome] == y)
    linhas.append(f"validacao cruzada, {nome} x v4: so a v4 acerta {n01} | so a outra acerta {n10} | McNemar p = {pc:.4f}")

real = pd.read_excel(DADOS / "test1_rotulado.xlsx")["clarity"].astype(str).values
ep1 = pd.read_excel(EP2.parent / "entrega" / "test1.xlsx")["clarity"].astype(str).values == real
linhas.append(f"\ntest1 (900 respostas): EP1 {ep1.mean()*100:.2f}%")
acertos = {}
for nome in MODELOS:
    acertos[nome] = np.array([ROTULOS[int(v)] for v in t1[nome]]) == real
    n01, n10, pc, pe = mcnemar(ep1, acertos[nome])
    linhas.append(f"  {nome:24s} {acertos[nome].mean()*100:.2f}% | contra o EP1: so o EP1 acerta {n01}, so o EP2 acerta {n10}"
                  f" | McNemar p = {pc:.4f} (exato {pe:.4f})")
for nome in list(MODELOS)[1:]:
    n01, n10, pc, pe = mcnemar(acertos[ref], acertos[nome])
    linhas.append(f"  {nome} x v4 no test1: so a v4 acerta {n01} | so a outra acerta {n10} | McNemar p = {pc:.4f}")
print("\n".join(linhas))
(SAIDA / "e21_qwen3_bert512_log.txt").write_text("\n".join(linhas) + "\n", encoding="utf-8")
