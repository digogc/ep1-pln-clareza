"""McNemar EP1 x um candidato do EP2 no test1, pedido pelo usuario em 02/11 para acompanhar o ganho do BERT de 256.

A escolha do modelo final NAO usa este resultado: ela ja esta fixada pela validacao cruzada (maior media entre
v2 + 256, v2 + 512 e v2 + 256 + 512; empate -> menos blocos). O test2 nao e tocado.
Uso:  python e17_mcnemar_candidato.py                  (v2, confere se reproduz o p = 0,047 do relatorio)
      python e17_mcnemar_candidato.py bert256_5d       (v2 + bloco BERT novo)
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import binom, chi2

from comum import SAIDA, V2, blocos_v2, carregar, meta
from dados import DADOS, EP2, RES, ROTULOS
from e05_avaliar import bloco_bert

d = carregar()
y, n, n1 = d["y"], len(d["treino"]), len(d["test1"])
fora, teste, _ = blocos_v2()
Xf, Xt = [fora[b] for b in V2], [teste[b] for b in V2]
for nome in sys.argv[1:]:
    _, of, te = bloco_bert(nome, n)
    Xf.append(of); Xt.append(te)
pred = meta().fit(np.hstack(Xf), y).predict(np.hstack(Xt))[:n1]

real = pd.read_excel(DADOS / "test1_rotulado.xlsx")["clarity"].astype(str).values
ep1 = pd.read_excel(EP2.parent / "entrega" / "test1.xlsx")["clarity"].astype(str).values
cand = np.array([ROTULOS[int(p)] for p in pred])
if not sys.argv[1:]:
    v2 = pd.read_excel(RES / "test1_ep2.xlsx")["clarity"].astype(str).values
    print(f"conferencia: rotulos iguais aos da v2 entregue em {np.mean(cand == v2)*100:.1f}% do test1")
a1, a2 = ep1 == real, cand == real
n01, n10 = int((a1 & ~a2).sum()), int((~a1 & a2).sum())
chi = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
nome = "v2" + "".join(f" + {s}" for s in sys.argv[1:])
linhas = [f"{nome}: acuracia no test1 EP1 {a1.mean()*100:.2f}% | candidato {a2.mean()*100:.2f}%",
          f"  so o EP1 acerta {n01} | so o candidato acerta {n10}",
          f"  McNemar qui-quadrado (com correcao) p = {chi2.sf(chi, 1):.4f} | exato p = {min(1.0, 2*binom.cdf(min(n01, n10), n01+n10, 0.5)):.4f}"]
print("\n".join(linhas))
with open(SAIDA / "e17_mcnemar_candidatos.txt", "a", encoding="utf-8") as f:
    f.write("\n".join(linhas) + "\n")
