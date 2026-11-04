"""EP2 - item 8 do relatorio: EP1 x EP2 no test1, com teste de McNemar.

Compara os 900 rotulos entregues no EP1 (entrega/test1.xlsx, na raiz do repositorio) com os
rotulos que o modelo do EP2 da para o mesmo test1 (ep2/resultados/test1_ep2.xlsx), usando os
rotulos verdadeiros do test1 fornecidos pelo professor. E o UNICO uso dos rotulos do test1.
Uso:  python ep2/codigo/04_mcnemar.py
"""
import numpy as np
import pandas as pd
from scipy.stats import binom, chi2

from dados import DADOS, EP2, RES, ROTULOS


def main():
    real = pd.read_excel(DADOS / "test1_rotulado.xlsx")["clarity"].astype(str).values
    ep1 = pd.read_excel(EP2.parent / "entrega" / "test1.xlsx")["clarity"].astype(str).values
    ep2 = pd.read_excel(RES / "test1_ep2.xlsx")["clarity"].astype(str).values
    assert len(real) == len(ep1) == len(ep2) == 900 and set(ep2) <= set(ROTULOS)
    a1, a2 = ep1 == real, ep2 == real
    n01 = int((a1 & ~a2).sum())   # so o EP1 acerta
    n10 = int((~a1 & a2).sum())   # so o EP2 acerta
    chi = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if n01 + n10 else 0.0
    p_chi = float(chi2.sf(chi, 1))
    p_exato = float(min(1.0, 2 * binom.cdf(min(n01, n10), n01 + n10, 0.5)))
    tab = pd.DataFrame({"EP2 acerta": [int((a1 & a2).sum()), n10], "EP2 erra": [n01, int((~a1 & ~a2).sum())]},
                       index=["EP1 acerta", "EP1 erra"])
    print(f"acuracia no test1: EP1 {a1.mean():.4f} | EP2 {a2.mean():.4f}")
    print("tabela de contingencia:\n" + tab.to_string())
    print(f"McNemar (qui-quadrado com correcao de continuidade): estatistica {chi:.3f}, p = {p_chi:.4f}")
    print(f"McNemar exato (binomial): p = {p_exato:.4f}")
    pd.DataFrame([dict(acc_ep1=a1.mean(), acc_ep2=a2.mean(), so_ep1_acerta=n01, so_ep2_acerta=n10,
                       mcnemar_chi2=chi, p_chi2=p_chi, p_exato=p_exato)]).to_csv(RES / "04_mcnemar.csv", index=False)


if __name__ == "__main__":
    main()
