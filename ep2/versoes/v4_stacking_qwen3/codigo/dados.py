"""Leitura dos dados do EP2 (ACH2118). Usado por todos os scripts de ep2/codigo/.

ep2/dados/ (copiar do e-Disciplinas; fora do git):
    train.xlsx            treino (o mesmo do EP1)
    test2.xlsx            teste oficial do EP2 (897 linhas, sem rotulo)
    test1_rotulado.xlsx   teste do EP1 com rotulos: SO para o teste de McNemar (EP1 x EP2)

Os arquivos do EP2 trazem caracteres de controle escritos como "_x0096_" (escape do Excel); o EP1
trazia o caractere em si. `normalizar` desfaz o escape, e assim os textos ficam identicos aos do EP1.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

EP2 = Path(__file__).resolve().parent.parent
DADOS, RES = EP2 / "dados", EP2 / "resultados"
ROTULOS = ["c1", "c234", "c5"]
SEED = 123


def normalizar(t):
    t = re.sub(r"_x([0-9A-Fa-f]{4})_", lambda m: chr(int(m.group(1), 16)), str(t))
    return re.sub(r"\s+", " ", t).strip()


def carregar(com_rotulo_test1=False):
    """Devolve dict com textos e rotulos. Os rotulos do test1 so sao lidos se pedidos explicitamente."""
    tr = pd.read_excel(DADOS / "train.xlsx")
    t1 = pd.read_excel(DADOS / "test1_rotulado.xlsx")
    t2 = pd.read_excel(DADOS / "test2.xlsx")
    d = {
        "treino": np.array([normalizar(t) for t in tr["resp_text"]]),
        "y": tr["clarity"].map({r: i for i, r in enumerate(ROTULOS)}).values.astype(int),
        "test1": np.array([normalizar(t) for t in t1["resp_text"]]),
        "test2": np.array([normalizar(t) for t in t2["resp_text"]]),
    }
    if com_rotulo_test1:
        d["y_test1"] = t1["clarity"].map({r: i for i, r in enumerate(ROTULOS)}).values.astype(int)
    return d
