"""Utilidades dos experimentos do EP2 (melhorias sobre a v2).

Tudo e avaliado nas MESMAS 5 dobras estratificadas (semente 123) do 02_stacking_ep2.py, e cada bloco novo
e salvo como previsoes fora da dobra (treino) + previsoes para test1/test2, no formato dos blocos da v2.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "codigo"))
from dados import RES, ROTULOS, SEED, carregar  # noqa: E402

warnings.filterwarnings("ignore")
EXP = Path(__file__).resolve().parents[1]
SAIDA = EXP / "resultados"
SAIDA.mkdir(exist_ok=True)
N_DOBRAS = 5


# checkpoints dos treinos longos (retomada apos reinicio do Windows); fora do OneDrive, porque sao grandes
CKPT = Path.home() / "ep2_ckpt"


def salvar_ckpt(nome, estado):
    """Grava o checkpoint de forma atomica: um reinicio no meio da gravacao nao corrompe o anterior."""
    import torch
    CKPT.mkdir(exist_ok=True)
    tmp = CKPT / f"{nome}.tmp"
    torch.save(estado, tmp)
    tmp.replace(CKPT / f"{nome}.pt")


def carregar_ckpt(nome):
    import torch
    arq = CKPT / f"{nome}.pt"
    return torch.load(arq, weights_only=False) if arq.exists() else None


def dobras(y):
    d = np.zeros(len(y), dtype=int)
    for k, (_, v) in enumerate(StratifiedKFold(N_DOBRAS, shuffle=True, random_state=SEED).split(y, y)):
        d[v] = k
    return d


def blocos_v2():
    """Blocos do stacking da v2: (fora da dobra, teste) por nome."""
    z = np.load(RES / "bases_stacking.npz")
    fora = {k[5:]: z[k] for k in z.files if k.startswith("fora_")}
    teste = {k[6:]: z[k] for k in z.files if k.startswith("teste_")}
    b, bt = np.load(RES / "bert_oof.npy"), np.r_[np.load(RES / "bert_test1.npy"), np.load(RES / "bert_test2.npy")]
    fora["bert"], teste["bert"] = (b[:, 2] - b[:, 0])[:, None], (bt[:, 2] - bt[:, 0])[:, None]
    return fora, teste, z["baseline"]


def salvar_bloco(nome, fora, teste):
    np.savez(SAIDA / f"bloco_{nome}.npz", fora=fora, teste=teste)


def carregar_bloco(nome):
    z = np.load(SAIDA / f"bloco_{nome}.npz")
    return z["fora"], z["teste"]


def meta(C=0.1):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def cv_stacking(X, y, dobra, C=0.1):
    """Acuracia por dobra do meta-classificador sobre os blocos X (fora da dobra)."""
    acc = []
    for k in range(N_DOBRAS):
        m = meta(C).fit(X[dobra != k], y[dobra != k])
        acc.append((m.predict(X[dobra == k]) == y[dobra == k]).mean())
    return np.array(acc)


def fmt(acc):
    return f"{acc.mean()*100:.2f} | " + " ".join(f"{a*100:.2f}" for a in acc)


V2 = ["lr_pal", "svc", "emb", "extremos", "bert"]
