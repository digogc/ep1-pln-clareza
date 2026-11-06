"""Compara modelos BERT de 2 dobras na MESMA metade de validacao (dobra k), antes de a outra dobra terminar.

Medidas: acerto do sinal do escore c5 - c1 nas respostas c1/c5 e acuracia de 3 classes de uma regressao logistica
sobre o escore (validacao cruzada de 5 dobras dentro da metade, semente 123). Referencia: BERT da v2 (bert_dobra<k>).
Uso:  python e10_comparar_metade.py 0 ordinal_ep1 lora16 ...
"""
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from comum import RES, SAIDA, SEED, carregar

k = int(sys.argv[1])
y = carregar()["y"]
ref = np.load(RES / f"bert_dobra{k}.npz")
idx = ref["idx"]
yb = y[idx]
ext = yb != 1
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
escores = {"v2 (ce)": ref["oof"][:, 2] - ref["oof"][:, 0]}
for nome in sys.argv[2:]:
    z = np.load(SAIDA / f"{nome}_dobra{k}.npz")
    assert (z["idx"] == idx).all()
    escores[nome] = z["oof"][:, 2] - z["oof"][:, 0]
print(f"metade {k}: {len(idx)} respostas ({ext.sum()} c1/c5)")
print(f"{'modelo':14s} sinal c1xc5 | 3 classes (RL no escore) | corr. com v2")
for nome, s in escores.items():
    sinal = ((s[ext] > 0) == (yb[ext] == 2)).mean()
    acc = cross_val_score(LogisticRegression(), s[:, None], yb, cv=cv).mean()
    print(f"{nome:14s} {sinal*100:6.2f}%     | {acc*100:6.2f}%                | {np.corrcoef(s, escores['v2 (ce)'])[0, 1]:.3f}")
if len(escores) > 1:
    pad = np.c_[[(s - s.mean()) / s.std() for s in escores.values()]].T
    acc = cross_val_score(LogisticRegression(), pad.mean(1)[:, None], yb, cv=cv).mean()
    acc2 = cross_val_score(LogisticRegression(), pad, yb, cv=cv).mean()
    print(f"media dos escores: {acc*100:.2f}% | todos os escores juntos na RL: {acc2*100:.2f}%")
