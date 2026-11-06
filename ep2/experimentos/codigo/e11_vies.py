"""Experimento 11: ajuste de vies por classe na saida do meta-classificador (validacao cruzada aninhada).

A regressao logistica do meta minimiza a log-loss, nao o erro. Aqui, em cada dobra externa, as probabilidades fora
da dobra das outras 4 dobras escolhem um vies aditivo no log da probabilidade de c1 e de c5 (grade de -0,3 a 0,3)
que maximiza a acuracia; o vies e aplicado na dobra externa.
"""
import itertools

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from comum import SEED, V2, blocos_v2, carregar, dobras, fmt, meta

y = carregar()["y"]
dobra = dobras(y)
fora, _, _ = blocos_v2()
X = np.hstack([fora[b] for b in V2])
grade = np.round(np.arange(-0.3, 0.31, 0.05), 2)
acc0, acc1, vieses = [], [], []
for k in range(5):
    tr, va = dobra != k, dobra == k
    interno = cross_val_predict(meta(), X[tr], y[tr], cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")
    lp = np.log(interno + 1e-9)
    melhor = max(itertools.product(grade, grade), key=lambda b: ((lp + [b[0], 0, b[1]]).argmax(1) == y[tr]).mean())
    m = meta().fit(X[tr], y[tr])
    pv = np.log(m.predict_proba(X[va]) + 1e-9)
    acc0.append((pv.argmax(1) == y[va]).mean())
    acc1.append(((pv + [melhor[0], 0, melhor[1]]).argmax(1) == y[va]).mean())
    vieses.append(melhor)
print("v2                ", fmt(np.array(acc0)))
print("v2 + vies ajustado", fmt(np.array(acc1)))
print("vieses (c1, c5) por dobra:", vieses)
