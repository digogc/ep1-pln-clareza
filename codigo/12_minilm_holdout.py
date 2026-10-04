"""Confere no holdout oficial (semente 123) as variacoes com MiniLM de 11_melhorias_rapidas.py.
Nao altera a entrega. Uso:  python codigo/12_minilm_holdout.py
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

import pipeline as P

c = P.Corpus(embeddings=("e5small", "minilm"))
n, y = c.n_treino, c.y
a, b = train_test_split(np.arange(n), test_size=0.2, random_state=P.SEED, stratify=y)
z = np.load(P.RES / "v2_blocos_holdout.npz")
Za = {k[2:]: z[k] for k in z.files if k.startswith("a_")}
Zb = {k[2:]: z[k] for k in z.files if k.startswith("b_")}
f = P.base_emb("minilm", 0.01)
ya = y[a]
oof = np.zeros((len(a), 3))
for _, (i, j) in enumerate(P.dobras(ya)):
    oof[j] = f(c, a[i], ya[i], a[j])
Za["minilm"], Zb["minilm"] = oof, f(c, a, ya, b)
for nome, blocos in (("v2", ["lr_pal", "svc", "emb_e5small", "extremos"]),
                     ("v2 + minilm", ["lr_pal", "svc", "emb_e5small", "extremos", "minilm"]),
                     ("minilm no lugar do e5", ["lr_pal", "svc", "minilm", "extremos"])):
    p = P.meta(0.1).fit(P.juntar(Za, blocos), ya).predict(P.juntar(Zb, blocos))
    print(f"holdout {nome}: {(p == y[b]).mean():.4f}")
