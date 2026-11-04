"""Compara o LoRA (r=16, dobra 0 do e02) com o fine-tuning completo da v2 na mesma metade de validacao."""
import sys; sys.path.insert(0, ".")
import numpy as np
from scipy.stats import binomtest
from comum import RES, SAIDA, carregar
d = carregar(); y = d["y"]
v = np.load(RES / "bert_dobra0.npz"); l = np.load(SAIDA / "lora16_dobra0.npz")
print(v.files, l.files)
assert (v["idx"] == l["idx"]).all()
b = l["idx"]; yb = y[b]
pv, pl = v["oof"].argmax(1), l["oof"].argmax(1)
av, al = (pv == yb), (pl == yb)
print(f"v2 BERT (FT) metade 0: {av.mean():.4f} | LoRA r16: {al.mean():.4f}")
n01, n10 = int((av & ~al).sum()), int((~av & al).sum())
print(f"so FT acerta {n01}, so LoRA acerta {n10}, p exato {binomtest(n10, n01 + n10).pvalue:.4f}")
print(f"concordancia FT x LoRA: {(pv == pl).mean():.3f}")
sm = lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)
for w in (0.5,):
    m = (sm(v["oof"]) * (1 - w) + sm(l["oof"]) * w).argmax(1)
    print(f"media das probabilidades FT+LoRA: {(m == yb).mean():.4f}")
sv, sl = v["oof"][:, 2] - v["oof"][:, 0], l["oof"][:, 2] - l["oof"][:, 0]
print(f"correlacao dos escores c5-c1: {np.corrcoef(sv, sl)[0,1]:.3f}")
