"""Decide se o BERT de 5 dobras parte do modelo adaptado (TAPT) ou do BERTimbau original.

Compara, na metade 0 do treino (validacao do fine-tuning de 2 dobras), o BERT com TAPT (tapt_dobra0) e o BERT da v2
(bert_dobra0), ambos com 128 tokens e o mesmo protocolo:
  - acuracia de 3 classes do proprio BERT (argmax);
  - stacking da v2 dentro da metade (5 dobras) com o BERT da v2 x com o BERT do TAPT no lugar.
Regra fixada antes de ver o resultado: usa o TAPT se ganhar pelo menos 0,3 ponto na acuracia do BERT E nao perder no
stacking. Grava resultados/usar_tapt.txt (lido pela fila) e o log e15_decidir_tapt_log.txt.
So usa o treino (validacao cruzada); nada do test1/test2.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold

from comum import RES, SAIDA, SEED, V2, blocos_v2, carregar, cv_stacking, fmt

y = carregar()["y"]
ref, tapt = np.load(RES / "bert_dobra0.npz"), np.load(SAIDA / "tapt_dobra0.npz")
idx = ref["idx"]
assert (tapt["idx"] == idx).all()
yy = y[idx]
dd = np.zeros(len(idx), int)
for k, (_, v) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(yy, yy)):
    dd[v] = k
f, _, _ = blocos_v2()
sem = np.hstack([f[b][idx] for b in V2 if b != "bert"])
esc = lambda o: (o[:, 2] - o[:, 0])[:, None]                                      # noqa: E731
acc_v2, acc_tapt = (ref["oof"].argmax(1) == yy).mean(), (tapt["oof"].argmax(1) == yy).mean()
st_v2, st_tapt = cv_stacking(np.hstack([sem, esc(ref["oof"])]), yy, dd), cv_stacking(np.hstack([sem, esc(tapt["oof"])]), yy, dd)
usar = acc_tapt - acc_v2 >= 0.003 and st_tapt.mean() >= st_v2.mean()
linhas = [f"metade 0 ({len(idx)} respostas)",
          f"BERT sozinho (argmax): v2 {acc_v2*100:.2f}% | TAPT {acc_tapt*100:.2f}% | diferenca {100*(acc_tapt-acc_v2):+.2f}",
          f"stacking com o BERT da v2    {fmt(st_v2)}",
          f"stacking com o BERT do TAPT  {fmt(st_tapt)}",
          f"decisao: {'usar o TAPT' if usar else 'usar o BERTimbau original'} no BERT de 5 dobras"]
print("\n".join(linhas))
(SAIDA / "e15_decidir_tapt_log.txt").write_text("\n".join(linhas) + "\n", encoding="utf-8")
if usar:
    (SAIDA / "usar_tapt.txt").write_text("sim\n")
else:
    (SAIDA / "usar_tapt.txt").unlink(missing_ok=True)
