"""Avalia blocos novos no stacking, nas mesmas 5 dobras da v2.

Uso:  python e05_avaliar.py lora16                 (BERT novo: sozinho, no lugar do BERT da v2 e junto com ele)
      python e05_avaliar.py lora16 emb_qwen3       (varios blocos; os que comecam com emb_/knn_/atributos sao
                                                    blocos prontos de resultados/bloco_<nome>.npz)
"""
import sys

import numpy as np

from comum import SAIDA, V2, blocos_v2, carregar, carregar_bloco, cv_stacking, dobras, fmt


def bloco_bert(nome, n):
    """Logits fora da dobra e escore c5 - c1. Sem o _oof.npy agregado (ex.: previsoes de fim de epoca,
    <nome>_ep1), junta os arquivos <nome>_dobra<k>.npz."""
    if (SAIDA / f"{nome}_oof.npy").exists():
        o = np.load(SAIDA / f"{nome}_oof.npy")
        t = np.r_[np.load(SAIDA / f"{nome}_test1.npy"), np.load(SAIDA / f"{nome}_test2.npy")]
    else:
        partes = sorted(SAIDA.glob(f"{nome}_dobra[0-9].npz"))
        if len(partes) < 2:
            sys.exit(f"{nome}: faltam dobras ({len(partes)} encontradas)")
        o, t = np.zeros((n, 3), np.float32), 0
        for arq in partes:
            z = np.load(arq)
            o[z["idx"]] = z["oof"]; t = t + np.r_[z["test1"], z["test2"]] / len(partes)
    return o, (o[:, 2] - o[:, 0])[:, None], (t[:, 2] - t[:, 0])[:, None]


def main():
    y = carregar()["y"]
    dobra = dobras(y)
    fora, _, _ = blocos_v2()
    sem_bert = [b for b in V2 if b != "bert"]
    novos = {}
    for nome in sys.argv[1:]:
        if (SAIDA / f"bloco_{nome}.npz").exists():
            novos[nome] = carregar_bloco(nome)[0]
        else:
            logits, novos[nome], _ = bloco_bert(nome, len(y))
            if (logits[:, 1] != 0).any():                     # 3 classes: argmax direto
                acc = [(logits[dobra == k].argmax(1) == y[dobra == k]).mean() for k in range(5)]
                print(f"{nome} sozinho (argmax)        ", fmt(np.array(acc)))
            print(f"{nome} sozinho (RL no escore)  ", fmt(cv_stacking(novos[nome], y, dobra)))
    v2 = np.hstack([fora[b] for b in V2])
    print("bert da v2 sozinho (RL no escore)", fmt(cv_stacking(fora["bert"], y, dobra)))
    print("v2                              ", fmt(cv_stacking(v2, y, dobra)))
    for nome, X in novos.items():
        print(f"v2 + {nome:28s}", fmt(cv_stacking(np.hstack([v2, X]), y, dobra)))
        print(f"v2 com {nome:9s} no lugar do bert ", fmt(cv_stacking(np.hstack([fora[b] for b in sem_bert] + [X]), y, dobra)))
    if len(novos) > 1:
        print("v2 + todos                      ", fmt(cv_stacking(np.hstack([v2] + list(novos.values())), y, dobra)))
    bert = [fora["bert"]] + [X for nome, X in novos.items() if not (SAIDA / f"bloco_{nome}.npz").exists()]
    if len(bert) > 1:                                     # media dos escores padronizados = 1 bloco so
        media = np.mean([(b - b.mean()) / b.std() for b in bert], axis=0)
        print("media dos BERTs sozinha         ", fmt(cv_stacking(media, y, dobra)))
        print("v2 com a media dos BERTs        ", fmt(cv_stacking(np.hstack([fora[b] for b in sem_bert] + [media]), y, dobra)))


if __name__ == "__main__":
    main()
