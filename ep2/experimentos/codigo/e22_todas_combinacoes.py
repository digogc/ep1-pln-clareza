"""Experimento 22 (06/11, pedido do usuario): TODAS as combinacoes de blocos no stacking, com o BERT 256 de 10 dobras.

Blocos prontos (sem retreino), 15 no total: os 4 do EP1 (TF-IDF com RL e SVM, extremos c1 x c5, MiniLM), Qwen3,
atributos, kNN (MiniLM e TF-IDF) e 7 BERTs: 128 em 2 dobras (o da v2), 128 em 5 dobras, 256 em 5 dobras (o da v4),
256 em 10 dobras, 512, ordinal (2 dobras) e extremos (2 dobras). Sao 2^15 - 1 = 32.767 subconjuntos, todos com o
meta-classificador da v4 (padronizacao + RL, C = 0,1), nas mesmas 5 dobras.

Escolher a maior media entre 32.767 e muito otimista. Por isso o script estima, com validacao cruzada aninhada,
quanto rende o PROCEDIMENTO "escolher a melhor combinacao": em cada dobra externa, a escolha e feita so com as outras
4 dobras (validacao cruzada interna nelas) e avaliada na dobra que ficou de fora. Empates vao para menos blocos.
Regra fixada antes de rodar (a mesma do e19, agora contra a v4): troca a v4 so se a melhor combinacao
(a) ganhar pelo menos 0,2 ponto na media, (b) ganhar em pelo menos 4 das 5 dobras e (c) o aninhado nao ficar abaixo
da v4. So usa o treino; nada do test1/test2.

Uso:  python e22_todas_combinacoes.py            (tudo; ~4 h em 8 processos: fase 1 ~1h20, fase 2 ~2h40)
      python e22_todas_combinacoes.py --medir    (so mede a velocidade em 400 combinacoes)
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"                         # um nucleo por processo; o paralelismo e entre processos
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse                                             # noqa: E402
import itertools                                            # noqa: E402
import time                                                 # noqa: E402

import numpy as np                                          # noqa: E402
from joblib import Parallel, delayed                        # noqa: E402
from scipy.stats import binomtest                           # noqa: E402

from comum import N_DOBRAS, SAIDA, blocos_v2, carregar, carregar_bloco, dobras, fmt, meta  # noqa: E402
from e05_avaliar import bloco_bert                          # noqa: E402

BLOCOS = ["lr_pal", "svc", "extremos", "emb", "qwen3", "atributos", "knn_minilm", "knn_tfidf",
          "bert", "b128", "b256", "b256_10d", "b512", "b_ordinal", "b_extremos"]
NOMES = {"lr_pal": "RL TF-IDF", "svc": "SVM TF-IDF", "extremos": "extremos TF-IDF", "emb": "MiniLM", "qwen3": "Qwen3",
         "atributos": "atributos", "knn_minilm": "kNN MiniLM", "knn_tfidf": "kNN TF-IDF", "bert": "BERT 128 (2d)",
         "b128": "BERT 128 (5d)", "b256": "BERT 256 (5d)", "b256_10d": "BERT 256 (10d)", "b512": "BERT 512",
         "b_ordinal": "BERT ordinal", "b_extremos": "BERT extremos"}
V4 = ["lr_pal", "svc", "extremos", "qwen3", "bert", "b256"]
PARES = list(itertools.combinations(range(N_DOBRAS), 2))


def montar():
    y = carregar()["y"]
    F, _, _ = blocos_v2()
    for nome, chave in [("bert5d", "b128"), ("bert256_5d", "b256"), ("bert256_10d", "b256_10d"),
                        ("bert512_5d", "b512"), ("ordinal", "b_ordinal"), ("extremos", "b_extremos")]:
        F[chave] = bloco_bert(nome, len(y))[1]
    F["qwen3"] = carregar_bloco("emb_qwen3")[0]
    for nome in ["atributos", "knn_minilm", "knn_tfidf"]:
        F[nome] = carregar_bloco(nome)[0]
    X = np.hstack([F[b] for b in BLOCOS]).astype(np.float64)
    cols, ini = [], 0
    for b in BLOCOS:
        cols.append(np.arange(ini, ini + F[b].shape[1])); ini += F[b].shape[1]
    return X, y, cols


def subX(m, X, cols):
    return X[:, np.concatenate([cols[i] for i in range(len(BLOCOS)) if m >> i & 1])]


def avaliar_plano(mascaras, X, y, dobra, cols):
    """Acerto por dobra de cada subconjunto (meta treinado nas outras 4 dobras)."""
    out = []
    for m in mascaras:
        Xs = subX(m, X, cols)
        out.append([(meta().fit(Xs[dobra != k], y[dobra != k]).predict(Xs[dobra == k]) == y[dobra == k]).mean()
                    for k in range(N_DOBRAS)])
    return np.array(out)


def avaliar_interno(mascaras, X, y, dobra, cols):
    """Para o aninhado: interno[k, j] = acerto na dobra j do meta treinado sem as dobras {k, j}
    (um ajuste serve aos dois lados do par)."""
    out = []
    for m in mascaras:
        Xs = subX(m, X, cols)
        a = np.full((N_DOBRAS, N_DOBRAS), np.nan)
        for k, j in PARES:
            tr = (dobra != k) & (dobra != j)
            mod = meta().fit(Xs[tr], y[tr])
            a[k, j] = (mod.predict(Xs[dobra == j]) == y[dobra == j]).mean()
            a[j, k] = (mod.predict(Xs[dobra == k]) == y[dobra == k]).mean()
        out.append(a)
    return np.array(out)


def em_paralelo(f, todas, X, y, dobra, cols, processos):
    lotes = [todas[i:i + 128] for i in range(0, len(todas), 128)]
    return np.concatenate(Parallel(processos, verbose=5)(delayed(f)(l, X, y, dobra, cols) for l in lotes))


def mascara(blocos):
    return sum(1 << BLOCOS.index(b) for b in blocos)


def rotulo(m):
    return " + ".join(NOMES[b] for i, b in enumerate(BLOCOS) if m >> i & 1)


def prever_fora(m, X, y, dobra, cols):
    Xs, p = subX(m, X, cols), np.zeros(len(y), int)
    for k in range(N_DOBRAS):
        p[dobra == k] = meta().fit(Xs[dobra != k], y[dobra != k]).predict(Xs[dobra == k])
    return p


def gravar(linhas):
    texto = "\n".join(linhas)
    print(texto, flush=True)
    (SAIDA / "e22_todas_combinacoes_log.txt").write_text(texto + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--medir", action="store_true")
    ap.add_argument("--processos", type=int, default=8)
    args = ap.parse_args()
    X, y, cols = montar()
    dobra = dobras(y)
    # menos blocos primeiro: no empate, o max fica com o subconjunto menor
    todas = sorted(range(1, 2 ** len(BLOCOS)), key=lambda m: (bin(m).count("1"), m))
    if args.medir:
        amostra = list(np.random.default_rng(123).choice(todas, 400, replace=False))
        t = time.time()
        Parallel(args.processos)(delayed(avaliar_plano)(amostra[i::args.processos], X, y, dobra, cols)
                                 for i in range(args.processos))
        Parallel(args.processos)(delayed(avaliar_interno)(amostra[i::args.processos], X, y, dobra, cols)
                                 for i in range(args.processos))
        dt = time.time() - t
        print(f"400 combinacoes em {dt:.0f} s -> todas as {len(todas)}: ~{dt * len(todas) / 400 / 60:.0f} min")
        return
    # fase 1: validacao cruzada simples de todas (retomavel: o resultado fica gravado)
    arq1, arq2 = SAIDA / "e22_plano.npz", SAIDA / "e22_interno.npz"
    t = time.time()
    if arq1.exists():
        plano = np.load(arq1)["plano"]
    else:
        plano = em_paralelo(avaliar_plano, todas, X, y, dobra, cols, args.processos)
        np.savez(arq1, mascaras=np.array(todas), plano=plano, blocos=BLOCOS)
    todas = np.array(todas)
    linhas = [f"fase 1: {len(todas)} combinacoes de {len(BLOCOS)} blocos, validacao cruzada simples "
              f"({(time.time() - t) / 60:.0f} min)"]
    media = plano.mean(1)
    iv4 = int(np.where(todas == mascara(V4))[0][0])
    ref = plano[iv4]
    ordem = np.argsort(-media, kind="stable")
    linhas.append(f"\nv4 (referencia): {fmt(ref)}  | posicao {int(np.where(ordem == iv4)[0][0]) + 1} de {len(todas)}")
    linhas.append("\n25 melhores (media | dobras 1-5 | dobras melhores que a v4):")
    for r, i in enumerate(ordem[:25], 1):
        linhas.append(f"{r:3d}. {fmt(plano[i])} ({int((plano[i] > ref).sum())}/5)  {rotulo(todas[i])}")
    linhas.append("\nmelhor combinacao por numero de blocos:")
    nb = np.array([bin(m).count("1") for m in todas])
    for q in range(1, len(BLOCOS) + 1):
        i = np.where(nb == q)[0][np.argmax(media[nb == q])]
        linhas.append(f"  {q:2d} blocos: {fmt(plano[i])}  {rotulo(todas[i])}")
    linhas.append("\nefeito medio de cada bloco (media das combinacoes com ele - sem ele; presenca entre as 1.000 melhores):")
    top = np.zeros(len(todas), bool); top[ordem[:1000]] = True
    for b in range(len(BLOCOS)):
        tem = (todas >> b & 1).astype(bool)
        linhas.append(f"  {NOMES[BLOCOS[b]]:18s} {100 * (media[tem].mean() - media[~tem].mean()):+.2f} pp | "
                      f"presente em {top[tem].sum() / 10:.0f}% das 1.000 melhores")
    melhor = int(ordem[0])
    pm, p4 = prever_fora(todas[melhor], X, y, dobra, cols), prever_fora(mascara(V4), X, y, dobra, cols)
    n01, n10 = int(((p4 != y) & (pm == y)).sum()), int(((p4 == y) & (pm != y)).sum())
    linhas.append(f"\nmelhor na validacao cruzada: '{rotulo(todas[melhor])}' ({fmt(plano[melhor])})")
    linhas.append(f"melhor x v4 nas previsoes fora da dobra: so a melhor acerta {n01}, so a v4 {n10}, "
                  f"McNemar exato p = {binomtest(n01, n01 + n10).pvalue:.3f}")
    linhas.append("\nfase 2 (validacao cruzada aninhada) em andamento...")
    gravar(linhas)
    linhas.pop()
    # fase 2: validacao cruzada aninhada do procedimento "escolher a combinacao com a maior media"
    t = time.time()
    if arq2.exists():
        interno = np.load(arq2)["interno"]
    else:
        interno = em_paralelo(avaliar_interno, list(todas), X, y, dobra, cols, args.processos)
        np.savez(arq2, mascaras=todas, interno=interno)
    linhas.append(f"\nfase 2: validacao cruzada aninhada ({(time.time() - t) / 60:.0f} min)")
    acc = []
    for k in range(N_DOBRAS):
        inner = np.nanmean(np.delete(interno[:, k, :], k, axis=1), axis=1)
        b = int(np.argmax(inner))
        acc.append(plano[b, k])
        linhas.append(f"  aninhado, dobra {k + 1}: escolheu '{rotulo(todas[b])}' (interno {100 * inner[b]:.2f}) "
                      f"-> {100 * plano[b, k]:.2f}")
    acc = np.array(acc)
    linhas.append(f"procedimento 'escolher a melhor' (aninhado): {fmt(acc)}  | v4: {fmt(ref)}")
    ok = (media[melhor] - ref.mean() >= 0.002 and (plano[melhor] > ref).sum() >= 4 and acc.mean() >= ref.mean())
    linhas.append(f"decisao pela regra fixada: {'trocar a v4 pela melhor' if ok else 'manter a v4'}")
    gravar(linhas)


if __name__ == "__main__":
    main()
