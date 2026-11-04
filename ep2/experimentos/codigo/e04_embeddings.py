"""Experimento 4: embeddings de sentenca de um modelo mais forte (Qwen3-Embedding-0.6B) como bloco.

--amostra N: sondagem barata, so N respostas do treino (estratificadas), comparando regressao logistica
sobre os embeddings novos x MiniLM (v2) x TF-IDF nas mesmas respostas, com validacao cruzada de 5 dobras.
Sem --amostra: calcula os embeddings de treino + test1 + test2 (cache em ep2/experimentos/cache/) e
gera o bloco fora da dobra para o stacking. Retomavel: os embeddings saem em partes de --parte respostas
(~/ep2_ckpt/emb_<nome>_parte<j>.npy), e as partes prontas sao puladas ao rodar de novo.
--dispositivo GPU: roda na GPU integrada (Iris Xe) via OpenVINO, no ambiente ~/ep2_ov (que tem o openvino), com o
modelo exportado em --ov (FP16; cosseno >= 0,998 com os embeddings da CPU nas 48 respostas conferidas). Enquanto
esse processo roda, uma execucao em CPU (a da fila) nao calcula nada e sai, para nao tirar CPU do fine-tuning.
Uso:  ~/ep2_ov/Scripts/python e04_embeddings.py --dispositivo GPU
"""
import argparse
import os
import time

import numpy as np
import psutil
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from comum import CKPT, EXP, N_DOBRAS, SAIDA, SEED, V2, blocos_v2, carregar, cv_stacking, dobras, fmt, salvar_bloco

INSTRUCAO = ("Instruct: Classify how clear a Brazilian government agency's answer to a citizen's "
             "freedom-of-information request is\nQuery: ")


def codificar(modelo, textos, max_len, lote, nome=None, parte=0, ov=None):
    """Embeddings normalizados. Com nome e parte, grava cada parte em CKPT e pula as ja prontas.
    Com ov (pasta do modelo exportado para OpenVINO), roda na GPU integrada."""
    from sentence_transformers import SentenceTransformer
    if ov:
        m = SentenceTransformer(ov, backend="openvino", model_kwargs={"device": "GPU"})
    else:
        m = SentenceTransformer(modelo, device="cpu")
    m.max_seq_length = max_len
    pref = INSTRUCAO if "Qwen3" in modelo else ""
    ordem = np.argsort([len(t) for t in textos], kind="stable")  # lotes de tamanho parecido
    parte = parte or len(textos)
    E, t0 = [], time.time()
    for j, ini in enumerate(range(0, len(textos), parte)):
        arq = CKPT / f"emb_{nome}_parte{j}.npy" if nome else None
        if arq and arq.exists():
            E.append(np.load(arq)); continue
        e = m.encode([pref + textos[i] for i in ordem[ini:ini + parte]], batch_size=lote, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        if arq:
            CKPT.mkdir(exist_ok=True); np.save(arq, e)
            print(f"  parte {j+1}/{-(-len(textos) // parte)} pronta | {(time.time()-t0)/60:.0f} min", flush=True)
        E.append(e)
    E = np.concatenate(E)
    out = np.zeros_like(E)
    out[ordem] = E
    return out


def lr_cv(X, y, C):
    cv = StratifiedKFold(N_DOBRAS, shuffle=True, random_state=SEED)
    return cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)), X, y, cv=cv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--nome", default="qwen3")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--lote", type=int, default=8)
    ap.add_argument("--amostra", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--parte", type=int, default=500)
    ap.add_argument("--dispositivo", choices=["cpu", "GPU"], default="cpu")
    ap.add_argument("--ov", default=str(CKPT.parent / "ep2_cache" / "qwen3_ov"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    d = carregar()
    y, n = d["y"], len(d["treino"])
    textos = np.r_[d["treino"], d["test1"], d["test2"]]
    minilm = np.load(EXP.parent / "cache" / "emb_minilm.npy")
    t0 = time.time()

    if args.amostra:
        idx, _ = train_test_split(np.arange(n), train_size=args.amostra, stratify=y, random_state=SEED)
        E = codificar(args.modelo, list(textos[idx]), args.max_len, args.lote)
        print(f"{args.nome}: {len(idx)} textos em {(time.time()-t0)/60:.1f} min ({len(idx)/(time.time()-t0):.2f} textos/s)", flush=True)
        np.save(EXP / "resultados" / f"sonda_{args.nome}_{args.amostra}.npy", E)
        ys = y[idx]
        tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(textos[idx])
        cv = StratifiedKFold(N_DOBRAS, shuffle=True, random_state=SEED)
        print("acuracia em validacao cruzada nas mesmas respostas (media | dobras):")
        print("  TF-IDF + RL          ", fmt(cross_val_score(LogisticRegression(C=1.0, max_iter=2000), tf, ys, cv=cv)))
        for C in (0.003, 0.01, 0.03):
            print(f"  MiniLM  + RL C={C:<5} ", fmt(lr_cv(minilm[idx], ys, C)))
        for C in (0.003, 0.01, 0.03):
            print(f"  {args.nome:7s} + RL C={C:<5} ", fmt(lr_cv(E, ys, C)))
        return

    cache = EXP / "cache" / f"emb_{args.nome}.npy"
    cache.parent.mkdir(exist_ok=True)
    trava = CKPT / f"emb_{args.nome}.gpu"
    if cache.exists() and (SAIDA / f"bloco_emb_{args.nome}.npz").exists():
        print(f"bloco emb_{args.nome} ja pronto (resultados/bloco_emb_{args.nome}.npz)")
        return
    if cache.exists():
        E = np.load(cache)
    elif args.dispositivo == "cpu" and trava.exists() and psutil.pid_exists(int(trava.read_text())):
        print(f"embeddings {args.nome} sendo calculados na GPU (PID {trava.read_text()}): a CPU nao faz nada aqui")
        return
    else:
        if args.dispositivo == "GPU":
            CKPT.mkdir(exist_ok=True); trava.write_text(str(os.getpid()))
        E = codificar(args.modelo, list(textos), args.max_len, args.lote, args.nome, args.parte,
                      args.ov if args.dispositivo == "GPU" else None)
        np.save(cache, E)
        trava.unlink(missing_ok=True)
        print(f"embeddings prontos em {(time.time()-t0)/60:.0f} min", flush=True)
    dobra = dobras(y)
    melhor = None
    for C in (0.003, 0.01, 0.03):
        fora = np.zeros((n, 3))
        for k in range(N_DOBRAS):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(E[:n][dobra != k], y[dobra != k])
            fora[dobra == k] = m.predict_proba(E[:n][dobra == k])
        acc = (fora.argmax(1) == y).mean()
        print(f"  {args.nome} + RL C={C}: {acc*100:.2f}")
        if melhor is None or acc > melhor[0]:
            melhor = (acc, C, fora)
    acc, C, fora = melhor
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(E[:n], y)
    salvar_bloco(f"emb_{args.nome}", np.log(fora + 1e-6), np.log(m.predict_proba(E[n:]) + 1e-6))
    f2, _, _ = blocos_v2()
    base = np.hstack([f2[b] for b in V2])
    print("v2                ", fmt(cv_stacking(base, y, dobra)))
    print(f"v2 + emb_{args.nome:9s}", fmt(cv_stacking(np.hstack([base, np.log(fora + 1e-6)]), y, dobra)))


if __name__ == "__main__":
    main()
