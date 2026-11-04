"""Previsor na GPU integrada (Iris Xe, via OpenVINO): tira a inferencia do caminho do fine-tuning.

O fine-tuning (e02_bert.py) roda na CPU. Ao fim de cada epoca/dobra ele parava ~15-30 min para prever a validacao,
o test1 e o test2. Com este processo ativo, o e02 so grava o modelo em ~/ep2_cache/modelos/ e deixa um pedido em
~/ep2_ckpt/fila_gpu/; daqui, a GPU faz as previsoes (mesmo truncamento cabeca + cauda, mesmo formato de saida
<nome>_dobra<k>.npz) enquanto a CPU segue treinando. Sem este processo, o e02 preve na CPU como antes.
A GPU roda em FP16: 22 textos/s com 256 tokens (CPU: ~7), diferenca maxima de 0,003 nos logits e mesmo argmax da CPU.

Roda no ambiente ~/ep2_ov (que tem o openvino):
    ~/ep2_ov/Scripts/python e12_previsor_gpu.py                 (fica esperando pedidos)
    ~/ep2_ov/Scripts/python e12_previsor_gpu.py --testar        (atende os pedidos, compara com a CPU e mantem o modelo)
"""
import argparse
import json
import os
import shutil
import time

import numpy as np
import psutil
import torch
from transformers import AutoTokenizer

from comum import CKPT, SAIDA, carregar
from e02_bert import FILA_GPU, TRAVA_GPU, codificar, ft, lote


def tres_colunas(out, objetivo):
    """Mesmo formato do prever3 do e02: escore unico s vira (-s/2, 0, s/2)."""
    if objetivo == "ce":
        return out.astype(np.float32)
    s = out[:, 0]
    return np.c_[-s / 2, np.zeros_like(s), s / 2].astype(np.float32)


def prever_ov(modelo, seqs, pad, tam=32):
    ordem = np.argsort([len(s) for s in seqs])
    out = None
    for i in range(0, len(seqs), tam):
        idx = ordem[i:i + tam]
        ii, am = lote([seqs[j] for j in idx], pad)
        lg = modelo(input_ids=ii, attention_mask=am, token_type_ids=torch.zeros_like(ii)).logits
        lg = lg.numpy() if hasattr(lg, "numpy") else np.asarray(lg)
        if out is None:
            out = np.zeros((len(seqs), lg.shape[1]), np.float32)
        out[idx] = lg
    return out


def atender(pedido, d, cache_seqs, testar):
    from optimum.intel import OVModelForSequenceClassification
    t0 = time.time()
    chave = (pedido["tok"], pedido["max_len"])
    if chave not in cache_seqs:
        tok = AutoTokenizer.from_pretrained(pedido["tok"])
        cache_seqs[chave] = (codificar(tok, list(d["treino"]) + list(d["test1"]) + list(d["test2"]), pedido["max_len"]),
                             tok.pad_token_id)
    seqs, pad = cache_seqs[chave]
    ft.PAD = pad
    b = np.load(os.path.join(pedido["dir"], "idx.npy"))
    n, lim = len(d["treino"]), pedido["limite"] or len(seqs)
    modelo = OVModelForSequenceClassification.from_pretrained(pedido["dir"], export=True, device="GPU")
    partes = [[seqs[j] for j in b], seqs[n:n + len(d["test1"])][:lim], seqs[n + len(d["test1"]):][:lim]]
    oof, lt1, lt2 = (tres_colunas(prever_ov(modelo, p, pad), pedido["objetivo"]) for p in partes)
    y = d["y"][b]
    if pedido["objetivo"] == "ce":
        acc = (oof.argmax(1) == y).mean()
    else:
        s, ext = oof[:, 2] - oof[:, 0], y != 1
        acc = ((s[ext] > 0) == (y[ext] == 2)).mean()
    nome, k = pedido["nome"], pedido["k"]
    tmp = SAIDA / f"{nome}_dobra{k}.tmp.npz"
    np.savez(tmp, idx=b, oof=oof, test1=lt1, test2=lt2)
    os.replace(tmp, SAIDA / f"{nome}_dobra{k}.npz")
    print(f"  {nome} dobra {k} (GPU): acerto na validacao inteira = {acc:.4f} | inferencia {(time.time()-t0)/60:.1f} min",
          flush=True)
    if testar:                                    # confere com a CPU (PyTorch) nas 300 primeiras da validacao
        from transformers import AutoModelForSequenceClassification
        cpu = AutoModelForSequenceClassification.from_pretrained(pedido["dir"])
        ref = tres_colunas(ft.prever(cpu, [seqs[j] for j in b[:300]], pad), pedido["objetivo"])
        print(f"  conferencia GPU x CPU em 300: diferenca maxima {np.abs(ref - oof[:300]).max():.4f} | "
              f"mesmo argmax {(ref.argmax(1) == oof[:300].argmax(1)).mean():.4f}", flush=True)
    else:
        shutil.rmtree(pedido["dir"], ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testar", action="store_true")
    ap.add_argument("--uma-vez", action="store_true", help="atende os pedidos pendentes e sai")
    args = ap.parse_args()
    torch.set_num_threads(2)
    FILA_GPU.mkdir(parents=True, exist_ok=True)
    TRAVA_GPU.write_text(str(os.getpid()))
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)   # a CPU e do fine-tuning
    d, cache_seqs = carregar(), {}
    print(f"previsor na GPU ativo (PID {os.getpid()}), esperando pedidos em {FILA_GPU}", flush=True)
    try:
        while True:
            pedidos = sorted(FILA_GPU.glob("*.json"), key=os.path.getmtime)
            for arq in pedidos:
                pedido = json.loads(arq.read_text())
                atender(pedido, d, cache_seqs, args.testar)
                arq.unlink()
            if args.uma_vez and not pedidos:
                break
            time.sleep(20)
    finally:
        if TRAVA_GPU.exists() and TRAVA_GPU.read_text() == str(os.getpid()):
            TRAVA_GPU.unlink()


if __name__ == "__main__":
    main()
