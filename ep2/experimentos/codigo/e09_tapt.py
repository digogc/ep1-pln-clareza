"""Experimento 9: pre-treino adaptativo a tarefa (TAPT, Gururangan et al., 2020, "Don't Stop Pretraining").

Continua o pre-treino do BERTimbau (modelagem de linguagem mascarada) nos textos do TREINO, sem rotulos, para o
modelo se acostumar com o vocabulario e o estilo das respostas do e-SIC antes do fine-tuning. Os textos do test1 e do
test2 nao sao usados (decisao conservadora, embora o TAPT nao use rotulos).

Mesmo truncamento cabeca + cauda e mesmas camadas congeladas do fine-tuning (embeddings + 4 camadas de baixo).
A perda e calculada so nas posicoes mascaradas (15%: 80% [MASK], 10% token aleatorio, 10% sem troca), o que evita
projetar todas as posicoes no vocabulario de 30 mil tokens e deixa o treino bem mais rapido em CPU.
Saida: ~/ep2_cache/bertimbau_tapt/ (modelo + tokenizador; fora do OneDrive, ~440 MB), usado depois com
       python e02_bert.py --modelo ~/ep2_cache/bertimbau_tapt --nome tapt ...
Retomavel: checkpoint a cada --ckpt-min minutos em ~/ep2_ckpt/ (o Windows reiniciou sozinho uma vez).
Uso:  python e09_tapt.py [--epocas 2 --lr 5e-5]   |   python e09_tapt.py --bench 20
"""
import argparse
import math
import time

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.utils import logging as hf_logging

from comum import CKPT, SEED, carregar, carregar_ckpt, salvar_ckpt
from e02_bert import codificar, ft, lote

hf_logging.set_verbosity_error()


def mascarar(ii, am, tok, gerador, p=0.15):
    """Sorteia as posicoes mascaradas (sem [CLS], [SEP] e padding) e devolve entrada, posicoes e alvos."""
    especial = (ii == tok.cls_token_id) | (ii == tok.sep_token_id) | (am == 0)
    sel = (torch.rand(ii.shape, generator=gerador) < p) & ~especial
    alvo = ii[sel]
    ent = ii.clone()
    r = torch.rand(ii.shape, generator=gerador)
    ent[sel & (r < 0.8)] = tok.mask_token_id
    aleat = sel & (r >= 0.8) & (r < 0.9)
    ent[aleat] = torch.randint(len(tok), (int(aleat.sum()),), generator=gerador)
    return ent, sel, alvo


def perda_mlm(modelo, ii, am, sel, alvo):
    h = modelo.base_model(input_ids=ii, attention_mask=am).last_hidden_state[sel]
    return torch.nn.functional.cross_entropy(modelo.cls(h), alvo)


@torch.no_grad()
def avaliar(modelo, seqs, tok, tam=32):
    """Perda MLM numa amostra fixa, com mascaras fixas (so para acompanhar a adaptacao ao dominio)."""
    modelo.eval()
    g = torch.Generator().manual_seed(SEED)
    tot, n = 0.0, 0
    for i in range(0, len(seqs), tam):
        ii, am = lote(seqs[i:i + tam], tok.pad_token_id)
        ent, sel, alvo = mascarar(ii, am, tok, g)
        tot += perda_mlm(modelo, ent, am, sel, alvo).item() * len(alvo); n += len(alvo)
    modelo.train()
    return tot / n


def copiar_pooler(origem, destino):
    """O modelo de MLM nao tem o pooler ([CLS] -> dense + tanh) que a classificacao usa; sem isto, o fine-tuning
    partiria de um pooler aleatorio. O TAPT nao mexe no pooler, entao copia o do BERTimbau original."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    orig = safe_open(hf_hub_download(origem, "model.safetensors"), "pt")
    arq = f"{destino}/model.safetensors"
    sd = load_file(arq)
    for k in orig.keys():
        if "pooler" in k:
            sd[k] = orig.get_tensor(k)
    save_file(sd, arq, metadata={"format": "pt"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="neuralmind/bert-base-portuguese-cased")
    ap.add_argument("--saida", default=str(CKPT.parent / "ep2_cache" / "bertimbau_tapt"))
    ap.add_argument("--epocas", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--congelar", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--ckpt-min", type=float, default=20)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    d = carregar()
    tok = AutoTokenizer.from_pretrained(args.modelo)
    ft.PAD = tok.pad_token_id
    seqs = codificar(tok, list(d["treino"]), args.max_len)
    rng = np.random.RandomState(SEED)
    monitor = [seqs[i] for i in rng.choice(len(seqs), 500, replace=False)]
    modelo = AutoModelForMaskedLM.from_pretrained(args.modelo)
    for p in modelo.base_model.embeddings.parameters():      # inclui a matriz de saida, que e a mesma (pesos atados)
        p.requires_grad = False
    for camada in modelo.base_model.encoder.layer[:args.congelar]:
        for p in camada.parameters():
            p.requires_grad = False
    params = [p for p in modelo.parameters() if p.requires_grad]
    print(f"{vars(args)} | {len(seqs)} textos do treino | parametros treinados {sum(p.numel() for p in params)/1e6:.1f} M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    passos = args.epocas * math.ceil(len(seqs) / args.lote)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * passos), passos)
    g = torch.Generator().manual_seed(SEED + 1)
    t0, passo = time.time(), 0
    treinaveis = {nm for nm, p in modelo.named_parameters() if p.requires_grad}
    ep0, i0, perm, acum = 0, 0, None, (0.0, 0)
    ck = None if args.bench else carregar_ckpt("tapt")
    if ck:
        modelo.load_state_dict(ck["modelo"], strict=False)
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        rng.set_state(ck["rng"]); g.set_state(ck["g"]); torch.set_rng_state(ck["torch_rng"])
        ep0, i0, passo, perm, acum = ck["ep"], ck["i"], ck["passo"], ck["perm"], ck["acum"]
        print(f"retomando do checkpoint: epoca {ep0+1}, passo {passo}/{passos}", flush=True)
    elif not args.bench:
        print(f"perda MLM na amostra de 500 antes do TAPT: {avaliar(modelo, monitor, tok):.4f}", flush=True)
    passo0, ultimo_ck = passo, time.time()

    def checkpoint(ep, i, perm, acum):
        salvar_ckpt("tapt", {"modelo": {nm: t for nm, t in modelo.state_dict().items() if nm in treinaveis},
                             "opt": opt.state_dict(), "sched": sched.state_dict(), "rng": rng.get_state(),
                             "g": g.get_state(), "torch_rng": torch.get_rng_state(), "ep": ep, "i": i,
                             "passo": passo, "perm": perm, "acum": acum})

    modelo.train()
    for ep in range(ep0, args.epocas):
        if ep > ep0 or i0 == 0:
            perm, inicio, (perda, n) = rng.permutation(len(seqs)), 0, (0.0, 0)
        else:
            inicio, (perda, n) = i0, acum
        for i in range(inicio, len(perm), args.lote):
            ii, am = lote([seqs[j] for j in perm[i:i + args.lote]], tok.pad_token_id)
            ent, sel, alvo = mascarar(ii, am, tok, g)
            loss = perda_mlm(modelo, ent, am, sel, alvo)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            passo += 1
            perda += loss.item(); n += 1
            if args.bench and passo == args.bench:
                print(f"BENCH: {passo * args.lote / (time.time() - t0):.2f} textos/s", flush=True)
                return
            if passo % 100 == 0:
                vel = (passo - passo0) * args.lote / (time.time() - t0)
                print(f"  epoca {ep+1} passo {passo}/{passos} | perda {perda/n:.3f} | {vel:.2f} textos/s"
                      f" | faltam ~{(passos - passo) * args.lote / vel / 60:.0f} min", flush=True)
            if args.ckpt_min and time.time() - ultimo_ck > args.ckpt_min * 60:
                checkpoint(ep, i + args.lote, perm, (perda, n)); ultimo_ck = time.time()
        print(f"fim da epoca {ep+1}: perda MLM na amostra de 500 = {avaliar(modelo, monitor, tok):.4f}", flush=True)
        if args.ckpt_min and ep < args.epocas - 1:
            checkpoint(ep + 1, 0, None, (0.0, 0)); ultimo_ck = time.time()
    modelo.save_pretrained(args.saida)
    tok.save_pretrained(args.saida)
    copiar_pooler(args.modelo, args.saida)
    (CKPT / "tapt.pt").unlink(missing_ok=True)
    print(f"modelo salvo em {args.saida} | {(time.time()-t0)/3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
