"""Experimento 2: fine-tuning do BERTimbau com LoRA (ou fine-tuning completo) e mais dobras.

LoRA (Hu et al., 2021) implementado a mao (sem a biblioteca peft): cada camada linear das camadas do topo
ganha um desvio de posto baixo B.A (r x d), treinado; os pesos originais ficam congelados. Em CPU isso tambem
acelera o treino, porque o gradiente dos pesos congelados nao e calculado.

Mesmo protocolo do 01_bert_finetune.py (truncamento cabeca + cauda, semente 123, previsoes fora da dobra),
com as saidas em ep2/experimentos/resultados/<nome>_dobra<k>.npz.
Uso:  python e02_bert.py --nome lora16 --lora 16 --lr 3e-4            (todas as dobras)
      python e02_bert.py --nome lora16 --lora 16 --lr 3e-4 --so-dobra 0
      python e02_bert.py --lora 16 --bench 20                          (so mede a velocidade)
"""
import argparse
import importlib.util
import json
import math
import time

import numpy as np
import psutil
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.utils import logging as hf_logging

from comum import CKPT, EXP, SAIDA, SEED, carregar, carregar_ckpt, salvar_ckpt

hf_logging.set_verbosity_error()
_spec = importlib.util.spec_from_file_location("ft", EXP.parent / "codigo" / "01_bert_finetune.py")
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)
codificar, lote, prever = ft.codificar, ft.lote, ft.prever
# previsoes na GPU integrada (e12_previsor_gpu.py): pedidos em FILA_GPU, modelos ajustados em MODELOS
FILA_GPU, TRAVA_GPU, MODELOS = CKPT / "fila_gpu", CKPT / "previsor_gpu.pid", CKPT.parent / "ep2_cache" / "modelos"


def gpu_ativa():
    try:
        return psutil.pid_exists(int(TRAVA_GPU.read_text()))
    except (OSError, ValueError):
        return False


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.escala = alpha / r
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.t() @ self.B.t()) * self.escala


def preparar(args):
    modelo = AutoModelForSequenceClassification.from_pretrained(args.modelo, num_labels=3 if args.objetivo == "ce" else 1)
    enc = modelo.base_model
    for p in enc.embeddings.parameters():
        p.requires_grad = False
    for camada in enc.encoder.layer[:args.congelar]:
        for p in camada.parameters():
            p.requires_grad = False
    if args.lora:
        for camada in enc.encoder.layer[args.congelar:]:
            for p in camada.parameters():
                p.requires_grad = False
            at = camada.attention
            for pai, nome in ((at.self, "query"), (at.self, "key"), (at.self, "value"), (at.output, "dense"),
                              (camada.intermediate, "dense"), (camada.output, "dense")):
                setattr(pai, nome, LoRALinear(getattr(pai, nome), args.lora, 2 * args.lora, args.lora_dropout))
    return modelo


def prever3(modelo, seqs, args):
    """Logits no formato (n, 3). Nos objetivos de escore unico s, devolve (-s/2, 0, s/2): assim c5 - c1 = s."""
    out = prever(modelo, seqs, ft.PAD)
    if args.objetivo == "ce":
        return out
    s = out[:, 0]
    return np.c_[-s / 2, np.zeros_like(s), s / 2].astype(np.float32)


def acerto(logits, y, objetivo):
    """3 classes: acerto do argmax. Escore unico: acerto do sinal em c1 x c5 (so nos extremos)."""
    if objetivo == "ce":
        return float((logits.argmax(1) == y).mean())
    s, ext = logits[:, 2] - logits[:, 0], y != 1
    return float(((s[ext] > 0) == (y[ext] == 2)).mean())


def ordem_agrupada(rng, a, comp, lote, mega=50):
    """Lotes de textos de tamanho parecido (como o group_by_length do Hugging Face): embaralha, ordena por tamanho
    dentro de blocos de mega*lote textos e embaralha a ordem dos lotes. Com o preenchimento dinamico, corta o custo
    dos textos curtos sem perder a aleatoriedade. Os lotes cheios vem primeiro, entao cada fatia de `lote` e um lote."""
    p = rng.permutation(a)
    lotes = []
    for s in range(0, len(p), mega * lote):
        g = p[s:s + mega * lote]
        g = g[np.argsort(-comp[g], kind="stable")]
        lotes += [g[i:i + lote] for i in range(0, len(g), lote)]
    cheios, resto = [x for x in lotes if len(x) == lote], [x for x in lotes if len(x) < lote]
    return np.concatenate([cheios[i] for i in rng.permutation(len(cheios))] + resto)


def treinar_dobra(k, a, b, d, seqs, args):
    torch.manual_seed(SEED + k)
    if args.objetivo == "extremos":
        a = a[d["y"][a] != 1]
    if args.limite:                                   # teste rapido do codigo
        a, b = a[:args.limite], b[:max(args.limite, 600)]
    modelo = preparar(args)
    params = [p for p in modelo.parameters() if p.requires_grad]
    print(f"  parametros treinados: {sum(p.numel() for p in params)/1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    passos = args.epocas * math.ceil(len(a) / args.lote)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * passos), passos)
    y = torch.tensor(d["y"])
    if args.objetivo == "ce":
        ce = nn.CrossEntropyLoss(label_smoothing=args.suavizacao)
        perda_fn = lambda lg, yy: ce(lg, yy)                                    # noqa: E731
    else:                                                                       # alvo ordinal 0 / 0,5 / 1
        bce = nn.BCEWithLogitsLoss()
        perda_fn = lambda lg, yy: bce(lg[:, 0], yy.float() / 2)                 # noqa: E731
    rng = np.random.RandomState(SEED + k)
    monitor = rng.choice(b, 600, replace=False)
    comp = np.array([len(s) for s in seqs])
    t0, passo = time.time(), 0
    nome_ck = f"{args.nome}_dobra{k}"
    treinaveis = {nm for nm, p in modelo.named_parameters() if p.requires_grad}
    ep0, i0, perm, acum = 0, 0, None, (0.0, 0, 0)
    ck = None if args.bench else carregar_ckpt(nome_ck)
    if ck:                                            # retoma do ultimo checkpoint (mesma ordem dos lotes)
        modelo.load_state_dict(ck["modelo"], strict=False)
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        rng.set_state(ck["rng"]); torch.set_rng_state(ck["torch_rng"])
        ep0, i0, passo, perm, acum = ck["ep"], ck["i"], ck["passo"], ck["perm"], ck["acum"]
        print(f"  retomando a dobra {k} do checkpoint: epoca {ep0+1}, passo {passo}/{passos}", flush=True)
    passo0, ultimo_ck = passo, time.time()

    def checkpoint(ep, i, perm, acum):
        salvar_ckpt(nome_ck, {"modelo": {nm: t for nm, t in modelo.state_dict().items() if nm in treinaveis},
                              "opt": opt.state_dict(), "sched": sched.state_dict(), "rng": rng.get_state(),
                              "torch_rng": torch.get_rng_state(), "ep": ep, "i": i, "passo": passo,
                              "perm": perm, "acum": acum})

    for ep in range(ep0, args.epocas):
        modelo.train()
        if ep > ep0 or i0 == 0:
            perm, inicio = (ordem_agrupada(rng, a, comp, args.lote) if args.agrupar else rng.permutation(a)), 0
            perda, acertos, n = 0.0, 0, 0
        else:
            inicio, (perda, acertos, n) = i0, acum
        for i in range(inicio, len(perm), args.lote):
            idx = perm[i:i + args.lote]
            ii, am = lote([seqs[j] for j in idx], ft.PAD)
            logits = modelo(input_ids=ii, attention_mask=am).logits
            loss = perda_fn(logits, y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            passo += 1
            perda += loss.item() * len(idx); n += len(idx)
            lg = logits.detach().numpy()
            acertos += acerto(lg if args.objetivo == "ce" else np.c_[-lg, 0 * lg, lg], d["y"][idx], args.objetivo) * len(idx)
            if args.bench and passo == args.bench:
                print(f"BENCH: {passo * args.lote / (time.time() - t0):.2f} textos/s", flush=True)
                return
            if passo % 50 == 0:
                vel = (passo - passo0) * args.lote / (time.time() - t0)
                print(f"  dobra {k} epoca {ep+1} passo {passo}/{passos} | perda {perda/n:.3f} acerto-treino {acertos/n:.3f}"
                      f" | {vel:.2f} textos/s | faltam ~{(passos - passo) * args.lote / vel / 60:.0f} min", flush=True)
            if args.ckpt_min and time.time() - ultimo_ck > args.ckpt_min * 60:
                checkpoint(ep, i + args.lote, perm, (perda, acertos, n))
                ultimo_ck = time.time()
        if args.prever_cada_epoca and ep < args.epocas - 1:
            gravar(modelo, k, b, d, seqs, f"{args.nome}_ep{ep+1}", args)
        else:
            acc = acerto(prever3(modelo, [seqs[j] for j in monitor], args), d["y"][monitor], args.objetivo)
            print(f"  dobra {k} fim da epoca {ep+1}: acerto em 600 linhas da validacao = {acc:.4f}", flush=True)
        if args.ckpt_min and ep < args.epocas - 1:
            checkpoint(ep + 1, 0, None, (0.0, 0, 0)); ultimo_ck = time.time()
    gravar(modelo, k, b, d, seqs, args.nome, args)
    (CKPT / f"{nome_ck}.pt").unlink(missing_ok=True)


def gravar(modelo, k, b, d, seqs, nome, args):
    """Previsoes da metade de validacao (fora da dobra), do test1 e do test2. Com o previsor da GPU ativo,
    so grava o modelo e o pedido; a GPU preve enquanto a CPU segue treinando."""
    if gpu_ativa() and not args.lora:
        destino = MODELOS / f"{nome}_dobra{k}"
        modelo.save_pretrained(destino)
        np.save(destino / "idx.npy", b)
        FILA_GPU.mkdir(parents=True, exist_ok=True)
        pedido = {"nome": nome, "k": k, "dir": str(destino), "tok": args.modelo, "max_len": args.max_len,
                  "objetivo": args.objetivo, "limite": args.limite}
        tmp = FILA_GPU / f"{nome}_dobra{k}.tmp"
        tmp.write_text(json.dumps(pedido))
        tmp.replace(FILA_GPU / f"{nome}_dobra{k}.json")
        print(f"  {nome} dobra {k}: modelo gravado, previsoes enviadas para a GPU", flush=True)
        return
    t1 = time.time()
    n = len(d["treino"])
    lim = args.limite or len(seqs)
    oof = prever3(modelo, [seqs[j] for j in b], args)
    lt1 = prever3(modelo, seqs[n:n + len(d["test1"])][:lim], args)
    lt2 = prever3(modelo, seqs[n + len(d["test1"]):][:lim], args)
    print(f"  {nome} dobra {k}: acerto na validacao inteira = {acerto(oof, d['y'][b], args.objetivo):.4f}"
          f" | inferencia {(time.time()-t1)/60:.0f} min", flush=True)
    np.savez(SAIDA / f"{nome}_dobra{k}.npz", idx=b, oof=oof, test1=lt1, test2=lt2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nome", default="bert_exp")
    ap.add_argument("--modelo", default="neuralmind/bert-base-portuguese-cased")
    ap.add_argument("--dobras", type=int, default=2)
    ap.add_argument("--so-dobra", type=int, default=None)
    ap.add_argument("--epocas", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--congelar", type=int, default=4)
    ap.add_argument("--lora", type=int, default=0, help="posto do LoRA (0 = fine-tuning completo)")
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    ap.add_argument("--suavizacao", type=float, default=0.0, help="label smoothing")
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--prever-cada-epoca", action="store_true", help="grava previsoes tambem ao fim de cada epoca")
    ap.add_argument("--agrupar", action="store_true", help="lotes agrupados por tamanho do texto (menos preenchimento)")
    ap.add_argument("--limite", type=int, default=0, help="so para testar o codigo: usa poucas linhas")
    ap.add_argument("--objetivo", choices=["ce", "ordinal", "extremos"], default="ce",
                    help="ce: 3 classes | ordinal: escore unico com alvo 0/0,5/1 | extremos: escore c1 x c5 so nos extremos")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ckpt-min", type=float, default=20, help="minutos entre checkpoints (0 = sem checkpoint)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    d = carregar()
    n = len(d["treino"])
    tok = AutoTokenizer.from_pretrained(args.modelo)
    ft.PAD = tok.pad_token_id
    seqs = codificar(tok, list(d["treino"]) + list(d["test1"]) + list(d["test2"]), args.max_len)
    print(f"{vars(args)}", flush=True)
    dobras = list(StratifiedKFold(args.dobras, shuffle=True, random_state=SEED).split(d["y"], d["y"]))
    for k, (a, b) in enumerate(dobras):
        if args.so_dobra is not None and k != args.so_dobra:
            continue
        if (SAIDA / f"{args.nome}_dobra{k}.npz").exists() or (FILA_GPU / f"{args.nome}_dobra{k}.json").exists():
            print(f"dobra {k} ja concluida, pulando", flush=True)
            continue
        t0 = time.time()
        treinar_dobra(k, a, b, d, seqs, args)
        if args.bench:
            return
        print(f"dobra {k} concluida em {(time.time()-t0)/3600:.1f} h", flush=True)
    if all((SAIDA / f"{args.nome}_dobra{k}.npz").exists() for k in range(args.dobras)):
        oof, t1, t2 = np.zeros((n, 3), np.float32), 0, 0
        for k in range(args.dobras):
            z = np.load(SAIDA / f"{args.nome}_dobra{k}.npz")
            oof[z["idx"]] = z["oof"]; t1 = t1 + z["test1"] / args.dobras; t2 = t2 + z["test2"] / args.dobras
        np.save(SAIDA / f"{args.nome}_oof.npy", oof); np.save(SAIDA / f"{args.nome}_test1.npy", t1)
        np.save(SAIDA / f"{args.nome}_test2.npy", t2)
        accs = [(oof[b].argmax(1) == d["y"][b]).mean() for _, b in dobras]
        print(f"{args.nome} sozinho, {args.dobras} dobras: {np.mean(accs):.4f} | por dobra {np.round(accs, 4)}")


if __name__ == "__main__":
    main()
