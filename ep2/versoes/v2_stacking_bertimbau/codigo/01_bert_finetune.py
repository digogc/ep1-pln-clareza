"""EP2 - fine-tuning do BERTimbau (neuralmind/bert-base-portuguese-cased) para a clareza {c1, c234, c5}.

Para servir de modelo-base do stacking, o BERT precisa de previsoes "fora da dobra" para todo o
treino. Como o fine-tuning em CPU e caro (~1,5 texto/s), usa-se validacao cruzada de 2 dobras
(semente 123): cada modelo treina em metade do treino e preve a outra metade, o test1 e o test2.
As previsoes de teste sao a media dos dois modelos.

Retomavel: cada dobra concluida fica em ep2/resultados/bert_dobra<k>.npz e e pulada ao rodar de novo.
Os rotulos do test1 nunca sao lidos aqui.
Uso:  python ep2/codigo/01_bert_finetune.py   [--epocas 2 --max-len 128 --lr 3e-5 --congelar 4]
"""
import argparse
import math
import time

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.utils import logging as hf_logging

from dados import RES, SEED, carregar

hf_logging.set_verbosity_error()


def codificar(tok, textos, max_len):
    """Truncamento cabeca + cauda: metade inicial e metade final da resposta."""
    ids = tok(list(textos), add_special_tokens=False)["input_ids"]
    meio, out = (max_len - 2) // 2, []
    for t in ids:
        if len(t) > max_len - 2:
            t = t[:meio] + t[-(max_len - 2 - meio):]
        out.append([tok.cls_token_id] + t + [tok.sep_token_id])
    return out


def lote(seqs, pad):
    m = max(len(s) for s in seqs)
    ii = torch.tensor([s + [pad] * (m - len(s)) for s in seqs])
    am = torch.tensor([[1] * len(s) + [0] * (m - len(s)) for s in seqs])
    return ii, am


@torch.no_grad()
def prever(modelo, seqs, pad, tam=32):
    modelo.eval()
    ordem = np.argsort([len(s) for s in seqs])          # lotes de tamanhos parecidos = menos padding
    out = np.zeros((len(seqs), 3), dtype=np.float32)
    for i in range(0, len(seqs), tam):
        idx = ordem[i:i + tam]
        ii, am = lote([seqs[j] for j in idx], pad)
        out[idx] = modelo(input_ids=ii, attention_mask=am).logits.numpy()
    return out


def treinar_dobra(k, a, b, d, seqs, args, tok):
    torch.manual_seed(SEED + k)
    modelo = AutoModelForSequenceClassification.from_pretrained(args.modelo, num_labels=3)
    for p in modelo.base_model.embeddings.parameters():
        p.requires_grad = False
    for camada in modelo.base_model.encoder.layer[:args.congelar]:
        for p in camada.parameters():
            p.requires_grad = False
    params = [p for p in modelo.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    passos = args.epocas * math.ceil(len(a) / args.lote)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * passos), passos)
    y = torch.tensor(d["y"])
    rng = np.random.RandomState(SEED + k)
    monitor = rng.choice(b, 600, replace=False)          # amostra fixa da metade de validacao
    t0, passo = time.time(), 0
    for ep in range(args.epocas):
        modelo.train()
        perm = rng.permutation(a)
        perda, acertos, n = 0.0, 0, 0
        for i in range(0, len(perm), args.lote):
            idx = perm[i:i + args.lote]
            ii, am = lote([seqs[j] for j in idx], tok.pad_token_id)
            out = modelo(input_ids=ii, attention_mask=am, labels=y[idx])
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            passo += 1
            perda += out.loss.item() * len(idx); acertos += (out.logits.argmax(1) == y[idx]).sum().item(); n += len(idx)
            if passo % 50 == 0:
                vel = passo * args.lote / (time.time() - t0)
                falta = (passos - passo) * args.lote / vel / 60
                print(f"  dobra {k} epoca {ep+1} passo {passo}/{passos} | perda {perda/n:.3f} acerto-treino {acertos/n:.3f}"
                      f" | {vel:.2f} textos/s | faltam ~{falta:.0f} min de treino", flush=True)
        acc = (prever(modelo, [seqs[j] for j in monitor], tok.pad_token_id).argmax(1) == d["y"][monitor]).mean()
        print(f"  dobra {k} fim da epoca {ep+1}: acerto em 600 linhas da metade de validacao = {acc:.4f}", flush=True)
    t1 = time.time()
    oof = prever(modelo, [seqs[j] for j in b], tok.pad_token_id)
    n = len(d["treino"])
    lt1 = prever(modelo, seqs[n:n + len(d["test1"])], tok.pad_token_id)
    lt2 = prever(modelo, seqs[n + len(d["test1"]):], tok.pad_token_id)
    print(f"  dobra {k}: acerto na metade de validacao inteira = {(oof.argmax(1) == d['y'][b]).mean():.4f}"
          f" | inferencia {(time.time()-t1)/60:.0f} min", flush=True)
    np.savez(RES / f"bert_dobra{k}.npz", idx=b, oof=oof, test1=lt1, test2=lt2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="neuralmind/bert-base-portuguese-cased")
    ap.add_argument("--dobras", type=int, default=2)
    ap.add_argument("--epocas", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--congelar", type=int, default=4, help="camadas inferiores congeladas (alem dos embeddings)")
    args = ap.parse_args()
    torch.set_num_threads(4)
    RES.mkdir(exist_ok=True)
    d = carregar()
    n = len(d["treino"])
    tok = AutoTokenizer.from_pretrained(args.modelo)
    seqs = codificar(tok, list(d["treino"]) + list(d["test1"]) + list(d["test2"]), args.max_len)
    print(f"{n} treino, {len(d['test1'])} test1, {len(d['test2'])} test2 | {vars(args)}", flush=True)
    dobras = list(StratifiedKFold(args.dobras, shuffle=True, random_state=SEED).split(d["y"], d["y"]))
    for k, (a, b) in enumerate(dobras):
        if (RES / f"bert_dobra{k}.npz").exists():
            print(f"dobra {k} ja concluida, pulando", flush=True)
            continue
        t0 = time.time()
        treinar_dobra(k, a, b, d, seqs, args, tok)
        print(f"dobra {k} concluida em {(time.time()-t0)/3600:.1f} h", flush=True)

    oof, t1, t2 = np.zeros((n, 3), np.float32), 0, 0
    for k in range(args.dobras):
        z = np.load(RES / f"bert_dobra{k}.npz")
        oof[z["idx"]] = z["oof"]; t1 = t1 + z["test1"] / args.dobras; t2 = t2 + z["test2"] / args.dobras
    np.save(RES / "bert_oof.npy", oof); np.save(RES / "bert_test1.npy", t1); np.save(RES / "bert_test2.npy", t2)
    accs = [(oof[b].argmax(1) == d["y"][b]).mean() for _, b in dobras]
    print(f"BERTimbau sozinho, validacao cruzada de {args.dobras} dobras: {np.mean(accs):.4f} | por dobra {np.round(accs, 4)}")


if __name__ == "__main__":
    main()
