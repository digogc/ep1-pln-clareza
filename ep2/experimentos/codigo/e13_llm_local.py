"""Experimento 13: LLM local (Qwen3-1.7B, int4) na GPU integrada, zero-shot, como bloco do stacking.

Alternativa gratuita a LLM via API: o modelo roda no proprio computador (OpenVINO, Iris Xe), entao nenhum texto sai
da maquina. Para cada resposta, uma unica passada pelo modelo, sem gerar texto: o prompt pede a nota de 1 a 5 que o
cidadao deu, e os atributos sao as probabilidades dos tokens "1" a "5" logo depois da abertura da resposta do
assistente (modo sem raciocinio do Qwen3). Zero-shot e sem a rubrica do e06 (que foi tirada dos dados): nenhum
rotulo entra no prompt, portanto nao ha vazamento.

piloto: as mesmas 400 respostas da dobra 1 do e06 -> velocidade e sinal (RL em validacao cruzada nas 400, x BERT).
completo: treino + test1 + test2 em partes retomaveis (~/ep2_ckpt) -> bloco llm_local_<prompt> e avaliacao no stacking.
Roda no ambiente ~/ep2_ov:  ~/ep2_ov/Scripts/python e13_llm_local.py piloto
                            ~/ep2_ov/Scripts/python e13_llm_local.py completo --prompt nota
"""
import argparse
import os
import time

import numpy as np
import psutil
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from transformers import AutoTokenizer

from comum import CKPT, SEED, V2, blocos_v2, carregar, cv_stacking, dobras, fmt, salvar_bloco
from e06_llm import BASE, CIDADAO

MODELO = CKPT.parent / "ep2_cache" / "qwen3_1.7b_ov"
MAX_TOK = 384                                    # resposta truncada em cabeca + cauda (192 + 192 tokens)
PROMPTS = {
    "nota": BASE + "\nResponda apenas com a nota de 1 a 5 que o cidadão provavelmente deu à clareza da resposta.",
    "cidadao": CIDADAO + "Responda apenas com um número de 1 a 5.",
}


class Avaliador:
    def __init__(self, prompt):
        from optimum.intel import OVModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(MODELO)
        self.modelo = OVModelForCausalLM.from_pretrained(MODELO, device="GPU")
        self.sistema = PROMPTS[prompt]
        self.digitos = [self.tok.convert_tokens_to_ids(str(k)) for k in range(1, 6)]

    def entrada(self, texto):
        ids = self.tok(texto, add_special_tokens=False)["input_ids"]
        if len(ids) > MAX_TOK:
            texto = self.tok.decode(ids[:MAX_TOK // 2]) + " [...] " + self.tok.decode(ids[-MAX_TOK // 2:])
        msgs = [{"role": "system", "content": self.sistema}, {"role": "user", "content": f"Resposta do órgão:\n{texto}"}]
        return self.tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)

    @torch.no_grad()
    def __call__(self, texto):
        enc = self.tok(self.entrada(texto), return_tensors="pt")
        lp = torch.log_softmax(self.modelo(**enc).logits[0, -1].float(), -1)
        d = lp[self.digitos]
        return np.r_[(d - torch.logsumexp(d, 0)).numpy(), torch.logsumexp(d, 0).exp().item()]   # 5 log-probs + massa


def atributos(A):
    """log-probs normalizadas das notas 1-5 e nota esperada."""
    p = np.exp(A[:, :5])
    return np.c_[A[:, :5], p @ np.arange(1, 6)]


def piloto(args, d):
    y, dobra = d["y"], dobras(d["y"])
    cand = np.where(dobra == 0)[0]
    idx, _ = train_test_split(cand, train_size=400, stratify=y[cand], random_state=SEED)
    av = Avaliador(args.prompt)
    t0 = time.time()
    A = np.array([av(d["treino"][i]) for i in idx])
    dt = time.time() - t0
    np.save(CKPT / f"llm_local_{args.prompt}_piloto.npy", A)
    X, nota = atributos(A), atributos(A)[:, 5]
    fora, _, _ = blocos_v2()
    bert = fora["bert"][idx, 0]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rl = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    print(f"prompt {args.prompt}: {len(idx)} respostas em {dt/60:.1f} min = {len(idx)/dt:.2f} respostas/s "
          f"| massa media nos digitos 1-5: {A[:, 5].mean():.3f}")
    print(f"  distribuicao da nota mais provavel: {np.bincount(A[:, :5].argmax(1), minlength=5)}")
    print(f"  Spearman com o rotulo: LLM {spearmanr(nota, y[idx])[0]:.3f} | BERT da v2 {spearmanr(bert, y[idx])[0]:.3f}"
          f" | LLM x BERT {spearmanr(nota, bert)[0]:.3f}")
    print(f"  RL (5 dobras nas 400): LLM {fmt(cross_val_score(rl, X, y[idx], cv=cv))}")
    print(f"                         BERT {fmt(cross_val_score(rl, bert[:, None], y[idx], cv=cv))}")
    print(f"                   BERT + LLM {fmt(cross_val_score(rl, np.c_[bert, X], y[idx], cv=cv))}")


def piloto2(args, d):
    """Piloto ampliado: as 400 do piloto + 1600 outras da dobra 1 -> ganho dentro do stacking completo da v2
    (5 dobras nas 2000 respostas; comparacao pareada v2 x v2 + LLM)."""
    y, dobra = d["y"], dobras(d["y"])
    cand = np.where(dobra == 0)[0]
    idx1, _ = train_test_split(cand, train_size=400, stratify=y[cand], random_state=SEED)
    resto = np.setdiff1d(cand, idx1)
    idx2, _ = train_test_split(resto, train_size=1600, stratify=y[resto], random_state=SEED)
    arq = CKPT / f"llm_local_{args.prompt}_piloto2.npy"
    if not arq.exists():
        av, t0 = Avaliador(args.prompt), time.time()
        A2 = []
        for i, j in enumerate(idx2):
            A2.append(av(d["treino"][j]))
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/1600 | {(time.time()-t0)/60:.0f} min", flush=True)
        np.save(arq, np.array(A2))
    idx = np.r_[idx1, idx2]
    X = atributos(np.r_[np.load(CKPT / f"llm_local_{args.prompt}_piloto.npy"), np.load(arq)])
    fora, _, _ = blocos_v2()
    v2 = np.hstack([fora[b][idx] for b in V2])
    yy, dd = y[idx], np.zeros(len(idx), int)
    for k, (_, v) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(yy, yy)):
        dd[v] = k
    a, b2 = cv_stacking(v2, yy, dd), cv_stacking(np.hstack([v2, X]), yy, dd)
    print(f"piloto ampliado ({len(idx)} respostas da dobra 1), stacking em 5 dobras nelas:")
    print("  LLM sozinho      ", fmt(cv_stacking(X, yy, dd)))
    print("  v2               ", fmt(a))
    print("  v2 + LLM         ", fmt(b2))
    print(f"  ganho por dobra: {np.round((b2 - a) * 100, 2)} | media {(b2 - a).mean()*100:+.2f} +- {(b2 - a).std(ddof=1)*100/np.sqrt(5):.2f}")


def completo(args, d):
    textos = list(np.r_[d["treino"], d["test1"], d["test2"]])
    n, av, partes, t0 = len(d["treino"]), None, [], time.time()
    for j, ini in enumerate(range(0, len(textos), args.parte)):
        arq = CKPT / f"llm_local_{args.prompt}_parte{j}.npy"
        if not arq.exists():
            av = av or Avaliador(args.prompt)
            np.save(arq, np.array([av(t) for t in textos[ini:ini + args.parte]]))
            print(f"  parte {j+1}/{-(-len(textos) // args.parte)} pronta | {(time.time()-t0)/60:.0f} min", flush=True)
        partes.append(np.load(arq))
    X = atributos(np.concatenate(partes))
    y, dobra = d["y"], dobras(d["y"])
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    fora_llm = np.zeros((n, 3))
    for k in range(5):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(X[:n][dobra != k], y[dobra != k])
        fora_llm[dobra == k] = m.predict_proba(X[:n][dobra == k])
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(X[:n], y)
    nome = f"llm_local_{args.prompt}"
    salvar_bloco(nome, np.log(fora_llm + 1e-6), np.log(m.predict_proba(X[n:]) + 1e-6))
    fora, _, _ = blocos_v2()
    v2 = np.hstack([fora[b] for b in V2])
    print(f"{nome} sozinho (RL)  ", fmt(cv_stacking(X[:n], y, dobra)))
    print("v2                ", fmt(cv_stacking(v2, y, dobra)))
    print(f"v2 + {nome}", fmt(cv_stacking(np.hstack([v2, np.log(fora_llm + 1e-6)]), y, dobra)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("acao", choices=["piloto", "piloto2", "completo"])
    ap.add_argument("--prompt", choices=list(PROMPTS), default="nota")
    ap.add_argument("--parte", type=int, default=500)
    args = ap.parse_args()
    torch.set_num_threads(1)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    CKPT.mkdir(exist_ok=True)
    d = carregar()
    {"piloto": piloto, "piloto2": piloto2, "completo": completo}[args.acao](args, d)


if __name__ == "__main__":
    main()
