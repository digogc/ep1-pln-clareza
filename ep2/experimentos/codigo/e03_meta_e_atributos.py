"""Experimento 3: (a) variacoes do meta-classificador; (b) bloco de atributos escritos a mao (estrutura,
tamanho, legibilidade, marcas de encaminhamento/anexo/recurso) com LightGBM.
"""
import re
import time

import lightgbm as lgb
import numpy as np

from comum import N_DOBRAS, SEED, V2, blocos_v2, carregar, cv_stacking, dobras, fmt, salvar_bloco

PADROES = {
    "url": r"https?://|www\.", "email": r"\S+@\S+", "numero": r"\d+", "lei": r"\blei\b|\bart\.|\bartigo|decreto|12\.527|7\.724",
    "anexo": r"anex", "recurso": r"recurso", "prazo": r"\bprazo", "encaminh": r"encaminh", "nao": r"\bn[ãa]o\b",
    "informamos": r"informamos|esclarecemos", "solicitacao": r"solicita[çc][ãa]o|pedido|demanda", "link": r"\blink|\bs[íi]tio|\bsite\b|portal",
    "contato": r"telefone|\btel\b|e-mail|email|ramal", "ola": r"\bol[áa]\b|bom dia|boa tarde", "prezado": r"\bprezad",
    "senhor": r"senhor|senhora|vossa", "atenciosamente": r"atenciosamente|cordialmente", "sic": r"\bsic\b|e-sic|ouvidoria",
    "negativa": r"n[ãa]o (foi poss[íi]vel|possui|h[áa]|dispomos|compete|cabe)|indeferid|sigilo|pessoal", "sim": r"\bsim\b",
    "enumeracao": r"(?:^|\s)(?:\d{1,2}[\)\.-]|[a-e]\))\s", "pergunta": r"\?", "exclamacao": r"!", "dois_pontos": r":",
}


def atributos(textos):
    linhas = []
    for t in textos:
        tl = t.lower()
        palavras = re.findall(r"\w+", t)
        frases = [f for f in re.split(r"[.!?]+\s", t) if f.strip()]
        silabas = sum(max(1, len(re.findall(r"[aeiouáéíóúâêôãõà]+", p.lower()))) for p in palavras) if palavras else 0
        np_, nf = max(len(palavras), 1), max(len(frases), 1)
        f = [np.log1p(len(t)), np.log1p(len(palavras)), np.log1p(len(frases)), np_ / nf,
             np.mean([len(p) for p in palavras]) if palavras else 0, silabas / np_,
             248.835 - 1.015 * (np_ / nf) - 84.6 * (silabas / np_),            # Flesch adaptado ao portugues
             sum(c.isdigit() for c in t) / max(len(t), 1), sum(c.isupper() for c in t) / max(len(t), 1),
             len(set(w.lower() for w in palavras)) / np_]
        f += [len(re.findall(p, tl)) for p in PADROES.values()]
        linhas.append(f)
    return np.array(linhas, np.float32)


def lgbm():
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=2)


def main():
    t0 = time.time()
    d = carregar()
    y, n = d["y"], len(d["treino"])
    dobra = dobras(y)
    fora, teste, _ = blocos_v2()
    base = np.hstack([fora[b] for b in V2])
    print("(a) meta-classificador sobre os blocos da v2 (media | dobras 1-5):")
    for C in (0.03, 0.1, 0.3):
        print(f"  regressao logistica C={C:<5}", fmt(cv_stacking(base, y, dobra, C)))
    acc = []
    for k in range(N_DOBRAS):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=7, min_child_samples=100,
                               random_state=SEED, verbose=-1, n_jobs=2).fit(base[dobra != k], y[dobra != k])
        acc.append((m.predict(base[dobra == k]) == y[dobra == k]).mean())
    print("  LightGBM                   ", fmt(np.array(acc)))

    A = atributos(np.r_[d["treino"], d["test1"], d["test2"]])
    print(f"\n(b) {A.shape[1]} atributos escritos a mao ({time.time()-t0:.0f}s)")
    oof = np.zeros((n, 3))
    for k in range(N_DOBRAS):
        oof[dobra == k] = lgbm().fit(A[:n][dobra != k], y[dobra != k]).predict_proba(A[:n][dobra == k])
    tst = lgbm().fit(A[:n], y).predict_proba(A[n:])
    salvar_bloco("atributos", np.log(oof + 1e-6), np.log(tst + 1e-6))
    print("  LightGBM sozinho nos atributos:", f"{(oof.argmax(1) == y).mean()*100:.2f}")
    print("  v2 + atributos             ", fmt(cv_stacking(np.hstack([base, np.log(oof + 1e-6)]), y, dobra)))
    print(f"tempo {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
