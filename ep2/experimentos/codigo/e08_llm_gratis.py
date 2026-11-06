"""Experimento 8: o piloto de engenharia de prompt do e06, numa API gratuita compativel com a da OpenAI
(Cerebras ou Groq, com modelos abertos como Llama e Qwen).

Mesmas garantias contra vazamento do e06 (usa a mesma amostra): as respostas avaliadas sao da dobra 1 do treino,
os exemplos dos prompts few-shot vem das dobras 2-5 (com assert de que nenhum texto se repete), nada do test1/test2
e enviado e a escolha do prompt e feita so no treino.

Cota gratuita: os pedidos vao em ordem resposta a resposta (todos os prompts de uma resposta antes da proxima), assim,
se a cota diaria acabar no meio, todos os prompts foram avaliados nas mesmas respostas. O progresso fica num .jsonl e
rodar de novo continua de onde parou.

Uso:  python e08_llm_gratis.py modelos                     lista os modelos disponiveis no provedor
      python e08_llm_gratis.py rodar   [--modelo M] [--n 300]
      python e08_llm_gratis.py avaliar [--modelo M]
Opcoes: --provedor cerebras|groq (padrao cerebras), --rpm pedidos por minuto (padrao 20).
Credencial: variavel CEREBRAS_API_KEY ou GROQ_API_KEY (de usuario, no Windows). A chave nunca e impressa.
"""
import argparse
import json
import re
import sys
import time

import numpy as np

from comum import RES, ROTULOS, SEED, carregar
from e06_llm import BASE, CIDADAO, DIR, RUBRICA, TAREFA_CLASSE, TAREFA_NOTA, TAREFA_PROBS, amostras, cortar

PROVEDORES = {"cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
              "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY")}

FORMATO = {
    "classe": '\n\nResponda só com um JSON no formato {"classe": "c1" | "c234" | "c5"}.',
    "nota": '\n\nResponda só com um JSON no formato {"nota": 1 a 5, "classe": "c1" | "c234" | "c5"}.',
    "probs": '\n\nResponda só com um JSON no formato {"p_c1": inteiro, "p_c234": inteiro, "p_c5": inteiro}.',
    "cot": ('\n\nAntes de decidir, analise a resposta em no máximo três frases: ela responde o que foi pedido? '
            'é direta ou burocrática? o cidadão ficaria satisfeito?\n'
            'Responda só com um JSON no formato {"analise": "...", "classe": "c1" | "c234" | "c5"}.'),
}

# nome: (texto do sistema, formato, n de exemplos por classe)
PROMPTS = {
    "zero_simples": (BASE + TAREFA_CLASSE, "classe", 0),
    "zero_rubrica": (BASE + RUBRICA + TAREFA_NOTA, "nota", 0),
    "zero_cidadao": (BASE + CIDADAO + TAREFA_NOTA, "nota", 0),
    "zero_probs":   (BASE + RUBRICA + TAREFA_PROBS, "probs", 0),
    "zero_cot":     (BASE + RUBRICA + TAREFA_CLASSE, "cot", 0),
    "few_12":       (BASE + TAREFA_CLASSE, "classe", 4),
}


def sistema(nome, d, exemplos):
    texto, formato, k = PROMPTS[nome]
    if k:
        X, y = d["treino"], d["y"]
        sel = list(np.random.RandomState(SEED).permutation([i for c in range(3) for i in exemplos[c][:k]]))
        texto += "\n\nExemplos de respostas reais já avaliadas pelos cidadãos:\n" + "".join(
            f"\n<exemplo>\n<resposta>{cortar(X[i], 1000)}</resposta>\n<classe>{ROTULOS[y[i]]}</classe>\n</exemplo>\n" for i in sel)
    return texto + FORMATO[formato]


def chave(var):
    import os
    valor = os.environ.get(var)
    if not valor and sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                valor = winreg.QueryValueEx(k, var)[0]
        except OSError:
            pass
    if not valor:
        sys.exit(f"defina a variavel de usuario {var} com a chave da API (a chave nunca e impressa)")
    return valor


def cliente(provedor):
    from openai import OpenAI
    url, var = PROVEDORES[provedor]
    return OpenAI(base_url=url, api_key=chave(var), max_retries=0)


def ler_json(txt):
    """Extrai o ultimo objeto JSON do texto (alguns modelos escrevem algo antes)."""
    if not txt:
        return None
    for m in reversed(re.findall(r"\{[^{}]*\}", txt, flags=re.S)):
        try:
            return json.loads(m)
        except json.JSONDecodeError:
            continue
    return None


def classe_de(s):
    try:
        if "p_c1" in s:
            return int(np.argmax([float(s["p_c1"]), float(s["p_c234"]), float(s["p_c5"])]))
        return ROTULOS.index(str(s["classe"]).strip().lower())
    except (KeyError, TypeError, ValueError):
        return None


def arquivo(args):
    return DIR / f"gratis_{args.provedor}_{re.sub(r'[^A-Za-z0-9.-]+', '_', args.modelo)}.jsonl"


def rodar(args, d, piloto, exemplos):
    from openai import APIStatusError, RateLimitError
    c = cliente(args.provedor)
    saida = arquivo(args)
    feitos = set()
    if saida.exists():
        for linha in open(saida, encoding="utf-8"):
            r = json.loads(linha)
            if r.get("saida") is not None:
                feitos.add(r["id"])
    sistemas = {nome: sistema(nome, d, exemplos) for nome in PROMPTS}
    intervalo, t_ult, tokens = 60.0 / args.rpm, 0.0, 0
    for n, i in enumerate(piloto[:args.n]):
        for nome in PROMPTS:
            rid = f"{nome}--{int(i)}"
            if rid in feitos:
                continue
            for tentativa in range(6):
                time.sleep(max(0.0, intervalo - (time.time() - t_ult)))
                t_ult = time.time()
                try:
                    r = c.chat.completions.create(
                        model=args.modelo, temperature=0, seed=SEED, max_tokens=400,
                        messages=[{"role": "system", "content": sistemas[nome]},
                                  {"role": "user", "content": f"<resposta>\n{cortar(d['treino'][i], 4000)}\n</resposta>"}])
                    break
                except RateLimitError as e:
                    msg = str(e).lower()
                    if "day" in msg or "daily" in msg or "quota" in msg:
                        print(f"cota diaria esgotada apos {n} respostas; rode de novo amanha para continuar", flush=True)
                        return
                    espera = float(e.response.headers.get("retry-after", 20 * (tentativa + 1)))
                    time.sleep(min(espera, 120))
                except APIStatusError as e:
                    if e.status_code >= 500:
                        time.sleep(10 * (tentativa + 1))
                        continue
                    raise
            else:
                print(f"{rid}: sem resposta depois de 6 tentativas, parando", flush=True)
                return
            txt = r.choices[0].message.content
            tokens += r.usage.total_tokens if r.usage else 0
            with open(saida, "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": rid, "saida": ler_json(txt), "bruto": txt, "fim": r.choices[0].finish_reason,
                                    "tokens": r.usage.total_tokens if r.usage else None}, ensure_ascii=False) + "\n")
        if (n + 1) % 10 == 0:
            print(f"{n+1}/{args.n} respostas | {tokens/1e3:.0f} mil tokens nesta execucao", flush=True)
    print("piloto completo em", saida)


def avaliar(args, d, piloto):
    y = d["y"]
    res = {}
    for linha in open(arquivo(args), encoding="utf-8"):
        r = json.loads(linha)
        nome, i = r["id"].split("--")
        if r.get("saida") is not None:
            res.setdefault(nome, {})[int(i)] = classe_de(r["saida"])
    comuns = [int(i) for i in piloto if all(res.get(nome, {}).get(int(i)) is not None for nome in PROMPTS)]
    if not comuns:
        sys.exit("nenhuma resposta avaliada por todos os prompts ainda")
    comuns = np.array(comuns)
    v2 = np.load(RES / "02_oof_pred_final.npy")
    print(f"{args.provedor} / {args.modelo}: {len(comuns)} respostas do treino avaliadas por todos os prompts")
    print(f"{'prompt':14s} acuracia | onde a v2 erra | dist. prevista c1/c234/c5")
    print(f"{'v2 (stacking)':14s} {(v2[comuns] == y[comuns]).mean()*100:6.2f}%")
    erra = v2[comuns] != y[comuns]
    for nome in PROMPTS:
        p = np.array([res[nome][i] for i in comuns])
        dist = np.bincount(p, minlength=3)
        print(f"{nome:14s} {(p == y[comuns]).mean()*100:6.2f}%  |   {(p[erra] == y[comuns][erra]).mean()*100:6.2f}%     | "
              f"{dist[0]}/{dist[1]}/{dist[2]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("acao", choices=["modelos", "rodar", "avaliar"])
    ap.add_argument("--provedor", choices=list(PROVEDORES), default="cerebras")
    ap.add_argument("--modelo", default="llama-3.3-70b")
    ap.add_argument("--n", type=int, default=300, help="quantas respostas do piloto avaliar")
    ap.add_argument("--rpm", type=float, default=20)
    args = ap.parse_args()
    if args.acao == "modelos":
        for m in cliente(args.provedor).models.list():
            print(m.id)
        return
    d = carregar()
    piloto, exemplos = amostras(d)
    if args.acao == "rodar":
        rodar(args, d, piloto, exemplos)
    else:
        avaliar(args, d, piloto)


if __name__ == "__main__":
    main()
