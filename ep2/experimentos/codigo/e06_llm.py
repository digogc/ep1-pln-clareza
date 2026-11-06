"""Experimento 6: LLM via API (Claude) com diferentes estrategias de engenharia de prompt, e o LLM como bloco do stacking.

Garantias contra vazamento de dados:
  1. Piloto so com o TREINO: as respostas avaliadas sao uma amostra estratificada da dobra 1 (das mesmas 5 dobras
     da v2). Nenhum texto do test1/test2 e enviado nesta fase e os rotulos do test1 nunca sao lidos.
  2. Os exemplos rotulados dos prompts few-shot vem SO das dobras 2-5; o script confere que nenhum exemplo e a mesma
     linha nem tem o mesmo texto de uma resposta avaliada (assert).
  3. A escolha do prompt e feita so no piloto (treino); test1/test2 nunca entram nessa decisao.
  4. O bloco do stacking (acao "completo") usa so prompts SEM exemplos rotulados (zero-shot): o LLM nao ve rotulo
     nenhum, entao a previsao de cada linha do treino e "fora da dobra" por construcao. Os textos do test1/test2 sao
     enviados como a qualquer outro modelo-base, sem rotulo.

Uso:  python e06_llm.py preparar [--modelo M]  monta o piloto, salva exemplos de prompt e estima o custo (sem API)
      python e06_llm.py enviar   [--modelo M]  envia o piloto pela Batch API (50% mais barata)
      python e06_llm.py coletar  [--modelo M]  baixa os resultados quando o lote terminar
      python e06_llm.py avaliar  [--modelo M]  acuracia de cada prompt no piloto (e da v2 nas mesmas respostas)
      python e06_llm.py completo --prompt P [--modelo M] [--enviar]   estima (ou envia) um prompt em treino+test1+test2
      python e06_llm.py coletar  --prompt P [--modelo M]              baixa o lote completo e grava o bloco do stacking
Modelo padrao: claude-haiku-5-5 (US$ 0,10 / 0,50 por milhao de tokens, metade na Batch API).
Credencial: variavel EP2_ANTHROPIC_API_KEY (nome proprio de proposito: uma ANTHROPIC_API_KEY global faria o Claude Code
usar a chave da API em vez da assinatura). A chave nunca e impressa.
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

from comum import EXP, RES, ROTULOS, SEED, carregar, dobras, salvar_bloco

# US$ por milhao de tokens: entrada, saida, leitura de cache, escrita de cache (5 min)
PRECOS = {"claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
          "claude-sonnet-5-5": (2.0, 10.0, 0.20, 2.5),
          "claude-opus-5-5": (4.0, 20.0, 0.20, 5.0)}
DESCONTO_LOTE = 0.5
N_PILOTO = 400
DIR = EXP / "resultados" / "llm"
DIR.mkdir(parents=True, exist_ok=True)

BASE = (
    "Você vai analisar respostas dadas por órgãos públicos federais brasileiros a pedidos de acesso à informação "
    "feitos por cidadãos no sistema e-SIC (Lei de Acesso à Informação). Depois de receber a resposta, o próprio "
    "cidadão avaliou a clareza dela numa escala de 1 a 5. As notas foram agrupadas em três classes:\n"
    "- c1: nota 1 (o cidadão achou a resposta nada clara);\n"
    "- c234: notas 2, 3 ou 4 (clareza intermediária);\n"
    "- c5: nota 5 (o cidadão achou a resposta muito clara).\n"
    "As três classes têm frequências parecidas. Você verá apenas o texto da resposta, não o pedido original.\n"
)
RUBRICA = (
    "\nPadrões observados em avaliações reais (são tendências, não regras):\n"
    "- Tendem a c5: respostas diretas e objetivas ao que foi perguntado, inclusive negativas objetivas "
    "(\"não há\", \"nenhum\"); tom pessoal e cordial; a informação está no próprio texto ou num anexo bem identificado.\n"
    "- Tendem a c1: respostas que falam sobre o pedido em vez de respondê-lo; texto burocrático ou padronizado; "
    "recusas, sigilo ou alegação de que o pedido não foi compreendido; remissão a pedidos anteriores; "
    "encaminhamento vago a outro órgão ou a um link genérico.\n"
    "- Tendem a c234: respostas parciais, entregas com ressalvas, encaminhamentos precisos, respostas corretas "
    "mas pouco objetivas.\n"
    "A avaliação é subjetiva e depende do que o cidadão esperava; pense em como um cidadão comum reagiria.\n"
)
CIDADAO = (
    "Imagine que você é o cidadão que fez o pedido de informação e acabou de receber a resposta abaixo. "
    "Leia como esse cidadão leria e diga que nota de 1 a 5 você daria para a clareza da resposta.\n"
)
TAREFA_CLASSE = "\nIndique a classe mais provável (c1, c234 ou c5)."
TAREFA_NOTA = "\nIndique a nota de 1 a 5 que o cidadão provavelmente deu e a classe correspondente."
TAREFA_PROBS = ("\nEstime a probabilidade (em %, somando 100) de cada classe. Use a incerteza real: "
                "a tarefa é difícil e mesmo especialistas acertam pouco mais de metade dos casos.")


def esquema(campos):
    props = {"classe": {"type": "string", "enum": ROTULOS}, "nota": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
             "p_c1": {"type": "integer"}, "p_c234": {"type": "integer"}, "p_c5": {"type": "integer"}}
    return {"type": "json_schema", "schema": {"type": "object", "properties": {c: props[c] for c in campos},
                                              "required": list(campos), "additionalProperties": False}}


# nome: (texto do sistema sem exemplos, campos de saida, n de exemplos por classe, esforco)
PROMPTS = {
    "zero_simples":   (BASE + TAREFA_CLASSE, ("classe",), 0, "low"),
    "zero_rubrica":   (BASE + RUBRICA + TAREFA_NOTA, ("nota", "classe"), 0, "low"),
    "zero_rubrica_m": (BASE + RUBRICA + TAREFA_NOTA, ("nota", "classe"), 0, "medium"),   # mais raciocinio
    "zero_cidadao":   (BASE + CIDADAO + TAREFA_NOTA, ("nota", "classe"), 0, "low"),
    "zero_probs":     (BASE + RUBRICA + TAREFA_PROBS, ("p_c1", "p_c234", "p_c5"), 0, "low"),
    "few_12":         (BASE + TAREFA_CLASSE, ("classe",), 4, "low"),
    "few_60":         (BASE + TAREFA_CLASSE, ("classe",), 20, "low"),
}


def cortar(t, n):
    return t if len(t) <= n else t[: int(n * 0.7)] + " [...] " + t[-int(n * 0.3):]


def amostras(d):
    """Piloto (dobra 1) e conjunto de exemplos (dobras 2-5), sem sobreposicao de linhas nem de textos."""
    y, X = d["y"], d["treino"]
    dobra = dobras(y)
    cand = np.where(dobra == 0)[0]
    piloto, _ = train_test_split(cand, train_size=N_PILOTO, stratify=y[cand], random_state=SEED)
    textos_piloto = set(X[piloto])
    pool = np.array([i for i in np.where(dobra != 0)[0] if X[i] not in textos_piloto])
    rng = np.random.RandomState(SEED)
    exemplos = {k: [] for k in range(3)}
    for c in range(3):
        exemplos[c] = list(rng.permutation(pool[(y[pool] == c) & (np.char.str_len(X[pool].astype(str)) < 1500)])[:20])
    todos = [i for c in range(3) for i in exemplos[c]]
    assert not set(todos) & set(piloto) and not {X[i] for i in todos} & textos_piloto
    assert set(dobra[todos]) <= {1, 2, 3, 4} and set(dobra[piloto]) == {0}
    return piloto, exemplos


def sistema(nome, d, exemplos):
    texto, _, k, _ = PROMPTS[nome]
    if not k:
        return [{"type": "text", "text": texto}]
    X, y = d["treino"], d["y"]
    rng = np.random.RandomState(SEED)
    sel = [i for c in range(3) for i in exemplos[c][:k]]
    sel = list(rng.permutation(sel))                             # classes misturadas, ordem fixa
    bloco = "\n\nExemplos de respostas reais já avaliadas pelos cidadãos:\n" + "".join(
        f"\n<exemplo>\n<resposta>{cortar(X[i], 1500)}</resposta>\n<classe>{ROTULOS[y[i]]}</classe>\n</exemplo>\n" for i in sel)
    return [{"type": "text", "text": texto}, {"type": "text", "text": bloco, "cache_control": {"type": "ephemeral"}}]


def pensamento(modelo, esforco):
    """Sem raciocinio estendido nos prompts de esforco baixo (classificacao direta); adaptativo no de esforco medio."""
    if esforco != "low" or "opus" in modelo:                      # no Opus 5.5 o pensamento nao pode ser desligado
        return {}
    return {"thinking": {"type": "disabled" if "haiku" in modelo else "between_tools"}}


def pedido(nome, d, exemplos, modelo, texto, cid):
    _, campos, _, esforco = PROMPTS[nome]
    return {"custom_id": cid,
            "params": {"model": modelo, "max_tokens": 4096, "system": sistema(nome, d, exemplos),
                       "output_config": {"effort": esforco, "format": esquema(campos)}, **pensamento(modelo, esforco),
                       "messages": [{"role": "user", "content": f"<resposta>\n{cortar(texto, 6000)}\n</resposta>"}]}}


def estimar(pedidos, modelo):
    """Custo aproximado (3 caracteres por token, conservador para o tokenizador novo)."""
    e, o, cl, _ = PRECOS[modelo]
    tok = lambda s: len(s) / 3.0                                  # noqa: E731
    tot = {"entrada": 0.0, "cache": 0.0, "saida": 0.0}
    for p in pedidos:
        sis = p["params"]["system"]
        tot["entrada"] += tok(sis[0]["text"]) + tok(p["params"]["messages"][0]["content"])
        if len(sis) > 1:
            tot["cache"] += tok(sis[1]["text"])
        tot["saida"] += 400 if p["params"]["output_config"]["effort"] == "medium" else 40
    melhor = (tot["entrada"] * e + tot["cache"] * cl + tot["saida"] * o) / 1e6
    pior = ((tot["entrada"] + tot["cache"]) * e + tot["saida"] * o) / 1e6
    return melhor * DESCONTO_LOTE, pior * DESCONTO_LOTE, tot


VAR_CHAVE = "EP2_ANTHROPIC_API_KEY"   # nome proprio: ANTHROPIC_API_KEY global faria o Claude Code usar a chave


def cliente():
    import anthropic
    chave = os.environ.get(VAR_CHAVE)
    if not chave and sys.platform == "win32":
        import winreg                                             # variavel de usuario definida apos abrir o app
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                chave = winreg.QueryValueEx(k, VAR_CHAVE)[0]
        except OSError:
            pass
    if not chave:
        sys.exit(f"defina a variavel {VAR_CHAVE} com a chave da API (a chave nunca e impressa)")
    return anthropic.Anthropic(api_key=chave)


def textos_completos(d):
    """Todas as respostas, com ids curtos: tr<i> (treino), ta<i> (test1), tb<i> (test2)."""
    return ([(f"tr{i}", t) for i, t in enumerate(d["treino"])] + [(f"ta{i}", t) for i, t in enumerate(d["test1"])]
            + [(f"tb{i}", t) for i, t in enumerate(d["test2"])])


def arq(args, tipo):
    alvo = f"completo_{args.prompt}" if args.prompt else "piloto"
    return DIR / f"{tipo}_{alvo}_{args.modelo}.{'txt' if tipo == 'lote' else 'jsonl'}"


def baixar(c, lote_id, saida):
    with open(saida, "w", encoding="utf-8") as f:
        for r in c.messages.batches.results(lote_id):
            linha = {"id": r.custom_id, "tipo": r.result.type}
            if r.result.type == "succeeded":
                m = r.result.message
                linha["stop"] = m.stop_reason
                txt = next((b.text for b in m.content if b.type == "text"), "")
                try:
                    linha["saida"] = json.loads(txt)
                except json.JSONDecodeError:
                    linha["saida"] = None
                u = m.usage
                linha["uso"] = {"entrada": u.input_tokens, "saida": u.output_tokens,
                                "cache_leitura": u.cache_read_input_tokens or 0, "cache_escrita": u.cache_creation_input_tokens or 0}
            f.write(json.dumps(linha, ensure_ascii=False) + "\n")
    print("resultados em", saida)


def custo_real(linhas, modelo):
    e, o, cl, ce = PRECOS[modelo]
    return sum((u["entrada"] * e + u["saida"] * o + u["cache_leitura"] * cl + u["cache_escrita"] * ce) / 1e6 * DESCONTO_LOTE
               for u in (r["uso"] for r in linhas if "uso" in r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("acao", choices=["preparar", "enviar", "coletar", "avaliar", "completo"])
    ap.add_argument("--modelo", choices=list(PRECOS), default="claude-haiku-5-5")
    ap.add_argument("--prompt", choices=[p for p in PROMPTS if p.startswith("zero")], default=None,
                    help="prompt do lote completo (so zero-shot, sem exemplos rotulados)")
    ap.add_argument("--enviar", action="store_true", help="com 'completo': envia o lote (sem isso, so estima o custo)")
    args = ap.parse_args()
    d = carregar()
    piloto, exemplos = amostras(d)

    if args.acao == "completo":
        if not args.prompt:
            sys.exit("indique --prompt (zero-shot)")
        pedidos = [pedido(args.prompt, d, exemplos, args.modelo, t, f"{args.prompt}--{cid}") for cid, t in textos_completos(d)]
        melhor, pior, tot = estimar(pedidos, args.modelo)
        print(f"{args.prompt} em treino+test1+test2: {len(pedidos)} pedidos | entrada ~{tot['entrada']/1e6:.1f} M tokens"
              f" | custo estimado com Batch API ({args.modelo}): US$ {pior:.2f}")
        if args.enviar:
            lote = cliente().messages.batches.create(requests=pedidos)
            arq(args, "lote").write_text(lote.id)
            print("lote enviado:", lote.id, lote.processing_status)
        return

    if args.acao == "coletar":
        c = cliente()
        lote_id = arq(args, "lote").read_text().strip()
        lote = c.messages.batches.retrieve(lote_id)
        print("status:", lote.processing_status, lote.request_counts)
        if lote.processing_status == "ended":
            baixar(c, lote_id, arq(args, "resultados"))
            if args.prompt:
                gravar_bloco(args, d)
        return

    pedidos = [pedido(nome, d, exemplos, args.modelo, d["treino"][i], f"{nome}--{int(i)}") for nome in PROMPTS for i in piloto]
    if args.acao == "preparar":
        melhor, pior, tot = estimar(pedidos, args.modelo)
        print(f"piloto: {len(piloto)} respostas do treino (dobra 1) x {len(PROMPTS)} prompts = {len(pedidos)} pedidos")
        print(f"tokens estimados: entrada {tot['entrada']/1e6:.2f} M | exemplos (cacheaveis) {tot['cache']/1e6:.2f} M | saida {tot['saida']/1e6:.2f} M")
        print(f"custo estimado com Batch API ({args.modelo}): US$ {melhor:.2f} (com cache) a US$ {pior:.2f} (sem cache)")
        for nome in PROMPTS:
            ex = next(p for p in pedidos if p["custom_id"].startswith(nome + "--"))
            with open(DIR / f"exemplo_prompt_{nome}.txt", "w", encoding="utf-8") as f:
                f.write("=== SYSTEM ===\n" + "".join(b["text"] for b in ex["params"]["system"]) +
                        "\n\n=== USER ===\n" + ex["params"]["messages"][0]["content"] +
                        "\n\n=== SAIDA ===\n" + json.dumps(ex["params"]["output_config"], ensure_ascii=False))
        print(f"exemplos de cada prompt em {DIR}")
    elif args.acao == "enviar":
        lote = cliente().messages.batches.create(requests=pedidos)
        arq(args, "lote").write_text(lote.id)
        print("lote enviado:", lote.id, lote.processing_status)
    elif args.acao == "avaliar":
        avaliar(args, d, piloto)


def atributos(s):
    """Saida do LLM -> atributos do bloco: nota (1-5) e classe em 1 de 3, ou log das probabilidades."""
    if s is None:
        return None
    if "p_c1" in s:
        p = np.array([s["p_c1"], s["p_c234"], s["p_c5"]], float) + 1.0
        return list(np.log(p / p.sum()))
    um = [0.0, 0.0, 0.0]
    um[ROTULOS.index(s["classe"])] = 1.0
    return ([float(s["nota"])] if "nota" in s else []) + um


def gravar_bloco(args, d):
    linhas = [json.loads(x) for x in open(arq(args, "resultados"), encoding="utf-8")]
    saidas = {r["id"].split("--")[1]: atributos(r.get("saida")) for r in linhas}
    ids = [cid for cid, _ in textos_completos(d)]
    validos = [v for v in saidas.values() if v is not None]
    media = np.mean(validos, axis=0)
    X = np.array([saidas.get(cid) if saidas.get(cid) is not None else media for cid in ids])
    faltam = sum(saidas.get(cid) is None for cid in ids)
    n = len(d["treino"])
    nome = f"llm_{args.prompt}_{args.modelo.split('-')[1]}"
    salvar_bloco(nome, X[:n], X[n:])
    print(f"bloco {nome}: {X.shape[1]} atributos | {faltam} respostas sem saida valida (preenchidas com a media)"
          f" | custo real US$ {custo_real(linhas, args.modelo):.2f}")
    print(f"avalie com: python e05_avaliar.py {nome}")


def classe_de(nome, s):
    if s is None:
        return None
    if "p_c1" in s:
        return int(np.argmax([s["p_c1"], s["p_c234"], s["p_c5"]]))
    return ROTULOS.index(s["classe"])


def avaliar(args, d, piloto):
    y = d["y"]
    res = {}
    linhas = [json.loads(x) for x in open(arq(args, "resultados"), encoding="utf-8")]
    for r in linhas:
        nome, i = r["id"].split("--")
        res.setdefault(nome, {})[int(i)] = r.get("saida")
    v2 = np.load(RES / "02_oof_pred_final.npy")[piloto]
    print(f"{args.modelo} | piloto: {len(piloto)} respostas do treino | custo real do lote: US$ {custo_real(linhas, args.modelo):.2f}")
    print(f"{'prompt':16s} acuracia  validas  | onde a v2 erra | dist. prevista c1/c234/c5")
    print(f"{'v2 (stacking)':16s} {(v2 == y[piloto]).mean()*100:6.2f}%")
    for nome in PROMPTS:
        p = np.array([classe_de(nome, res.get(nome, {}).get(int(i))) if res.get(nome, {}).get(int(i)) is not None else -1 for i in piloto])
        ok = p >= 0
        acc = (p[ok] == y[piloto][ok]).mean() * 100 if ok.any() else float("nan")
        erra = ok & (v2 != y[piloto])
        acc_erra = (p[erra] == y[piloto][erra]).mean() * 100 if erra.any() else float("nan")
        dist = np.bincount(p[ok], minlength=3)
        print(f"{nome:16s} {acc:6.2f}%  {ok.sum():4d}     |   {acc_erra:6.2f}%     | {dist[0]}/{dist[1]}/{dist[2]}")


if __name__ == "__main__":
    main()
