"""Atributos manuais (linguísticos e estruturais) de uma resposta do e-SIC.

A ideia é capturar o que pode levar um cidadão a achar uma resposta clara ou não:
tamanho, legibilidade, vocabulário raro/jurídico, se houve negativa, redirecionamento,
anexo, link, citação de leis, formalidades, de qual órgão/ano é o protocolo etc.
"""
import re
from collections import Counter

import numpy as np
import pandas as pd

VOGAIS = re.compile(r"[aeiouáéíóúâêôãõàü]+", re.I)
PALAVRA = re.compile(r"[A-Za-zÀ-ÿ]+")
FRASE = re.compile(r"[.!?;:]+(?:\s|$)")
NUP = re.compile(r"(\d{5})\.\d{6}/(\d{4})-\d{2}")
ANO = re.compile(r"\b(20[0-2]\d)\b")
SIGLA = re.compile(r"\b[A-ZÀ-Ý]{3,10}\b")

# contagem de ocorrências de expressões (em minúsculas)
MARCADORES = {
    "anexo": r"anex[oa]s?\b|anexad[oa]s?|em anexo|arquivo",
    "link": r"https?://|www\.|\.gov\.br|\.br/",
    "email": r"[\w.\-]+@[\w\-]+\.[\w.]+",
    "telefone": r"\(\d{2}\)\s?\d{4,5}[\- ]?\d{4}|0800",
    "negativa": r"não é possível|não será possível|impossibilidade|indefer|negad[oa]|não pode ser|não podemos|inviáve|inviabil",
    "sigilo": r"sigil|restrit[oa]|reservad[oa]|informaç(?:ão|ões) pessoa|confidencia",
    "nao_possui": r"não possu|não disp[õo]e|não dispomos|inexist|não h[áa] |não exist|não cont[ée]m|não foi localizad|não consta",
    "nao_compete": r"não compete|não cabe|não é de competência|foge à competência|não se enquadra|não trata de|não é o canal|não encontra respaldo|fora do escopo",
    "trabalho_adicional": r"trabalhos? adiciona|desarrazoad|desproporciona|genéric",
    "redireciona": r"encaminh|direcionad|dirija|dirigir-se|procure|entre em contato|entrar em contato|deverá solicitar|deve ser solicitad|fala\.br|e-ouv|ouvidoria",
    "pede_esclarecimento": r"reformul|esclareça|não está clar|especifi(?:que|car)|não foi possível compreender|não ficou clar",
    "recurso": r"recurso|recorrer|1ª instância|primeira instância|instância",
    "prazo": r"prazo|prorrog|dias úteis|\d+ dias",
    "lei": r"\blei\b|decreto|portaria|resolução|instrução normativa|\bart\.|artigo|inciso|§|parágrafo|12\.527|7\.724",
    "lai": r"12\.527|7\.724|lei de acesso",
    "saudacao_generica": r"prezad[oa]\(a\)|senhor\(a\)|prezado \(a\)|prezado cidadão|prezada cidadã|prezado\(a\) cidadão|prezado usuário|prezado solicitante|prezado requerente|prezados",
    "despedida": r"atenciosamente|cordialmente|respeitosamente|att\.?,",
    "agradece": r"agradec|à disposição|a disposição|colocamo-nos|estamos à",
    "satisfacao": r"pesquisa de satisfação|avali(?:e|ar) (?:o|a|este|nosso)",
    "informamos": r"informamos|esclarecemos|comunicamos|ressaltamos|cumpre|cabe destacar|salientamos",
    "segue": r"\bsegue[m]?\b",
    "tabela_dados": r"planilha|tabela|quadro|relatório|dados abertos|transparência",
    "desculpa": r"desculp|lamenta|infelizmente",
    "juridiques": r"outrossim|destarte|supracitad|em epígrafe|vossa senhoria|v\. ?s[ªa]\.|ex vi|nos termos d|com fulcro|à luz d|em tela|haja vista|referid[oa]",
    "sic": r"\bsic\b|serviço de informaç(?:ão|ões) ao cidadão|e-sic",
}
MARCADORES = {k: re.compile(v, re.I) for k, v in MARCADORES.items()}

_zipf_cache = {}


def _zipf(p):
    """Frequência Zipf da palavra em português (0 = desconhecida, ~7 = muito comum)."""
    if p not in _zipf_cache:
        try:
            from wordfreq import zipf_frequency
            _zipf_cache[p] = zipf_frequency(p, "pt")
        except Exception:  # sem wordfreq: atributo neutro
            _zipf_cache[p] = 3.0
    return _zipf_cache[p]


def atributos_de(t):
    d = {}
    n = len(t)
    palavras = PALAVRA.findall(t)
    npal = max(len(palavras), 1)
    letras = sum(len(p) for p in palavras)
    frases = max(len(FRASE.findall(t)), 1)
    d["n_caracteres"] = n
    d["n_palavras"] = len(palavras)
    d["log_palavras"] = np.log1p(len(palavras))
    d["n_frases"] = frases
    d["palavras_por_frase"] = npal / frases
    d["letras_por_palavra"] = letras / npal
    silabas = sum(len(VOGAIS.findall(p)) or 1 for p in palavras)
    d["silabas_por_palavra"] = silabas / npal
    # Índice de Flesch adaptado ao português (maior = mais fácil de ler)
    d["flesch_pt"] = 248.835 - 1.015 * (npal / frases) - 84.6 * (silabas / npal)
    d["frac_palavras_longas"] = sum(len(p) > 9 for p in palavras) / npal
    minus = [p.lower() for p in palavras]
    d["riqueza_lexical"] = len(set(minus)) / npal
    z = np.array([_zipf(p) for p in minus]) if minus else np.array([3.0])
    d["zipf_medio"] = float(z.mean())
    d["zipf_p10"] = float(np.percentile(z, 10))
    d["frac_palavras_raras"] = float((z < 3.0).mean())
    d["frac_palavras_desconhecidas"] = float((z == 0).mean())
    maius = sum(c.isupper() for c in t)
    alfa = max(sum(c.isalpha() for c in t), 1)
    d["frac_maiusculas"] = maius / alfa
    d["frac_digitos"] = sum(c.isdigit() for c in t) / max(n, 1)
    d["frac_pontuacao"] = sum(c in ".,;:!?()[]\"'/-" for c in t) / max(n, 1)
    d["n_quebras"] = t.count("  ")  # quebras de linha viraram espaços duplos na planilha
    d["quebras_por_100_palavras"] = 100 * d["n_quebras"] / npal
    d["n_itens_numerados"] = len(re.findall(r"(?:^|\s)(?:\d{1,2}[.)]|[IVX]{1,4}\s?[-–)]|[a-z]\))\s", t))
    d["n_interrogacoes"] = t.count("?")
    d["n_aspas"] = t.count('"') + t.count("“")
    d["n_parenteses"] = t.count("(")
    d["n_valores_reais"] = len(re.findall(r"R\$", t))
    for nome, rx in MARCADORES.items():
        c = len(rx.findall(t))
        d[f"m_{nome}"] = c
        d[f"m_{nome}_por_100"] = 100 * c / npal
    # saudação com nome próprio ("Prezado João,")
    d["saudacao_com_nome"] = int(bool(re.match(r"\s*(?:prezad[oa]|car[oa]|sr\.?|sra\.?|senhor[a]?)\s+(?:sr\.?ª?\s+|sra\.?\s+|senhor[a]?\s+)?[A-ZÀ-Ý][a-zà-ÿ]+", t, re.I))
                              and not MARCADORES["saudacao_generica"].search(t[:60]))
    d["comeca_com_prezado"] = int(bool(re.match(r"\s*prezad", t, re.I)))
    # protocolo (NUP): os 5 primeiros dígitos identificam o órgão; depois vem o ano
    m = NUP.search(t)
    d["tem_protocolo"] = int(bool(m))
    d["orgao_protocolo"] = int(m.group(1)) if m else -1
    anos = [int(a) for a in ANO.findall(t)]
    d["ano_protocolo"] = int(m.group(2)) if m else -1
    d["ano_max"] = max(anos) if anos else -1
    d["ano_min"] = min(anos) if anos else -1
    d["n_anos_citados"] = len(anos)
    siglas = SIGLA.findall(t)
    d["n_siglas"] = len(siglas)
    d["sigla_principal"] = Counter(siglas).most_common(1)[0][0] if siglas else ""
    return d


def tabela_atributos(textos):
    df = pd.DataFrame([atributos_de(t) for t in textos])
    # frequência do texto idêntico no córpus (resposta-modelo?) - não usa rótulos
    cont = Counter(textos)
    df["repeticoes_do_texto"] = [cont[t] for t in textos]
    # inícios e finais iguais indicam o mesmo modelo de resposta / mesmo órgão
    ini = Counter(t[:60] for t in textos)
    fim = Counter(t[-60:] for t in textos)
    df["repeticoes_do_inicio"] = [ini[t[:60]] for t in textos]
    df["repeticoes_do_final"] = [fim[t[-60:]] for t in textos]
    # categóricas -> códigos inteiros (categorias raras agrupadas)
    for col, minimo in (("sigla_principal", 15), ("orgao_protocolo", 10)):
        vc = df[col].value_counts()
        mant = set(vc[vc >= minimo].index)
        df[col] = df[col].where(df[col].isin(mant), other=("__outros__" if col == "sigla_principal" else -2))
        df[col] = df[col].astype("category").cat.codes
    return df


CATEGORICAS = ["sigla_principal", "orgao_protocolo"]
