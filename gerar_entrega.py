"""Monta a entrega do EP1: confere a planilha, gera o relatorio em HTML e cria entrega_EP1.zip.

Uso:  python gerar_entrega.py
O .zip leva: test1.xlsx, o relatorio (relatorio.pdf se existir; senao relatorio.html) e apresentacao.pdf.
"""
import re
import zipfile
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent
ENT = RAIZ / "entrega"
CSS = """body{font-family:Georgia,serif;max-width:820px;margin:32px auto;padding:0 16px;line-height:1.45;color:#111}
h1{font-size:1.5em}h2{font-size:1.15em;margin-top:1.6em}table{border-collapse:collapse;margin:10px 0;font-size:.92em}
th,td{border:1px solid #999;padding:4px 8px;text-align:left;vertical-align:top}th{background:#eee}
code{font-family:Consolas,monospace;font-size:.92em;background:#f3f3f3;padding:0 3px}"""


def relatorio_html():
    import markdown
    md = (RAIZ / "RELATORIO.md").read_text(encoding="utf-8")
    corpo = markdown.markdown(md, extensions=["tables", "sane_lists"])
    html = f"<!doctype html><html lang='pt-br'><head><meta charset='utf-8'><title>Relatorio EP1</title><style>{CSS}</style></head><body>{corpo}</body></html>"
    (ENT / "relatorio.html").write_text(html, encoding="utf-8")
    return md


def conferir_planilha():
    o, e = pd.read_excel(RAIZ / "test1.xlsx"), pd.read_excel(ENT / "test1.xlsx")
    ok = (e.shape == (900, 2) and list(e.columns) == ["resp_text", "clarity"]
          and (o["resp_text"] == e["resp_text"]).all() and set(e["clarity"]) <= {"c1", "c234", "c5"}
          and e["clarity"].notna().all())
    print(f"[{'ok' if ok else 'ERRO'}] entrega/test1.xlsx: {e.shape[0]} linhas, colunas {list(e.columns)}, "
          f"rotulos {e['clarity'].value_counts().to_dict()}")
    return ok


def main():
    ENT.mkdir(exist_ok=True)
    ok = conferir_planilha()
    md = relatorio_html()
    print("[ok] entrega/relatorio.html gerado (abra no navegador e salve como PDF em entrega/relatorio.pdf)")
    pend = len(re.findall(r"\[PREENCHER", md))
    if pend:
        print(f"[FALTA] {pend} campo(s) [PREENCHER] no RELATORIO.md (nomes, numeros USP, link do repositorio)")
    rel = ENT / "relatorio.pdf" if (ENT / "relatorio.pdf").exists() else ENT / "relatorio.html"
    if rel.suffix == ".html":
        print("[FALTA] entrega/relatorio.pdf (o .zip vai com o relatorio em HTML enquanto o PDF nao existir)")
    apr = ENT / "apresentacao.pdf"
    if not apr.exists():
        print("[FALTA] entrega/apresentacao.pdf (apresentacao de ate 10 min, em PDF)")
    arquivos = [ENT / "test1.xlsx", rel] + ([apr] if apr.exists() else [])
    with zipfile.ZipFile(RAIZ / "entrega_EP1.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in arquivos:
            z.write(f, f.name)
    completo = ok and not pend and rel.suffix == ".pdf" and apr.exists()
    print(f"entrega_EP1.zip montado com: {[f.name for f in arquivos]}")
    print("ENTREGA COMPLETA" if completo else "ENTREGA INCOMPLETA - veja os itens [FALTA] acima")


if __name__ == "__main__":
    main()
