# EP1 — ACH2118 Introdução ao Processamento de Língua Natural

Classificação da clareza de respostas do e-SIC em `c1`, `c234` e `c5`.

**Modelo final:** stacking de quatro modelos-base (regressão logística em TF-IDF de palavras, SVM linear em TF-IDF de palavras e caracteres, regressão logística em embeddings do MiniLM e classificador `c1`×`c5` dos extremos), combinados por regressão logística.

| Validação cruzada, 5 dobras, semente 123 | Dobra 1 | Dobra 2 | Dobra 3 | Dobra 4 | Dobra 5 | Média |
|---|---|---|---|---|---|---|
| Baseline oficial | 45,91% | 45,28% | 45,52% | 44,77% | 46,22% | 45,54% |
| Modelo final | 46,70% | 47,35% | 47,56% | 45,89% | 47,39% | **46,98%** |

## Onde está cada coisa

| Caminho | Conteúdo |
|---|---|
| `entrega/` | **o que vai no .zip**: `test1.xlsx` (teste rotulado), `relatorio.html` (relatório, para salvar como PDF) e, quando pronta, `apresentacao.pdf` |
| `gerar_entrega.py` | regenera `entrega/relatorio.html` a partir do `RELATORIO.md`, confere a planilha e monta `entrega_EP1.zip` |
| `RELATORIO.md` | relatório no modelo do professor (editar aqui) |
| `reproducao/` | **código para reproduzir o resultado**: script único `stacking_ep1.py`, com as versões MiniLM (final) e e5 |
| `RESULTADOS.md` | histórico completo dos experimentos e comparações |
| `RESUMO-EP1.md` | resumo do enunciado |
| `codigo/` | etapas exploratórias (grades, modelos descartados, comparações); `pipeline.py` e `08_final.py` são a versão de desenvolvimento do modelo final |
| `resultados/` | tabelas e logs de cada experimento |
| `predicoes/` | planilhas de teste de versões anteriores e alternativas (não são a entrega) |
| `dados/` | caches de desenvolvimento (TF-IDF, atributos, embeddings); podem ser apagados, são recalculados |
| `resposta_dimitri/` | modelo de um colega (BERTimbau ajustado + árvores); não entrou na entrega e fica fora do repositório |

## Reproduzir

```bash
cd reproducao
pip install -r requirements.txt
python stacking_ep1.py
```

Detalhes em `reproducao/README.md`.

## Montar a entrega

1. Preencher nomes, números USP e link do repositório no `RELATORIO.md`.
2. Colocar a apresentação em `entrega/apresentacao.pdf`.
3. Rodar `python gerar_entrega.py`, abrir `entrega/relatorio.html` no navegador e salvar como PDF em `entrega/relatorio.pdf`, e rodar `python gerar_entrega.py` de novo para montar o `entrega_EP1.zip`.
