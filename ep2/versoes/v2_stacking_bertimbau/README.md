# EP2 — ACH2118 Introdução ao Processamento de Língua Natural (EACH-USP)

Mesma tarefa do EP1 (clareza das respostas do e-SIC em `c1`, `c234`, `c5`), agora com **fine-tuning do BERTimbau**, avaliado sozinho e como quinto modelo-base do stacking do EP1.

## Resultado

| Validação cruzada, 5 dobras, semente 123 | Dobra 1 | Dobra 2 | Dobra 3 | Dobra 4 | Dobra 5 | Média |
|---|---|---|---|---|---|---|
| Baseline oficial (TF-IDF + regressão logística) | 45,91% | 45,28% | 45,52% | 44,77% | 46,22% | 45,54% |
| BERTimbau ajustado, sozinho | 43,89% | 45,16% | 45,52% | 43,73% | 46,09% | 44,88% |
| Stacking do EP1 (4 blocos) | 46,68% | 47,23% | 47,39% | 45,79% | 47,54% | 46,92% |
| **Stacking do EP2 (4 blocos + BERTimbau)** | 47,52% | 47,35% | 48,03% | 46,12% | 47,44% | **47,29%** |

**EP1 × EP2 no test1:** 48,56% → **50,44%**; McNemar χ² = 3,938, **p = 0,047** (o EP2 é significativamente melhor).

**Modelo final:** o stacking do EP1 com o BERTimbau ajustado (`neuralmind/bert-base-portuguese-cased`) como quinto bloco. O BERT entra no stacking resumido no escore `logit(c5) − logit(c1)`.

## Onde está cada coisa

| Você procura | Está em |
|---|---|
| **A planilha do test2 rotulada (entrega)** | [`entrega/test2.xlsx`](entrega/test2.xlsx) |
| **O relatório** | [`RELATORIO_EP2.md`](RELATORIO_EP2.md) |
| Fine-tuning do BERTimbau | [`codigo/01_bert_finetune.py`](codigo/01_bert_finetune.py) |
| Stacking (gera a planilha do test2) | [`codigo/02_stacking_ep2.py`](codigo/02_stacking_ep2.py) |
| Análise do modelo (eli5 e erros) | [`codigo/03_analise.py`](codigo/03_analise.py) → [`resultados/analise/`](resultados/analise/) |
| Teste de McNemar EP1 × EP2 no test1 | [`codigo/04_mcnemar.py`](codigo/04_mcnemar.py) |
| Saídas e logs | [`resultados/`](resultados/) |

## Como reproduzir

1. Coloque os arquivos do professor em `ep2/dados/` com estes nomes:
   `train.xlsx`, `test2.xlsx` e `test1_rotulado.xlsx` (o test1 com rótulos, usado só no McNemar).
2. Instale as dependências (as mesmas do EP1):

   ```bash
   pip install -r modelo_final/requirements.txt
   pip install eli5==0.16.0
   ```

3. Rode, a partir da raiz do repositório:

   ```bash
   python ep2/codigo/02_stacking_ep2.py
   python ep2/codigo/03_analise.py
   python ep2/codigo/04_mcnemar.py
   ```

O `02_stacking_ep2.py` leva cerca de 7 minutos em CPU. Ele mostra a acurácia de cada modelo em cada dobra e grava `ep2/entrega/test2.xlsx`.

As previsões do BERTimbau já vêm prontas em `ep2/resultados/bert_*.npy`, porque o fine-tuning leva cerca de 6 horas em CPU. Para refazê-lo do zero, apague esses três arquivos e os `bert_dobra*.npz` e rode antes:

```bash
python ep2/codigo/01_bert_finetune.py
```

O script é retomável: se for interrompido, rodar de novo pula as dobras já concluídas.

## Regras do enunciado que o código respeita

- Treino só com o `train.xlsx` (o mesmo do EP1). O test1 e o test2 entram sem rótulos, apenas no vocabulário do TF-IDF e na inferência.
- Os rótulos do test1 são lidos **somente** em `04_mcnemar.py`. Nenhuma escolha de modelo usa o test1 ou o test2.
- A entrega contém apenas os rótulos do **test2**. As previsões do test1 ficam em `resultados/test1_ep2.xlsx`, só para o McNemar, e não vão no .zip.
- Semente 123 em tudo.
