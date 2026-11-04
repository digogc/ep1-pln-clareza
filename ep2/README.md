# EP2 — ACH2118 Introdução ao Processamento de Língua Natural (EACH-USP)

Mesma tarefa do EP1 (clareza das respostas do e-SIC em `c1`, `c234`, `c5`), agora com **modelos de linguagem pré-treinados** no stacking do EP1: dois BERTimbau ajustados para a tarefa (128 e 256 subpalavras) e os embeddings do Qwen3-Embedding-0.6B no lugar dos do MiniLM.

## Resultado

| Validação cruzada, 5 dobras, semente 123 | Dobra 1 | Dobra 2 | Dobra 3 | Dobra 4 | Dobra 5 | Média |
|---|---|---|---|---|---|---|
| Baseline oficial (TF-IDF + regressão logística) | 45,91% | 45,28% | 45,52% | 44,77% | 46,22% | 45,54% |
| BERTimbau 128, sozinho | 43,89% | 45,16% | 45,52% | 43,73% | 46,09% | 44,88% |
| BERTimbau 256, sozinho | 45,36% | 46,73% | 46,76% | 45,74% | 46,57% | 46,23% |
| Stacking do EP1 (4 blocos) | 46,68% | 47,23% | 47,39% | 45,79% | 47,54% | 46,92% |
| Stacking do EP2 v2 (EP1 + BERT 128) | 47,52% | 47,35% | 48,03% | 46,12% | 47,44% | 47,29% |
| Stacking do EP2 v3 (v2 + BERT 256) | 47,95% | 47,72% | 48,26% | 46,49% | 48,28% | 47,74% |
| **Stacking do EP2 v4 (v3 com Qwen3 no lugar do MiniLM)** | 48,62% | 47,92% | 48,08% | 46,54% | 48,41% | **47,91%** |

**EP1 × EP2 (v4) no test1:** 48,56% → **51,56%**; McNemar χ² = 5,878, **p = 0,015** (o EP2 é significativamente melhor; ver no relatório a seção 8 sobre os candidatos avaliados no test1).

**Modelo final (v4):** stacking de seis blocos: `lr_pal`, `svc` e `extremos` (TF-IDF, do EP1), `qwen3` (regressão logística sobre os embeddings do Qwen3-Embedding-0.6B) e dois BERTimbau ajustados (`neuralmind/bert-base-portuguese-cased`), com 128 e 256 subpalavras. Cada BERT entra no stacking resumido no escore `logit(c5) − logit(c1)`.

## Onde está cada coisa

| Você procura | Está em |
|---|---|
| **A planilha do test2 rotulada (entrega)** | [`entrega/test2.xlsx`](entrega/test2.xlsx) |
| **O relatório** | [`RELATORIO_EP2.md`](RELATORIO_EP2.md) |
| Fine-tuning do BERTimbau 128 | [`codigo/01_bert_finetune.py`](codigo/01_bert_finetune.py) |
| Fine-tuning do BERTimbau 256 e demais experimentos | [`experimentos/codigo/`](experimentos/codigo/) (`e02_bert.py`, `e04_embeddings.py` etc.) |
| Versões anteriores do modelo (v2, v3) | [`versoes/`](versoes/) |
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

O `02_stacking_ep2.py` leva poucos minutos em CPU. Ele mostra a acurácia de cada modelo em cada dobra e grava `ep2/entrega/test2.xlsx`.

As previsões dos modelos neurais já vêm prontas em `ep2/resultados/` (`bert_*`, `bert256_*` e `bloco_qwen3.npz`), porque os fine-tunings levam de 6 a ~40 horas em CPU. Para refazê-los do zero:

```bash
python ep2/codigo/01_bert_finetune.py
```

```bash
python ep2/experimentos/codigo/e02_bert.py --nome bert256_5d --max-len 256 --dobras 5 --threads 8
```

```bash
python ep2/experimentos/codigo/e04_embeddings.py
```

O primeiro gera os `bert_*` (BERT 128, ~6 h). O segundo gera, em `ep2/experimentos/resultados/`, os `bert256_5d_{oof,test1,test2}.npy` (~39 h), que devem ser copiados para `ep2/resultados/` sem o `_5d` no nome. O terceiro gera o `bloco_emb_qwen3.npz`, que deve ser copiado como `ep2/resultados/bloco_qwen3.npz`. Os fine-tunings são retomáveis: se forem interrompidos, rodar de novo continua de onde parou.

## Regras do enunciado que o código respeita

- Treino só com o `train.xlsx` (o mesmo do EP1). O test1 e o test2 entram sem rótulos, apenas no vocabulário do TF-IDF e na inferência.
- Os rótulos do test1 são lidos **somente** para o teste de McNemar: em `04_mcnemar.py` e, nos experimentos, em `e17`, `e20` e `e21`. As escolhas de modelo usam a validação cruzada no treino; a seção 8 do relatório lista todos os candidatos avaliados no test1. O test2 nunca é usado para comparar modelos.
- A entrega contém apenas os rótulos do **test2**. As previsões do test1 ficam em `resultados/test1_ep2.xlsx`, só para o McNemar, e não vão no .zip.
- Semente 123 em tudo.
