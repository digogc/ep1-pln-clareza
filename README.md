# EP1 — ACH2118 Introdução ao Processamento de Língua Natural (EACH-USP)

Classificação da clareza de respostas do e-SIC em três classes: `c1`, `c234` e `c5`.

## Resultado final

| Validação cruzada, 5 dobras, semente 123 | Dobra 1 | Dobra 2 | Dobra 3 | Dobra 4 | Dobra 5 | Média |
|---|---|---|---|---|---|---|
| Baseline oficial (TF-IDF + regressão logística) | 45,91% | 45,28% | 45,52% | 44,77% | 46,22% | 45,54% |
| **Modelo final (stacking)** | 46,70% | 47,35% | 47,56% | 45,89% | 47,39% | **46,98%** |

**Modelo final:** stacking de quatro modelos-base, combinados por regressão logística:

1. regressão logística sobre TF-IDF de palavras;
2. SVM linear sobre TF-IDF de palavras e caracteres;
3. regressão logística sobre embeddings de sentença (`paraphrase-multilingual-MiniLM-L12-v2`, congelado);
4. classificador `c1`×`c5` treinado só nos extremos.

## Onde está cada coisa

| Você procura | Está em |
|---|---|
| **O código do modelo final** | [`modelo_final/stacking_ep1.py`](modelo_final/stacking_ep1.py) (script único) |
| **Como rodar** | [`modelo_final/README.md`](modelo_final/README.md) |
| **A planilha de teste rotulada (entrega)** | [`entrega/test1.xlsx`](entrega/test1.xlsx) |
| **O relatório** | [`RELATORIO.md`](RELATORIO.md) |
| Testes que não entraram no modelo final | [`experimentos/`](experimentos/) (índice em [`experimentos/README.md`](experimentos/README.md)) |

Só as pastas `modelo_final/` e `entrega/` e o `RELATORIO.md` fazem parte do resultado. Tudo em `experimentos/` é histórico.

## Rodar o modelo final

Coloque `train.xlsx` e `test1.xlsx` na pasta `modelo_final/` (ou na raiz) e execute:

```bash
cd modelo_final
pip install -r requirements.txt
python stacking_ep1.py
```

O script mostra a acurácia do baseline e do stacking em cada dobra e grava o teste rotulado em `modelo_final/saida/test1_minilm.xlsx`. Leva cerca de 4 minutos em CPU (os embeddings já vêm em `modelo_final/cache/`).

## Estrutura

```
README.md            este arquivo
RELATORIO.md         relatório no modelo do professor
modelo_final/        código e saídas do modelo final
entrega/             o que vai no .zip: test1.xlsx, relatório e apresentação
gerar_entrega.py     confere a planilha e monta o .zip da entrega
experimentos/        testes anteriores e alternativas descartadas
```
