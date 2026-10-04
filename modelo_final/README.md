# EP1 ACH2118 — reprodução do stacking

Um único script (`stacking_ep1.py`) reproduz o resultado do grupo: mede o baseline oficial e o stacking em validação cruzada de 5 dobras (semente 123), mostra a acurácia por dobra e grava a planilha de teste rotulada.

## Como rodar

1. Coloque `train.xlsx` e `test1.xlsx` nesta pasta (ou na pasta de cima).
2. Instale as bibliotecas (Python 3.10 ou mais novo):

```bash
pip install -r requirements.txt
```

3. Rode a versão desejada:

```bash
python stacking_ep1.py
```

```bash
python stacking_ep1.py --versao e5
```

```bash
python stacking_ep1.py --versao ambas
```

Sem argumento, roda a versão MiniLM (a da entrega).

## As duas versões

| Versão | Modelo de embeddings (Hugging Face) | Validação cruzada | Baseline oficial |
|---|---|---|---|
| `minilm` (entrega) | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 46,98% | 45,54% |
| `e5` | `intfloat/multilingual-e5-small` | 46,80% | 45,54% |

A única diferença entre elas é o modelo que gera os embeddings; os outros três blocos são iguais.

## Tempo (CPU de 4 núcleos, sem GPU)

- TF-IDF e modelos-base: cerca de 4 minutos (as duas versões juntas, com os embeddings em cache).
- Embeddings: 20 a 30 minutos por modelo, **só na primeira vez**. Os vetores ficam em `cache/emb_<versao>.npy`; se a pasta `cache/` vier junto com o código, essa etapa é pulada.

## Saídas (pasta `saida/`)

- `test1_<versao>.xlsx`: teste rotulado, no mesmo formato do `test1.xlsx` (900 linhas, colunas `resp_text` e `clarity`).
- `resultado_<versao>.csv`: acurácia do baseline e do stacking em cada dobra.

## O modelo

Quatro modelos-base e um meta-classificador (regressão logística, `C=0,1`) que combina as saídas deles:

| Bloco | Representação | Classificador |
|---|---|---|
| `lr_pal` | TF-IDF de palavras (1–2-gramas) | Regressão logística, `C=1` |
| `svc` | TF-IDF de palavras (1–2) + caracteres (2–5) | SVM linear, `C=0,1` |
| `emb` | Embeddings de sentença (384 dimensões, modelo congelado) | Regressão logística, `C=0,01` |
| `extremos` | TF-IDF de palavras + caracteres | Regressão logística treinada só com `c1` e `c5` |

Cada modelo-base treina em 4 dobras e prevê a quinta; o meta-classificador aprende sobre essas previsões "fora da dobra". Para o teste, os modelos-base são retreinados com todo o treino.

Pequenas diferenças na segunda casa decimal podem aparecer com outras versões das bibliotecas.
