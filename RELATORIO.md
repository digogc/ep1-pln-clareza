# ACH2118 Introdução ao Processamento de Língua Natural — Relatório do EP1

## 1. Integrantes

| Nome | Número USP |
|---|---|
| **[PREENCHER]** | **[PREENCHER]** |
| **[PREENCHER]** | **[PREENCHER]** |

Desistências/ausências (opcional): **[PREENCHER ou remover]**

## 2. Estratégia

A tarefa é uma classificação ternária {`c1`, `c234`, `c5`} da clareza de respostas do e-SIC. Foram avaliadas várias famílias de modelos, sempre comparadas ao baseline oficial nas mesmas divisões de dados: modelos lexicais (TF-IDF de palavras e de caracteres com classificadores lineares), atributos manuais com gradient boosting (LightGBM), embeddings estáticos e rede convolucional (TextCNN), embeddings de sentença de transformers pequenos (paraphrase-multilingual-MiniLM-L12-v2 e multilingual-e5-small, usados congelados, em CPU), memórias de respostas idênticas e de respostas vizinhas, e a combinação desses modelos por stacking.

**O modelo final é um stacking de quatro modelos-base**, combinados por um meta-classificador de regressão logística:

1. `lr_pal`: regressão logística sobre TF-IDF de palavras (1–2-gramas);
2. `svc`: LinearSVC sobre TF-IDF de palavras (1–2-gramas) e de caracteres (2–5-gramas);
3. `emb_minilm`: regressão logística sobre embeddings de sentença do paraphrase-multilingual-MiniLM-L12-v2 (384 dimensões);
4. `extremos`: classificador `c1` × `c5` treinado apenas nas respostas dos dois extremos, cuja pontuação funciona como um eixo "ruim–bom" e ajuda a posicionar a classe intermediária.

Embeddings do multilingual-e5-small, LightGBM sobre atributos manuais, regressão ordinal, memória de respostas idênticas e memória de vizinhos foram avaliados, mas não entraram no modelo final por não melhorarem a combinação.

## 3. Pré-processamento

- **Todos os modelos:** junção de espaços e quebras de linha repetidos em um único espaço. Sem remoção de stopwords, stemming ou lematização.
- **TF-IDF:** minúsculas, `sublinear_tf=True`, `min_df=2` para n-gramas de palavras e `min_df=3` para n-gramas de caracteres (`char_wb`). O vocabulário e o IDF são calculados sobre treino + teste, sem rótulos.
- **Embeddings de sentença:** truncamento em 256 subpalavras (no e5, prefixo `query: `, exigido pelo modelo), vetores normalizados; padronização (média 0, desvio 1) antes da regressão logística.
- **Entradas do meta-classificador:** pontuações ou log-probabilidades dos modelos-base, padronizadas.

## 4. Parâmetros avaliados na busca em grade

**Modelos lexicais (32 configurações)**

| Parâmetro | Valores |
|---|---|
| Representação | {palavras 1-gram, palavras 1–2-gram, caracteres 2–5-gram, palavras 1–2-gram + caracteres 2–5-gram} |
| Classificador | {regressão logística, LinearSVC, Complement Naive Bayes} |
| `C` (regressão logística) | {0,3; 1; 3} |
| `C` (LinearSVC) | {0,03; 0,1; 0,3} |
| `alpha` (Complement NB) | {0,1; 0,5} |

**Combinação por stacking (seleção passo a passo)**

| Parâmetro | Valores |
|---|---|
| Blocos candidatos | {`lr_pal`, `svc`, `emb_minilm`, `emb_e5small`, SVM só de palavras, `extremos`, regressão ordinal, LightGBM sobre 87 atributos manuais, memória de respostas idênticas, memória de vizinhos (TF-IDF e embeddings)} |
| Seleção | a cada passo entra o bloco que mais aumenta a acurácia em validação cruzada |
| `C` do meta-classificador | {0,01; 0,1; 1} |

**Outros modelos avaliados (não usados no modelo final)**

| Modelo | Parâmetro | Valores |
|---|---|---|
| LightGBM sobre atributos manuais | `num_leaves` / `min_child_samples` / `colsample_bytree` / `n_estimators` | {7, 15, 31, 63} / {20, 60} / {0,3; 0,6} / {100..600} |
| Embeddings estáticos + classificador | Representação | {spaCy, Word2Vec, Doc2Vec, LSA} |
| TextCNN | Embeddings / épocas | {congelados, ajustados} / 1..5 |
| NB-SVM, regressão logística com 1–3-gramas, MLP sobre embeddings | — | uma configuração cada |
| Só TF-IDF + SVM | Representação / `C` | {palavras; palavras + caracteres} / {0,01; 0,03; 0,1; 0,3; 1} |
| Embeddings no stacking | Modelo / `C` | {MiniLM; e5-small; os dois; concatenados} / {0,003; 0,01; 0,03; 0,1} |
| `lr_pal` no stacking | `C` / pesos de classe | {0,5; 1; 2} / {sem; balanceado} |
| Meta-classificador não linear (LightGBM) | `num_leaves` | {4, 8, 16} |

## 5. Valores ótimos

| Componente | Parâmetro | Valor |
|---|---|---|
| `lr_pal` | Representação / `C` | palavras 1–2-gram / 1 |
| `svc` | Representação / `C` | palavras 1–2-gram + caracteres 2–5-gram / 0,1 |
| `emb_minilm` | Modelo / `C` | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 / 0,01 |
| `extremos` | Representação / `C` | palavras 1–2-gram + caracteres 2–5-gram / 1 |
| Combinação | Blocos | `lr_pal` + `svc` + `emb_minilm` + `extremos` |
| | `C` do meta-classificador | 0,1 |

## 6. Procedimento

- **Semente única:** 123 em todas as divisões e modelos.
- **Separação treino/validação:** validação cruzada estratificada de 5 dobras (`StratifiedKFold`, embaralhada) sobre as 20.092 respostas do `train.xlsx`. Além disso, o protocolo do baseline oficial foi repetido: holdout único de 20% com `train_test_split(test_size=0.2, random_state=123, stratify=Y)`.
- **Baseline oficial:** TF-IDF padrão + `LogisticRegression(class_weight='balanced', max_iter=100)`, medido nas mesmas divisões que o modelo.
- **Stacking sem vazamento:** cada modelo-base é treinado em 4 dobras e prevê a quinta; o meta-classificador é treinado sobre essas previsões "fora da dobra". No holdout, todo o procedimento (inclusive as dobras internas) usa apenas os 80% de treino.
- **Cuidado com vazamento nas memórias:** memórias calculadas com "leave-one-out" inflaram a validação cruzada de um meta-classificador não linear (48,1%), ganho que desapareceu no holdout (46,7%). As memórias passaram a ser calculadas fora da dobra e, assim, não trouxeram ganho; ficaram fora do modelo final.
- **Rotulação do teste:** os modelos-base são retreinados com todo o treino e aplicados ao `test1.xlsx`; o meta-classificador treinado sobre as previsões fora da dobra produz os rótulos.
- **O conjunto de teste foi usado sem rótulos** apenas no vocabulário/IDF do TF-IDF.

## 7. Resultado

| Tarefa | Modelo final | Acurácia |
|---|---|---|
| Clareza {c1, c234, c5} | Stacking: `lr_pal` + `svc` + `emb_minilm` + `extremos` | **46,98%** em validação cruzada de 5 dobras (desvio entre dobras: 0,62) |

Acurácia por dobra: 46,70% · 47,35% · 47,56% · 45,89% · 47,39%. Para referência, nas mesmas dobras o baseline oficial (TF-IDF + regressão logística) obtém 45,54% em média.

## 8. Repositório de código

https://github.com/digogc/ep1-pln-clareza

## 9. Instruções de reprodução

O resultado final é reproduzido por um único script, `modelo_final/stacking_ep1.py`, que não depende de nenhum outro arquivo do projeto.

1. Python 3.10 ou mais novo e as bibliotecas de `modelo_final/requirements.txt` (pandas, numpy, scipy, scikit-learn, openpyxl, torch, sentence-transformers): `pip install -r requirements.txt`
2. Colocar `train.xlsx` e `test1.xlsx` na pasta `modelo_final/` (ou na pasta acima dela).
3. Na pasta `modelo_final/`, executar:

   | Comando | O que faz |
   |---|---|
   | `python stacking_ep1.py` | modelo final (embeddings MiniLM): validação cruzada por dobra, baseline oficial e planilha de teste rotulada |
   | `python stacking_ep1.py --versao e5` | versão alternativa com embeddings do multilingual-e5-small |
   | `python stacking_ep1.py --versao ambas` | as duas versões |

4. Saídas em `modelo_final/saida/`: `test1_minilm.xlsx` (teste rotulado, igual ao entregue) e `resultado_minilm.csv` (acurácia por dobra).

Tempo em CPU de 4 núcleos, sem GPU: cerca de 4 minutos para TF-IDF e modelos-base, mais 20 a 30 minutos na primeira execução para extrair os embeddings (ficam em `modelo_final/cache/`; se a pasta `cache/` do repositório for mantida, essa etapa é pulada). A semente é 123 em todo o pipeline. A pasta `experimentos/` do repositório guarda as etapas exploratórias (grades, comparações e modelos descartados) e não é necessária para reproduzir o resultado.
