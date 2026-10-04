> Histórico dos experimentos. O modelo final e como rodá-lo estão em `../modelo_final/` e no `../README.md`. Os caminhos citados abaixo (`codigo/`, `resultados/`, `predicoes/`) são relativos a esta pasta `experimentos/`.

# EP1 — Resultados (ACH2118, classificação de clareza no e-SIC)

## Modelo final (entrega) — stacking com MiniLM, semente 123

**46,98% em validação cruzada (baseline oficial: 45,54%) e 47,70% no holdout oficial (baseline: 46,75%).** Planilha em `entrega/test1.xlsx` (rótulos `c1` 294 · `c234` 264 · `c5` 342).

Blocos: regressão logística em TF-IDF de palavras + LinearSVC em TF-IDF de palavras e caracteres + regressão logística em embeddings do paraphrase-multilingual-MiniLM-L12-v2 + classificador `c1`×`c5` dos extremos; meta-classificador de regressão logística (`C=0,1`).

| | Dobra 1 | Dobra 2 | Dobra 3 | Dobra 4 | Dobra 5 | Média |
|---|---|---|---|---|---|---|
| Baseline oficial | 45,91% | 45,28% | 45,52% | 44,77% | 46,22% | 45,54% |
| **Modelo final (MiniLM)** | 46,70% | 47,35% | 47,56% | 45,89% | 47,39% | **46,98%** |
| v2 (e5-small no lugar do MiniLM) | 47,25% | 46,78% | 46,84% | 45,55% | 47,59% | 46,80% |

A diferença entre a versão com MiniLM e a v2 (com e5-small) é de 0,18 ponto e 3 dobras em 5: é do tamanho do ruído. A v2 está guardada em `predicoes/test1_v2_stacking_atual.xlsx`.

Busca final em torno desse modelo (`codigo/15_busca_final.py`, `resultados/15_busca_final.csv`): nenhuma de 11 variações (outros valores de `C`, e5 junto, embeddings concatenados, SVM de palavras, eixo ordinal, LightGBM de atributos, retirar blocos) superou a média de 46,98%.

Não concluídos por falta de tempo: stacking com o BERTimbau ajustado pelo Dimitri (inferência em CPU interrompida em 2.256 de 5.119 textos; parciais em `resultados/13_bert_logits.npz`, avaliação pronta em `codigo/14_stacking_com_bert.py`) e um terceiro modelo de embeddings (distiluse, ~30 min de extração).

---

## Versão 2 (anterior) — semente 123, comparação pareada com o baseline oficial

**Modelo final: 46,80% em validação cruzada (baseline oficial: 45,54%) e 47,50% no holdout oficial (baseline: 46,75%).** A planilha rotulada está em `entrega/test1.xlsx`.

| Protocolo (semente 123) | Baseline oficial | Modelo final | Diferença |
|---|---|---|---|
| 5 dobras estratificadas | 45,54% ± 0,50 | **46,80% ± 0,69** | +1,26 (vence nas 5 dobras: +1,3 · +1,5 · +1,3 · +0,8 · +1,4) |
| Holdout 80/20 (protocolo do professor) | 46,75% | **47,50%** | +0,75 |

Nas 20.092 previsões fora da dobra, o modelo acerta 1.578 respostas que o baseline erra, e o baseline acerta 1.325 que o modelo erra.

**Modelo final:** stacking (meta-classificador de regressão logística, `C=0,1`) de quatro modelos-base: regressão logística em TF-IDF de palavras, LinearSVC em TF-IDF de palavras + caracteres, regressão logística em embeddings de sentença (multilingual-e5-small) e um classificador `c1`×`c5` treinado só nos extremos.

Planilha de teste: 900 linhas, colunas `resp_text` e `clarity`, textos idênticos ao original; rótulos `c1` 291 · `c234` 275 · `c5` 334.

### Por que o baseline "mudou" de 44,6% para ~46%

O código oficial usa um holdout único de 20% sem agrupar respostas repetidas. Cerca de 10% das linhas de validação têm cópia idêntica no treino, e o baseline acerta mais nelas (48,6% contra 45,4%). Nas dobras antigas, textos idênticos nunca se separavam, e por isso o mesmo baseline marcava 44,55%. Além disso, o holdout único varia muito com a semente (43,4% a 47,0% em 20 sementes); a semente 123 dá 46,75% de acurácia e 46,48% de F1 macro, que é o número que o código oficial imprime. Como 100 das 900 respostas do teste também têm cópia no treino, o protocolo oficial é o que mais se parece com o teste real. Detalhes em `resultados/00_baseline_log.txt`.

### Seleção passo a passo dos blocos (`resultados/07_avaliacao_log.txt`)

| Passo | Bloco acrescentado | Validação cruzada | Holdout |
|---|---|---|---|
| 1 | `lr_pal` | 46,53% | 48,22% |
| 2 | `svc` | 46,61% | 47,57% |
| 3 | `emb_e5small` | 46,73% | 47,50% |
| 4 | `extremos` | **46,80%** | 47,50% |
| 5 | memória de respostas idênticas | 46,78% | 47,23% |
| 6 | regressão ordinal | 46,63% | 47,23% |
| 7 | LightGBM de atributos manuais | 46,58% | 46,80% |
| 8 | memória de vizinhos | 46,69% | 47,42% |

### O que esta versão mostrou

- **Todas as combinações ficam entre 46,5% e 46,8%** na validação cruzada. Os ganhos de cada bloco depois do primeiro são de 0,1 ponto, do tamanho do ruído; o teto com modelos leves parece estar aí.
- **Um ganho aparente de +1,4 ponto era vazamento.** Um meta-classificador LightGBM sobre memórias "leave-one-out" marcou 48,1% na validação cruzada (58,6% nas linhas repetidas), mas só 46,7% no holdout rigoroso. Com as memórias calculadas fora da dobra, o ganho some. Não foi usado.
- **Também não ajudaram:** NB-SVM, regressão logística com 1–3-gramas, MLP sobre embeddings.
- **Risco:** a margem sobre o baseline é de ~1 ponto. Em 900 respostas de teste o erro-padrão é de ~1,7 ponto, então o resultado no teste pode ficar acima ou abaixo do baseline por acaso.
- **Não testado:** fine-tuning de transformer (pesado demais para CPU).

Arquivos desta versão: `codigo/pipeline.py`, `codigo/05_embeddings_sentenca.py`, `codigo/07_avaliar.py`, `codigo/08_final.py`, `resultados/07_*`, `resultados/08_*`. A entrega anterior está guardada em `predicoes/test1_entrega_anterior.xlsx`.

---

# Versão 1 (histórico) — dobras agrupadas por texto

> Os números abaixo usam dobras em que textos idênticos nunca se separam; não são comparáveis aos da versão 2. A planilha citada nesta parte foi substituída pela da versão 2.

**Modelo final: 46,37% de acurácia em validação cruzada**, contra 44,56% do baseline oficial (TF-IDF + regressão logística) e 34,3% da classe majoritária.

A planilha de teste rotulada pelo modelo final está em **`entrega/test1.xlsx`**.

---

## 1. Quadro geral

| Abordagem | Melhor modelo | Acurácia (validação cruzada, 5 dobras) | Diferença para o baseline |
|---|---|---|---|
| Classe majoritária | sempre `c5` | 34,30% | −10,3 |
| **Baseline oficial** | TF-IDF padrão + regressão logística padrão | **44,56%** | — |
| 1 — Lexical | TF-IDF palavras (1–2-gram) + caracteres (2–5-gram), LinearSVC C=0,1 | 45,66% | +1,1 |
| 2 — Atributos manuais | LightGBM em 87 atributos | 42,79% | −1,8 |
| 3a — Embeddings + classificador | LSA (300d) + regressão logística | 44,65% | +0,1 |
| 3b — Rede neural | TextCNN com embeddings ajustados, 4 épocas | 44,94% | +0,4 |
| **Final — combinação** | **TF-IDF + atributos + memória de respostas repetidas (stacking)** | **46,37%** ± 0,73 | **+1,8** |

Todos os números usam a mesma divisão de dobras (`dados/folds.csv`), em que textos idênticos nunca ficam separados entre treino e validação.

Detalhes de cada abordagem: `resultados/01_tfidf.md`, `resultados/02_atributos.md`, `resultados/03_embeddings_redes.md`.

---

## 2. O modelo final

É um **stacking**: uma regressão logística (meta-classificador) aprende a combinar as saídas de dois modelos-base e de um atributo de memória.

| Componente | O que fornece | Entradas no meta-classificador |
|---|---|---|
| TF-IDF + LinearSVC (abordagem 1) | pontuação de cada classe a partir do vocabulário | 3 |
| LightGBM em 87 atributos manuais (abordagem 2) | probabilidade de cada classe a partir de tamanho, estrutura, marcadores e órgão | 3 |
| Memória de respostas repetidas | como as *outras* linhas do treino com texto idêntico foram rotuladas (proporção por classe + quantidade) | 4 |

Meta-classificador: regressão logística com `C=1` sobre as 10 entradas padronizadas.

### Como as saídas dos modelos-base são geradas sem vazamento

Cada modelo-base é treinado em 4 dobras e prevê a quinta. Repetindo isso para as 5 dobras, toda linha do treino ganha uma previsão feita por um modelo que **não viu aquela linha** (previsão "fora da dobra"). O meta-classificador é treinado sobre essas previsões. No teste, os modelos-base são retreinados com todo o treino.

### A memória de respostas repetidas

- 12,0% das linhas do treino (2.413) têm outra linha com texto idêntico. No teste, 100 das 900 respostas têm cópia no treino.
- As contagens de repetição são feitas depois de juntar espaços repetidos; comparando o texto bruto, os números são um pouco menores (por exemplo, 88 respostas de teste com cópia no treino, como consta no `RESUMO-EP1.md`).
- Para cada linha, o atributo conta os rótulos das **outras** cópias (a própria linha fica de fora).
- Essas respostas-modelo puxam para `c1`: entre as linhas repetidas, 43,3% são `c1`, contra 31,6% no córpus todo.
- Só copiar o rótulo mais comum das outras cópias acerta 49,9% nessas linhas; o TF-IDF sozinho acerta 45,8% nelas.

### Busca em grade da combinação

| Parâmetro | Valores |
|---|---|
| Modelos-base incluídos | todos os 15 subconjuntos de {TF-IDF, atributos, LSA, TextCNN} |
| Memória de respostas repetidas | {sem · com} |
| `C` do meta-classificador | {0,01 · 0,1 · 1} |

São 90 combinações (`resultados/04_ensemble_grid.csv`).

| Combinação | Sem memória | Com memória |
|---|---|---|
| Só TF-IDF | 45,64% | 46,11% |
| **TF-IDF + atributos** | 45,94% | **46,37%** |
| TF-IDF + TextCNN | 45,93% | 46,18% |
| TF-IDF + LSA | 45,91% | 46,00% |
| Os 4 modelos | 45,69% | 45,97% |

### Resultado do modelo final

| Medida | Valor |
|---|---|
| Acurácia em validação cruzada | **46,37%** ± 0,73 |
| Acurácia por dobra | 47,47% · 46,82% · 45,92% · 45,36% · 46,29% |
| Diferença para o baseline, por dobra | +2,0 · +1,5 · +2,2 · +1,5 · +1,9 (vence nas 5) |
| Estimativa aninhada (ver abaixo) | 46,00% ± 0,79 |
| Acurácia em linhas com texto repetido | 49,73% |
| Acurácia nas demais linhas | 45,91% |

Matriz de confusão (previsões fora da dobra, 20.092 respostas):

| | previsto c1 | previsto c234 | previsto c5 | acerto da classe |
|---|---|---|---|---|
| **real c1** | 3.087 | 1.573 | 1.687 | 48,6% |
| **real c234** | 2.019 | 2.135 | 2.699 | 31,2% |
| **real c5** | 1.221 | 1.576 | 4.095 | 59,4% |

### Estimativa aninhada: 46,00%

Os 46,37% são levemente otimistas, porque a melhor das 90 combinações foi escolhida nas mesmas dobras em que foi medida. Para corrigir isso, repeti a escolha **dentro** de cada dobra: a combinação é selecionada usando só as outras 4 dobras e avaliada na que ficou de fora. O resultado é **46,00%** (+1,4 sobre o baseline), e essa é a expectativa mais honesta para dados novos. Em 3 das 5 dobras a combinação escolhida foi a mesma do modelo final (TF-IDF + atributos + memória), e em todas as 5 a memória entrou.

### Planilha de teste

| | |
|---|---|
| Arquivo | `entrega/test1.xlsx` |
| Formato | planilha `test1`, 900 linhas, colunas `resp_text` e `clarity`, textos idênticos ao original |
| Rótulos | `c1` 273 · `c234` 254 · `c5` 373 |
| Concordância com o TF-IDF sozinho | 86,3% das linhas |

---

## 3. O que os resultados dizem

1. **O problema tem um teto baixo.** Palavras, atributos manuais, vetores densos e rede convolucional ficam todos entre 43% e 46%. O rótulo é a nota que um usuário deu, e respostas idênticas recebem notas diferentes (309 dos 495 textos repetidos têm rótulos conflitantes). Boa parte da nota não está no texto.
2. **A nota mede satisfação, não legibilidade.** Respostas que negam o pedido puxam para `c1`; respostas curtas ou com anexo puxam para `c5`. Medidas clássicas de clareza textual (Flesch, tamanho de frase, palavras raras) são o grupo de atributos mais fraco.
3. **O vocabulário completo ainda é a melhor representação.** O TF-IDF de palavras + caracteres supera embeddings pré-treinados genéricos, embeddings treinados no córpus e a rede convolucional.
4. **A classe do meio é a mais difícil.** `c234` tem o menor acerto em todos os modelos (31% no final) e é confundida com as duas vizinhas.
5. **Combinar ajuda pouco, mas de forma consistente.** Os atributos manuais sozinhos perdem do baseline, mas somam +0,3 ponto ao TF-IDF porque erram em lugares diferentes. Incluir LSA e TextCNN não acrescenta nada.
6. **A memória de respostas repetidas é o ganho mais barato.** Soma +0,4 ponto ao modelo final e melhora 43 das 45 combinações testadas (ganho médio de +0,3).
7. **Todos os modelos decoram o treino.** As acurácias de treino vão de 56% a 89% contra ~45% na validação; por isso a regularização forte e a escolha por validação cruzada.

---

## 4. Riscos e limites

- **A margem sobre o baseline é estreita.** +1,8 ponto (+1,4 na estimativa aninhada) equivale a cerca de 13 a 16 respostas em 900. O erro-padrão de uma acurácia medida em 900 linhas é de cerca de 1,7 ponto, então o resultado no teste pode variar para cima ou para baixo.
- **O baseline do professor pode ser diferente.** O enunciado não informa a configuração exata; usei TF-IDF e regressão logística com os parâmetros padrão do scikit-learn.
- **Critério "inovação".** O modelo final não é só TF-IDF + classificador linear: inclui LightGBM sobre atributos manuais, memória de duplicatas e stacking. Mas o componente principal continua sendo o TF-IDF com SVM, e quem decide se isso pontua é o professor.
- **Não testado: fine-tuning de BERT em português (BERTimbau).** É a família que mais provavelmente passaria dos ~46%, mas precisa de GPU e acesso ao Hugging Face, que não estavam disponíveis no ambiente em que rodei.
- **Reprodutibilidade.** As sementes estão fixas, mas pequenas variações são esperadas com outras versões das bibliotecas.

---

## 5. O que ainda falta para a entrega

- [ ] Preencher nomes e números USP em `RELATORIO.md`
- [ ] Subir a pasta `codigo/` (e `dados/folds.csv`) em um repositório e colocar o link no relatório
- [ ] Converter o relatório para o formato de entrega
- [ ] Preparar a apresentação em PDF (até 10 min)
- [ ] Montar o `.zip` com relatório, apresentação e `entrega/test1.xlsx`

---

## 6. Arquivos

| Caminho | Conteúdo |
|---|---|
| `entrega/test1.xlsx` | **planilha de teste rotulada pelo modelo final** |
| `RELATORIO.md` | rascunho do relatório no modelo do professor |
| `RESULTADOS.md` | este arquivo |
| `RESUMO-EP1.md` | resumo do enunciado |
| `requirements.txt` | bibliotecas e versões usadas |
| `codigo/common.py` | leitura dos dados, dobras, gravação da planilha |
| `codigo/01_tfidf.py` | abordagem 1 |
| `codigo/atributos.py`, `codigo/02_atributos_lgbm.py` | abordagem 2 |
| `codigo/03_embeddings.py`, `codigo/03_textcnn.py` | abordagem 3 |
| `codigo/04_ensemble.py` | modelo final (combinação) |
| `dados/folds.csv` | dobra de cada linha do treino |
| `resultados/0X_*.md` | explicação de cada abordagem |
| `resultados/0X_*_grid.csv` | todas as configurações testadas, com acurácia por dobra |
| `resultados/0X_*_log.txt` | saída de cada execução |
| `resultados/04_ensemble_aninhada.csv` | validação aninhada do modelo final |
| `resultados/04_ensemble_coeficientes.csv` | pesos do meta-classificador |
| `resultados/*.npy` | previsões fora da dobra e de teste de cada modelo |
| `predicoes/test1_*.xlsx` | teste rotulado por cada abordagem isolada (não são a entrega) |
