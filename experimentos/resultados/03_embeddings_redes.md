# Abordagem 3 — Embeddings estáticos e rede neural (CPU)

Aqui o texto deixa de ser uma contagem de palavras e vira **vetores densos**. Testei dois caminhos:

- **3a — um vetor por resposta + classificador** (`codigo/03_embeddings.py`)
- **3b — rede convolucional TextCNN sobre a sequência de palavras** (`codigo/03_textcnn.py`)

**Resultado em uma linha:** máximo de **44,94%** (TextCNN com embeddings ajustados), contra 44,56% do baseline oficial — um empate técnico. Nenhuma variação desta família passa da abordagem 1 (45,66%).

## Como foi avaliado

Mesma validação cruzada das abordagens anteriores: 5 dobras fixas (`dados/folds.csv`), textos idênticos na mesma dobra, métrica acurácia. As representações não supervisionadas (Word2Vec, Doc2Vec, LSA, IDF) são ajustadas sobre treino + teste **sem rótulos**, o que o enunciado permite.

---

## 3a — Um vetor por resposta + classificador

### Representações

| Nome | Dimensões | O que é |
|---|---|---|
| `spacy_media` | 300 | média dos vetores pré-treinados do spaCy (`pt_core_news_lg`) |
| `spacy_idf` | 300 | idem, com cada palavra ponderada pelo IDF |
| `w2v_idf` | 200 | Word2Vec skip-gram treinado no próprio córpus, média ponderada por IDF |
| `doc2vec` | 200 | Doc2Vec (PV-DBOW) treinado no próprio córpus |
| `lsa` | 300 | TF-IDF de palavras (1–2-gram) reduzido por SVD (semântica latente) |
| `tudo` | 1.000 | concatenação de `spacy_idf` + `w2v_idf` + `doc2vec` + `lsa` |

Dos 51.345 termos distintos do córpus, 31.632 (62%) têm vetor pré-treinado no spaCy.

### Busca em grade

| Classificador | Parâmetro | Valores |
|---|---|---|
| Regressão logística (atributos padronizados) | `C` | {0,001 · 0,01 · 0,1 · 1} |
| Rede MLP (só em `spacy_idf`, `w2v_idf` e `tudo`) | camadas ocultas / `alpha` | {(256), 0,01} · {(256), 1} · {(512, 256), 1} |

São 33 configurações; a tabela completa está em `03_embeddings_grid.csv`.

### Resultados — melhor de cada representação

| Representação | Melhor classificador | Acurácia validação | Acurácia treino |
|---|---|---|---|
| `spacy_idf` | Regressão logística, C=0,01 | 42,11% | 45,9% |
| `spacy_media` | Regressão logística, C=0,1 | 42,17% | 46,4% |
| `doc2vec` | Regressão logística, C=0,01 | 43,41% | 46,1% |
| `w2v_idf` | MLP (512, 256), alpha=1 | 43,63% | 48,2% |
| `tudo` | Regressão logística, C=0,001 | 44,56% | 52,0% |
| **`lsa`** | **Regressão logística, C=0,01** | **44,65%** ± 0,43 | 48,2% |

O melhor (LSA + regressão logística) empata com o baseline oficial: +0,09 ponto na média, vencendo em 2 das 5 dobras.

---

## 3b — TextCNN

### Arquitetura

1. Cada resposta é cortada em **200 palavras** (25,6% das respostas são mais longas e ficam truncadas).
2. Cada palavra entra como seu vetor do spaCy, **reduzido de 300 para 100 dimensões por PCA** (mantém 77% da variância e deixa o treino em CPU cerca de 3 vezes mais rápido). Vocabulário de 28.123 palavras, 22.279 com vetor pré-treinado.
3. Convoluções de 2, 3, 4 e 5 palavras, **100 filtros cada**, com max-pooling: cada filtro aprende a detectar uma expressão.
4. Dropout de 0,5 e camada linear de saída com 3 classes.

Treino com AdamW (taxa 1e-3), lotes de 64, suavização de rótulos de 0,05.

### Busca em grade

| Parâmetro | Valores |
|---|---|
| Embeddings | {congelados · ajustados durante o treino} |
| Épocas | {1 · 2 · 3 · 4 · 5} |

### Resultados

| Embeddings | Épocas | Acurácia validação | Acurácia treino |
|---|---|---|---|
| Congelados | 1 | 39,61% | 40,2% |
| Congelados | 2 | 41,44% | 42,5% |
| Congelados | 3 | 41,61% | 43,2% |
| Congelados | 4 | 42,82% | 44,5% |
| Congelados | 5 | 42,43% | 44,8% |
| Ajustados | 1 | 41,33% | 42,5% |
| Ajustados | 2 | 44,09% | 48,4% |
| Ajustados | 3 | 44,59% | 54,0% |
| **Ajustados** | **4** | **44,94%** ± 0,53 | 61,1% |
| Ajustados | 5 | 43,86% | 68,4% |

A previsão do teste é a média dos 5 modelos (um por dobra) na época escolhida.

### Matriz de confusão do melhor TextCNN (previsões fora da dobra)

| | previsto c1 | previsto c234 | previsto c5 | acerto da classe |
|---|---|---|---|---|
| **real c1** | 3.572 | 927 | 1.848 | 56,3% |
| **real c234** | 2.711 | 1.349 | 2.793 | 19,7% |
| **real c5** | 1.796 | 987 | 4.109 | 59,6% |

---

## Máximo da abordagem: 44,94%

**TextCNN com embeddings do spaCy ajustados durante o treino, 4 épocas.**
Fica +0,38 ponto acima do baseline oficial na média, mas vence em apenas 3 das 5 dobras (de −0,6 a +1,8 ponto). Na prática é um **empate com o baseline**.

## Quadro geral até aqui

| Abordagem | Melhor modelo | Acurácia validação |
|---|---|---|
| Classe majoritária | — | 34,3% |
| Baseline oficial | TF-IDF padrão + regressão logística | 44,56% |
| 1 — Lexical | TF-IDF palavras + caracteres, LinearSVC | **45,66%** |
| 2 — Atributos manuais | LightGBM em 87 atributos | 42,79% |
| 3a — Embeddings + classificador | LSA + regressão logística | 44,65% |
| 3b — Rede neural | TextCNN, embeddings ajustados | 44,94% |

## O que os resultados dizem

1. **Vetores pré-treinados genéricos não ajudam.** A média dos vetores do spaCy é a pior representação (42,1%), e o TextCNN com esses vetores congelados para em 42,8%. Eles foram treinados em notícias e texto geral, e 38% do vocabulário do córpus nem tem vetor.
2. **Aprender com o próprio córpus é melhor.** Word2Vec e Doc2Vec treinados nas respostas (43,4–43,6%) superam o spaCy, e o TextCNN ganha 2 pontos quando pode ajustar os embeddings. O vocabulário do e-SIC é muito específico.
3. **Tirar a média das palavras joga informação fora.** O LSA, que vem direto do TF-IDF, é a melhor representação densa (44,65%) e só empata com o baseline. Compactar a resposta em poucas centenas de números não acrescenta nada ao TF-IDF completo.
4. **O TextCNN decora rápido.** Entre a época 4 e a 5, o treino sobe de 61% para 68% e a validação cai de 44,9% para 43,9%. A janela boa é estreita, o que torna o modelo instável.
5. **O TextCNN quase ignora a classe do meio.** Prevê `c234` em só 16% dos casos e acerta 19,7% deles; em compensação acerta 56% dos `c1` e 60% dos `c5`. No teste, rotulou só 150 das 900 linhas como `c234`.
6. **Combinar não rompe o teto.** Regressão logística sobre as saídas dos modelos, nas mesmas dobras:

   | Combinação | Acurácia |
   |---|---|
   | Só TF-IDF | 45,64% |
   | TF-IDF + TextCNN | 45,87% |
   | TF-IDF + LSA | 45,87% |
   | TF-IDF + atributos + LSA + TextCNN | 45,66% |

   Os ganhos (+0,2 ponto) estão dentro do ruído entre dobras.
7. **Três famílias diferentes chegam ao mesmo lugar.** Palavras, atributos manuais, vetores densos e rede convolucional ficam todos entre 43% e 46%. Isso reforça que o limite vem dos rótulos ruidosos, e não da falta de ajuste dos modelos.

## Conclusão para o EP

- **Não entregar esta abordagem sozinha.** O TextCNN não é TF-IDF/BoW, então conta para inovação, mas a margem de +0,4 ponto sobre o baseline é pequena demais para garantir que ele não fique abaixo no teste (o que custa 20%).
- O LSA deriva do TF-IDF e provavelmente não pontua em inovação.
- O que ainda falta testar é um modelo com **contexto pré-treinado em português** (fine-tuning do BERTimbau). É a única família que traz conhecimento de fora do córpus em volume suficiente para tentar passar dos ~46%. Precisa de GPU.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `codigo/03_embeddings.py` | parte 3a (~17 min em 2 núcleos) |
| `codigo/03_textcnn.py` | parte 3b (~28 min em 2 núcleos) |
| `resultados/03_embeddings_grid.csv` | as 33 configurações da parte 3a, com acurácia por dobra |
| `resultados/03_textcnn_grid.csv` | as 10 configurações da parte 3b, com acurácia por dobra |
| `resultados/03_embeddings_log.txt`, `03_textcnn_log.txt` | saídas das execuções |
| `resultados/03_*_oof_melhor.npy` / `03_*_teste_melhor.npy` | probabilidades dos melhores modelos (treino fora da dobra e teste), para o ensemble |
| `predicoes/test1_embeddings.xlsx` | teste rotulado por LSA + regressão logística (`c1` 262 · `c234` 294 · `c5` 344) |
| `predicoes/test1_textcnn.xlsx` | teste rotulado pelo TextCNN (`c1` 349 · `c234` 150 · `c5` 401) |

Dependências adicionais: `spacy` com o modelo `pt_core_news_lg` (`python -m spacy download pt_core_news_lg`), `gensim`, `torch`.
