# Abordagem 1 — Modelos lexicais (TF-IDF + classificador linear)

Esta é a família do baseline oficial do professor. Serve para duas coisas: medir a barra que precisa ser superada e descobrir o **máximo** que esse tipo de modelo alcança. Pelo enunciado, ela **não pontua em "inovação"**.

## Como foi avaliado

- **Validação cruzada de 5 dobras**, estratificada por classe. A divisão está fixa em `dados/folds.csv` e será reutilizada por todas as abordagens, para que os números sejam comparáveis.
- **Textos idênticos ficam sempre na mesma dobra.** O treino tem 1.660 linhas repetidas; sem esse cuidado, a mesma resposta apareceria no treino e na validação e inflaria o resultado.
- O TF-IDF é ajustado apenas na parte de treino de cada dobra.
- **Pré-processamento:** só a junção de espaços e quebras de linha repetidos; o TF-IDF passa tudo para minúsculas.
- Métrica: **acurácia** (a do enunciado).

## O que foi testado (busca em grade)

| Parâmetro | Valores |
|---|---|
| Representação | palavras 1-gram · palavras 1–2-gram · caracteres 2–5-gram (`char_wb`) · palavras 1–2-gram + caracteres 2–5-gram |
| TF-IDF | `sublinear_tf=True`; `min_df=2` (palavras) e `min_df=3` (caracteres) |
| Regressão logística — `C` | {0,3 · 1 · 3} |
| SVM linear (LinearSVC) — `C` | {0,03 · 0,1 · 0,3} |
| Complement Naive Bayes — `alpha` | {0,1 · 0,5} |

São 32 configurações mais o baseline. A tabela completa está em `01_tfidf_grid.csv`.

## Resultados

Referências:

| Modelo | Acurácia (validação cruzada) |
|---|---|
| Classe majoritária | 34,3% |
| **Baseline oficial** — TF-IDF padrão + regressão logística padrão | **44,56%** ± 0,73 |

Melhor de cada representação:

| Representação | Nº de atributos | Melhor classificador | Acurácia validação | Acurácia treino |
|---|---|---|---|---|
| Palavras 1-gram | 22,7 mil | LinearSVC, C=0,1 | 45,13% | 66,3% |
| Palavras 1–2-gram | 144 mil | Regressão logística, C=1 | 45,37% | 78,7% |
| Caracteres 2–5-gram | 142 mil | LinearSVC, C=0,3 | 45,37% | 74,5% |
| **Palavras 1–2-gram + caracteres 2–5-gram** | 286 mil | **LinearSVC, C=0,1** | **45,66%** ± 0,92 | 77,7% |

Melhor de cada classificador (em qualquer representação):

| Classificador | Melhor acurácia |
|---|---|
| LinearSVC | 45,66% |
| Regressão logística | 45,62% |
| Complement Naive Bayes | 44,88% |

### Máximo da abordagem: 45,66%

**TF-IDF de palavras (1–2-gram) + caracteres (2–5-gram), LinearSVC com C=0,1.**
Ganho de **+1,1 ponto percentual** sobre o baseline oficial, e o modelo vence o baseline nas 5 dobras (de +0,7 a +1,8 ponto).

Matriz de confusão do melhor modelo (previsões fora da dobra, 20.092 respostas):

| | previsto c1 | previsto c234 | previsto c5 | acerto da classe |
|---|---|---|---|---|
| **real c1** | 2.843 | 1.897 | 1.607 | 44,8% |
| **real c234** | 1.941 | 2.505 | 2.407 | 36,6% |
| **real c5** | 1.202 | 1.865 | 3.825 | 55,5% |

## O que os resultados dizem

1. **A abordagem lexical está saturada.** As 32 configurações ficam todas entre 43,9% e 45,7% — uma faixa de menos de 2 pontos. Trocar n-gramas, classificador ou regularização quase não muda nada. Não adianta refinar mais essa grade.
2. **O ganho sobre o baseline é real, mas pequeno.** +1,1 ponto, consistente nas 5 dobras, porém do mesmo tamanho do desvio entre dobras (~0,9). Como a melhor configuração foi escolhida entre 32 na mesma validação, o número também é levemente otimista. No teste de 900 linhas, 1 ponto equivale a 9 respostas — pode sumir por acaso.
3. **Há muito overfitting.** A acurácia no treino vai de 56% a 89%, contra ~45% na validação. Quanto menos regularização (C maior), mais o treino sobe e a validação cai. O modelo decora vocabulário que não generaliza.
4. **A classe do meio é a mais difícil.** `c234` tem só 36,6% de acerto e é confundida quase igualmente com `c1` e `c5`. `c5` é a mais fácil (55,5%).
5. **O problema tem um teto baixo.** O rótulo é a nota que um usuário deu, e respostas idênticas recebem notas diferentes (291 dos 463 textos repetidos têm rótulos conflitantes). Parte da nota não está no texto.

## Conclusão para o EP

- Para superar o baseline com segurança e pontuar em inovação, é preciso **outra família de modelo** — não mais ajuste nesta.
- Este modelo fica como **referência forte** e como candidato a entrar em uma combinação (ensemble) com as próximas abordagens.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `codigo/common.py` | leitura dos dados, dobras, gravação da planilha de teste |
| `codigo/01_tfidf.py` | esta abordagem (rode com `python codigo/01_tfidf.py`; ~15 min em 2 núcleos) |
| `dados/folds.csv` | dobra de cada linha do treino |
| `resultados/01_tfidf_grid.csv` | todas as configurações, com acurácia por dobra |
| `resultados/01_tfidf_log.txt` | saída da execução |
| `resultados/01_tfidf_oof_melhor.npy` / `01_tfidf_teste_melhor.npy` | pontuações do melhor modelo (treino fora da dobra e teste), para o ensemble |
| `predicoes/test1_tfidf.xlsx` | teste rotulado por esta abordagem (900 linhas, 2 colunas; `c1` 276 · `c234` 290 · `c5` 334) |

Dependências: `pandas`, `scikit-learn`, `scipy`, `joblib`, `openpyxl`.
