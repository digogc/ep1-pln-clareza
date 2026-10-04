# Abordagem 2 — Atributos manuais + gradient boosting (LightGBM)

Aqui o modelo **não vê nenhuma palavra diretamente**. Cada resposta vira 87 números que tentam medir o que faria um cidadão achá-la clara ou não, e um LightGBM (árvores com boosting) aprende sobre esses números.

**Resultado em uma linha:** máximo de **42,79%** — **abaixo** do baseline oficial (44,56%). Sozinha, esta abordagem não serve como entrega.

## Os 87 atributos

| Grupo | Qtde | Exemplos |
|---|---|---|
| Tamanho e estrutura | 14 | nº de palavras, frases e quebras de linha, itens numerados, fração de maiúsculas, dígitos e pontuação |
| Legibilidade e vocabulário | 10 | palavras por frase, sílabas por palavra, índice de Flesch adaptado ao português, fração de palavras raras (frequência Zipf, pacote `wordfreq`), riqueza lexical |
| Marcadores de conteúdo | 52 | contagem (absoluta e por 100 palavras) de expressões de 25 tipos: negativa, sigilo, "não compete", "trabalho adicional", anexo, link, lei/decreto, recurso, prazo, redirecionamento, despedida, juridiquês etc.; saudação com nome próprio |
| Órgão, datas e repetição | 11 | sigla mais frequente no texto, órgão e ano do protocolo, anos citados, quantas vezes o mesmo texto/início/final aparece no córpus |

Os atributos de repetição e os códigos das siglas são calculados sobre treino + teste juntos, **sem usar rótulos** (o enunciado permite usar o teste não rotulado). O código está em `codigo/atributos.py`.

## Como foi avaliado

Mesma validação cruzada da abordagem 1: 5 dobras fixas (`dados/folds.csv`), textos idênticos sempre na mesma dobra, métrica acurácia. Os atributos usam o texto original, sem limpeza, porque os espaços duplos marcam as quebras de linha.

## Busca em grade

| Parâmetro | Valores |
|---|---|
| `num_leaves` | {7 · 15 · 31 · 63} |
| `min_child_samples` | {20 · 60} |
| `colsample_bytree` | {0,3 · 0,6} |
| `n_estimators` | {100 · 200 · 300 · 400 · 600} |
| Fixos | `learning_rate=0,03`, `subsample=0,7`, `reg_lambda=1` |

São 80 configurações; a tabela completa está em `02_atributos_grid.csv`.

## Resultados

| Modelo | Acurácia validação | Acurácia treino |
|---|---|---|
| Classe majoritária | 34,3% | — |
| Baseline oficial (TF-IDF + regressão logística) | 44,56% | 71,1% |
| Melhor da abordagem 1 (TF-IDF palavras + caracteres, LinearSVC) | 45,66% | 77,7% |
| **Melhor LightGBM nos 87 atributos** | **42,79%** ± 0,63 | 56,1% |
| Regressão logística nos 85 atributos numéricos (melhor C) | 40,75% | — |

### Máximo da abordagem: 42,79%

**LightGBM com `num_leaves=31`, `min_child_samples=60`, `colsample_bytree=0,3`, `n_estimators=100`.**
Fica **1,8 ponto abaixo do baseline oficial** e perde nas 5 dobras (de −0,8 a −2,7 ponto).

As 80 configurações ficam todas entre 41,8% e 42,8%: a grade não é o gargalo, os atributos é que são.

### Quanto cada grupo de atributos consegue sozinho

| Atributos usados | Qtde | Acurácia | Peso no modelo completo (ganho) |
|---|---|---|---|
| Todos | 87 | 42,79% | 100% |
| Só marcadores de conteúdo | 52 | 41,39% | 36,3% |
| Só órgão, datas e repetição | 11 | 40,30% | 24,2% |
| Só tamanho e estrutura | 14 | 39,90% | 18,6% |
| Só legibilidade e vocabulário | 10 | 39,50% | 20,9% |

O atributo individual mais importante é a **sigla principal** (14,1% do ganho), que funciona como um identificador do órgão que respondeu. A lista completa está em `02_atributos_importancia.csv`.

### Matriz de confusão (previsões fora da dobra)

| | previsto c1 | previsto c234 | previsto c5 | acerto da classe |
|---|---|---|---|---|
| **real c1** | 2.345 | 1.732 | 2.270 | 36,9% |
| **real c234** | 1.673 | 2.221 | 2.959 | 32,4% |
| **real c5** | 1.036 | 1.824 | 4.032 | 58,5% |

## O que os atributos revelam sobre o problema

Proporção de cada classe no treino quando o marcador aparece (geral: c1 31,6% · c234 34,1% · c5 34,3%):

| Resposta contém… | Linhas | c1 | c234 | c5 |
|---|---|---|---|---|
| "trabalho adicional", "desproporcional", "genérico" | 716 | **46,4%** | 37,4% | 16,2% |
| "não compete", "não é o canal", "não trata de" | 733 | **44,9%** | 34,1% | 21,0% |
| sigilo, informação restrita/pessoal | 1.667 | **40,6%** | 36,3% | 23,2% |
| negativa ("não é possível", "indeferido") | 2.170 | **40,3%** | 35,1% | 24,6% |
| menção a anexo/arquivo | 6.791 | 26,5% | 34,3% | **39,1%** |

Tamanho da resposta:

| Nº de palavras | c1 | c234 | c5 |
|---|---|---|---|
| até 41 | 29,6% | 29,2% | **41,2%** |
| 42 a 79 | 30,0% | 32,4% | 37,6% |
| 80 a 133 | 29,3% | 33,3% | 37,4% |
| 134 a 221 | 34,2% | 36,3% | 29,5% |
| mais de 221 | 34,9% | 39,5% | 25,7% |

## O que os resultados dizem

1. **Os sinais existem e fazem sentido, mas são fracos.** Respostas que negam o pedido puxam para `c1`; respostas curtas ou com anexo puxam para `c5`. Mesmo o marcador mais forte só leva a classe dominante a ~46%, e ele aparece em menos de 4% das respostas.
2. **A nota mede satisfação, não legibilidade.** Os atributos clássicos de clareza textual (Flesch, tamanho de frase, palavras raras) são o grupo mais fraco: 39,5%. O que mais ajuda é saber *se o cidadão recebeu o que pediu* e *qual órgão respondeu*.
3. **87 números resumem menos do que o vocabulário inteiro.** O TF-IDF captura esses mesmos sinais e muitos outros (nomes de órgãos, assuntos, fórmulas de resposta), por isso fica 2–3 pontos acima.
4. **Menos overfitting que o TF-IDF, mas ainda presente.** Treino 56% contra validação 43%. Modelos maiores sobem o treino até 92% sem ganhar nada na validação.
5. **O modelo "chuta" `c5` com frequência.** Acerta 58,5% dos `c5`, mas só 36,9% dos `c1` e 32,4% dos `c234`. No teste ele prevê `c5` em 407 das 900 linhas.
6. **A não linearidade ajuda.** LightGBM (42,8%) supera a regressão logística nos mesmos atributos (40,7%): as interações entre atributos importam.

## Vale combinar com a abordagem 1?

Os dois modelos concordam em 61,9% das respostas. Em 12,8% dos casos só o LightGBM acerta, e em 15,7% só o TF-IDF acerta. Um teste rápido de combinação (regressão logística sobre as saídas dos dois, nas mesmas dobras) deu:

| Modelo | Acurácia |
|---|---|
| Só TF-IDF | 45,64% |
| TF-IDF + atributos | 45,94% |

Ganho de **+0,3 ponto** — dentro do ruído entre dobras. A complementaridade é pequena.

## Conclusão para o EP

- **Não entregar esta abordagem sozinha:** fica abaixo do baseline oficial, o que custa 20% da nota.
- Ela conta como método diferente de TF-IDF/BoW e rende boa **explicação** para o relatório e a apresentação (o que torna uma resposta "clara" para o usuário).
- Pode entrar como componente de um ensemble, mas o ganho esperado é pequeno.
- O salto de acurácia precisa vir de um modelo que entenda o texto melhor que TF-IDF — o candidato natural é o fine-tuning do BERTimbau.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `codigo/atributos.py` | cálculo dos 87 atributos |
| `codigo/02_atributos_lgbm.py` | grade, análise por grupo, modelo final (rode com `python codigo/02_atributos_lgbm.py`; ~20 min em 2 núcleos) |
| `resultados/02_atributos_grid.csv` | as 80 configurações, com acurácia por dobra |
| `resultados/02_atributos_grupos.csv` | acurácia de cada grupo de atributos e da regressão logística |
| `resultados/02_atributos_importancia.csv` | importância (ganho) de cada atributo |
| `resultados/02_atributos_log.txt` | saída da execução |
| `resultados/02_atributos_oof_melhor.npy` / `02_atributos_teste_melhor.npy` | probabilidades do melhor modelo (treino fora da dobra e teste), para o ensemble |
| `predicoes/test1_atributos.xlsx` | teste rotulado por esta abordagem (`c1` 238 · `c234` 255 · `c5` 407) — **não usar como entrega** |

Dependências adicionais: `lightgbm`, `wordfreq`.
