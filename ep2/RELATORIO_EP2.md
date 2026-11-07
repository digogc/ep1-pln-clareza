# ACH2118 Introdução ao Processamento de Língua Natural — Relatório do EP2

## 1. Integrantes

| Nome | Número USP |
|---|---|
| Dimitri Prado | 14746022 |
| Lucas de Lyra Monteiro | 15471435 |
| Rodrigo Gonçalves Cardoso | 14658330 |
| Rodrigo Lyusei Suguimoto | 14670877 |

## 2. Estratégia

A tarefa é a mesma do EP1: classificar a clareza de respostas do e-SIC em {`c1`, `c234`, `c5`}. A evolução em relação ao EP1 vem de **modelos neurais de linguagem pré-treinados** incorporados ao stacking do EP1:
- **dois BERTimbau ajustados para a tarefa (fine-tuning)** (`neuralmind/bert-base-portuguese-cased`), um lendo 128 e outro 256 subpalavras da resposta;
- os **embeddings de sentença do Qwen3-Embedding-0.6B**, no lugar dos do MiniLM usados no EP1.

**De onde partimos: o EP1.**
- **O que testamos:** no EP1 avaliamos várias famílias de modelos, sempre contra o baseline oficial e nas mesmas dobras:
  - TF-IDF de palavras e de caracteres com regressão logística, SVM e Naive Bayes;
  - 87 atributos escritos à mão com LightGBM;
  - embeddings estáticos (Word2Vec, Doc2Vec, LSA) e uma rede convolucional (TextCNN);
  - embeddings de sentença congelados (MiniLM e e5-small);
  - memórias de respostas idênticas e de respostas vizinhas.
- **O modelo entregue foi um stacking de quatro blocos**, escolhidos passo a passo pela validação cruzada: regressão logística sobre TF-IDF de palavras, SVM sobre palavras e caracteres, regressão logística sobre os embeddings do MiniLM e um classificador `c1` × `c5` treinado só nos extremos.
- **Desempenho:**
  - 46,98% na validação cruzada (46,92% quando recalculado no ambiente do EP2, seção 7), contra 45,54% do baseline;
  - 48,56% no test1.
- **Duas lições do EP1 orientaram o EP2:**
  - **Representações pré-treinadas complementam o TF-IDF.** Sozinhos, os embeddings do MiniLM eram fracos (43,08%), mas melhoravam o stacking, porque erram em respostas diferentes das que o TF-IDF erra.
  - **O BERT ficou pela metade.** Um integrante do grupo (Dimitri) ajustou um BERTimbau no EP1, que acertou 44,97% numa divisão própria (80% treino, 20% teste), mas ele não entrou no stacking a tempo. O modelo tinha sido treinado em 80% do treino, sem previsões fora da dobra para o restante, e a inferência em CPU não terminou no prazo.

**Por que BERT.**
- **Outros grupos da turma usaram BERT no EP1 com bons resultados.** Isso indicou que ajustar um modelo pré-treinado à tarefa era o caminho mais promissor para o EP2.
- **É o passo seguinte à primeira lição do EP1.** No fine-tuning, todas as camadas do modelo se adaptam à tarefa, em vez de usar embeddings congelados como o MiniLM.
- **O BERTimbau foi pré-treinado em português brasileiro.** E já tínhamos um ajuste dele do EP1 como referência: o truncamento cabeça + cauda que usamos é o mesmo do modelo do Dimitri.
- **O enunciado pede inovação.** TF-IDF com regressão logística, SVM ou Naive Bayes, ou repetir o EP1, não pontuam.

**O modelo final (v4) é um stacking de seis modelos-base**, combinados por um meta-classificador de regressão logística:

1. `lr_pal`: regressão logística sobre TF-IDF de palavras (1–2-gramas);
2. `svc`: SVM linear sobre TF-IDF de palavras (1–2-gramas) e de caracteres (2–5-gramas);
3. `qwen3`: regressão logística sobre embeddings de sentença do **Qwen3-Embedding-0.6B** (congelado, com uma instrução que descreve a tarefa);
4. `extremos`: regressão logística `c1` × `c5` treinada apenas nos extremos (eixo "ruim–bom");
5. `bert`: **BERTimbau ajustado com 128 subpalavras** de entrada;
6. `bert256`: **BERTimbau ajustado com 256 subpalavras** de entrada.

No stacking, cada BERT entra resumido num único escore ordinal, `logit(c5) − logit(c1)`, em vez dos três logits. É o mesmo eixo "ruim → bom" do bloco `extremos`, e essa forma foi a melhor na validação cruzada (seção 4).

**Como funciona o stacking.** São dois níveis de modelos, e a arquitetura é a mesma do EP1.
1. **Nível 0, os modelos-base.** Cada um dos seis modelos lê a resposta e devolve poucos números, 12 no total por resposta:
   - as pontuações das três classes (`lr_pal`, `svc` e `qwen3`, 3 números cada);
   - o escore `c1` × `c5` do `extremos` (1 número);
   - o escore `logit(c5) − logit(c1)` de cada BERT (1 número cada).
2. **Previsões fora da dobra.** Para o nível 1 aprender sem vazamento, os 12 números de cada resposta do treino vêm de modelos que não viram aquela resposta. O treino é dividido em 5 dobras: cada modelo-base treina em 4 e prevê a quinta. O BERT 128 usa 2 dobras.
3. **Nível 1, o meta-classificador.** Uma regressão logística (`C` = 0,1) sobre os 12 números padronizados aprende quanto confiar em cada bloco e em que direção. É ela que dá a classe final.
4. **No test2:**
   - os modelos sobre TF-IDF e embeddings são retreinados com todo o treino;
   - cada BERT prevê pela média dos modelos das dobras;
   - o meta-classificador, treinado nas previsões fora da dobra, combina os 12 números.

**O que mudou em relação ao EP1.** Os blocos de TF-IDF, o meta-classificador e as dobras continuam os mesmos. Entraram os dois BERTs, e o Qwen3 substituiu o MiniLM.

**Quanto cada bloco contribui.** Validação cruzada da v4 sem um bloco de cada vez (dados do `e22`, seção 4):

| Bloco retirado | Acurácia | Diferença | Dobras que pioram |
|---|---|---|---|
| nenhum (v4) | 47,91% | — | — |
| BERT 256 | 47,36% | −0,55 | 5 de 5 |
| regressão logística sobre TF-IDF | 47,54% | −0,37 | 4 de 5 |
| Qwen3 | 47,69% | −0,22 | 4 de 5 |
| SVM | 47,75% | −0,16 | 5 de 5 |
| extremos | 47,78% | −0,13 | 4 de 5 |
| BERT 128 | 47,81% | −0,10 | 3 de 5 |

Todos os blocos contribuem, e o BERT 256 é o que mais pesa. Nos pesos do meta-classificador, o escore dele é a entrada de maior peso: +0,24 para `c5` e −0,22 para `c1`, com as entradas padronizadas.

O modelo foi construído em versões, cada uma validada no treino antes de substituir a anterior:

| Versão | Mudança | Validação cruzada |
|---|---|---|
| EP1 | stacking de 4 blocos (`lr_pal`, `svc`, `emb` MiniLM, `extremos`) | 46,92% |
| v2 | + BERTimbau 128 | 47,29% |
| v3 | + BERTimbau 256 | 47,74% |
| **v4 (final)** | **Qwen3-Embedding no lugar do MiniLM** | **47,91%** |

**Por que os BERTs ajudam (o que os resultados mostram).**
- **Com 256 subpalavras, o BERT sozinho já passa o baseline.** As respostas são longas: a mediana tem 182 subpalavras, e 63% delas passam de 128. Com 256 subpalavras, o BERTimbau sozinho acerta 46,23%, contra 44,88% com 128 e 45,54% do baseline oficial.
- **Os BERTs são complementares aos modelos de TF-IDF.** Eles leem a resposta de outro jeito (subpalavras, ordem e contexto) e erram em respostas diferentes.
- **No stacking, essa diferença vira ganho.** No treino, o EP2 passa a acertar 1.283 respostas que o EP1 errava e passa a errar 1.084 que o EP1 acertava.

**Outras abordagens avaliadas e não adotadas** (detalhes e números na seção 4):
- variações do BERT: LoRA, pré-treino adaptativo à tarefa (TAPT), objetivos ordinal e de extremos, 512 subpalavras e 10 dobras;
- blocos extras no stacking: vizinhos mais próximos (kNN) e 34 atributos escritos à mão;
- todas as 32.767 combinações de 15 blocos no stacking;
- meta-classificadores não lineares;
- um LLM local (Qwen3-1.7B) em zero-shot.

**Dados.** Usamos apenas os arquivos fornecidos: o treino é o mesmo do EP1 (`train.xlsx`), como pede o enunciado, e nenhum dado externo foi usado. No início do EP2 chegamos a localizar a referência da base original, publicada com um artigo do próprio professor. Seguindo a orientação dele de usar só o que foi fornecido, ela não foi baixada nem usada. Os modelos pré-treinados (BERTimbau, Qwen3) são públicos e foram usados como ponto de partida. Nenhum dado dos conjuntos de teste é usado com rótulo. Cerca de 11% dos textos do test1 (100) e do test2 (106) são idênticos a algum texto do treino. São modelos padrão de resposta dos órgãos (mediana de 13 cópias no treino por texto), ou seja, pedidos diferentes respondidos com o mesmo texto, e não as mesmas linhas.

## 3. Pré-processamento

- **Todos os modelos:** desfaz os escapes de caracteres de controle do Excel presentes nos arquivos do EP2 (por exemplo `_x0096_`), o que torna os textos idênticos aos do EP1, e junta espaços e quebras de linha repetidos. Sem remoção de stopwords, stemming ou lematização.
- **BERTimbau:**
  - tokenizador WordPiece do próprio modelo, que mantém maiúsculas e acentos;
  - truncamento **cabeça + cauda**: as 63 primeiras e as 63 últimas subpalavras no BERT 128; as 127 primeiras e as 127 últimas no BERT 256;
  - o início traz a saudação e o assunto; o fim traz a conclusão, o encaminhamento e a assinatura do órgão;
  - comprimento das respostas do treino em subpalavras: mediana 182; 63% passam de 128, 36% de 256 e 11% de 512.
- **TF-IDF:** minúsculas, `sublinear_tf=True`, `min_df=2` (palavras) e `min_df=3` (caracteres `char_wb`); vocabulário e IDF calculados sobre treino + test1 + test2, sem rótulos.
- **Embeddings do Qwen3:**
  - cada resposta é precedida da instrução da tarefa, como recomenda o modelo: `Instruct: Classify how clear a Brazilian government agency's answer to a citizen's freedom-of-information request is` / `Query: <resposta>`;
  - truncamento em 256 tokens;
  - vetores normalizados (norma 1) e padronizados antes da regressão logística.

## 4. Parâmetros avaliados na busca em grade

**BERTimbau (fine-tuning)**

| Parâmetro | Valores |
|---|---|
| Modelo de partida | {BERTimbau base, multilingual-e5-small, BERTimbau com pré-treino adaptativo à tarefa (TAPT)} |
| Comprimento máximo (cabeça + cauda) | {128, 256, 512} subpalavras |
| Forma de ajuste | {fine-tuning completo, LoRA com posto 16} |
| Camadas congeladas | {0, 4, 6, 8} (além dos embeddings) |
| Objetivo de treino | {entropia cruzada de 3 classes, ordinal (um escore com alvo 0 / 0,5 / 1), extremos (escore c1 × c5 treinado só nessas duas classes)} |
| Taxa de aprendizado | 3e-5 (fine-tuning) e 3e-4 (LoRA), com aquecimento linear em 10% dos passos e decaimento linear |
| Épocas | {1, 2} |
| Lote | 16, com ou sem agrupamento dos lotes por tamanho do texto (só muda a velocidade) |
| Dobras do fine-tuning (previsões fora da dobra) | {2, 5, 10} |
| Entrada do BERT no stacking | {3 logits, log-softmax, 3 logits padronizados, escore `logit(c5) − logit(c1)`, média dos escores de vários BERTs} |

**Embeddings de sentença congelados**

| Parâmetro | Valores |
|---|---|
| Modelo | {paraphrase-multilingual-MiniLM-L12-v2, multilingual-e5-small, Qwen3-Embedding-0.6B} |
| `C` da regressão logística | {0,003; 0,01; 0,03; 0,1} |

**Stacking**

| Parâmetro | Valores |
|---|---|
| Blocos candidatos | os da v2 + {BERT 128 de 5 dobras, BERT 256 de 5 e de 10 dobras, BERT 512, BERT ordinal, BERT de extremos, BERTs da 1ª época, Qwen3 (no lugar do MiniLM ou junto dele), 34 atributos escritos à mão, kNN sobre TF-IDF, kNN sobre MiniLM, LLM local} |
| Combinações de blocos | 24 configurações escolhidas à mão e, depois, todos os 32.767 subconjuntos de 15 blocos |
| Meta-classificador | {regressão logística, LightGBM, HistGradientBoosting} |
| `C` da regressão logística do meta | {0,01; 0,03; 0,1; 0,3; 1} |
| Ajuste dos vieses de classe para a acurácia | {não, sim (escolhidos em validação cruzada interna)} |

**Demais modelos-base** (herdados do EP1)

| Parâmetro | Valores |
|---|---|
| Representação lexical | {palavras 1-gram, palavras 1–2-gram, caracteres 2–5-gram, palavras + caracteres} |
| Classificador lexical | {regressão logística, LinearSVC, Complement Naive Bayes} |
| `C` (regressão logística / LinearSVC) | {0,3; 1; 3} / {0,01; 0,03; 0,1; 0,3; 1} |

Cada fine-tuning completo do BERT leva de 6 a ~60 h em CPU (seção 6), por isso a grade do BERT não é exaustiva. Modelo de partida e camadas congeladas foram comparados pelo custo medido em treinos curtos. As outras opções foram comparadas em treinos completos, a maioria na mesma metade de validação (BERTs de 2 dobras), antes de investir nas 5 dobras.

**Resultados da busca** (acurácia em validação cruzada; "metade 0" indica comparação nas mesmas 10.046 respostas de validação do BERT de 2 dobras):

| O que foi testado | Resultado | Decisão |
|---|---|---|
| Comprimento 128 / 256 / 512 (BERT sozinho, 5 dobras) | 45,21 / **46,23** / 46,42%; no stacking: v2 + 256 = **47,74%**, v2 + 512 = 47,65%, v2 + 256 + 512 = 47,69% | 256, pela regra fixada antes (maior média no stacking; empate → menos blocos) |
| Manter o BERT 128 ao lado do 256 | v3 sem o BERT 128: 47,69% (contra 47,74%) | mantém os dois |
| LoRA (posto 16) × fine-tuning completo (metade 0) | 43,73 × **44,66%**; McNemar p = 0,018 | fine-tuning completo |
| TAPT: pré-treino mascarado nos textos do treino, 2 épocas (6,3 h; perda de 1,93 para 0,71) (metade 0) | BERT sozinho 43,93 × 44,66% (−0,74); stacking 47,45 × 47,26% | BERTimbau original: a regra fixada antes exigia +0,3 ponto no BERT sozinho |
| Objetivo ordinal × entropia cruzada (metade 0, regressão logística no escore) | 43,63 × 43,87% | entropia cruzada |
| 1 × 2 épocas (BERT 128, 5 dobras) | 43,42 × **45,21%** | 2 épocas |
| Entrada do BERT no stacking (v2) | 3 logits 47,06 / log-softmax 47,05 / padronizados 47,15 / **escore 47,29%** | escore |
| Média dos escores do BERT 128 e do 256 num bloco só | 47,38% (contra 47,74% com blocos separados) | blocos separados |
| Qwen3 × MiniLM (embeddings sozinhos) | **43,82** × 43,08% | Qwen3 |
| Qwen3 no stacking (v3) | no lugar do MiniLM: **47,91%** (melhor em 4 das 5 dobras); junto do MiniLM: 47,88% | Qwen3 no lugar do MiniLM (v4) |
| `C` dos embeddings do Qwen3 | 0,003 / 0,01 / 0,03: **43,86** / 43,48 / 43,53% | 0,003 |
| kNN (k = 25) sobre TF-IDF / MiniLM (v2) | 47,06 / 47,27% (contra 47,29%) | não entra |
| 34 atributos escritos à mão (links, e-mails, números, citações de leis, anexos, saudações, negativas etc.) (v2) | 47,16% (contra 47,29%) | não entra |
| Meta-classificador não linear | LightGBM 46,73% (v2: 47,29%); HistGradientBoosting 47,58% (v3: 47,74%) | regressão logística |
| `C` do meta (v3) | 0,01 / 0,03 / 0,1 / 0,3 / 1: 47,76 / 47,75 / 47,74 / 47,75 / 47,72% | 0,1 (diferença desprezível) |
| Vieses de classe ajustados (v2 / v3) | 47,01 / 47,55% (contra 47,29 / 47,74%) | não |
| LLM local Qwen3-1.7B (int4, GPU integrada), zero-shot, probabilidades das notas 1 a 5 (piloto em 2.000 respostas) | sozinho 34,55%; v2 + LLM: +0,45 ± 0,44 ponto | não seguiu: ganho incerto e ~13 h de GPU para o conjunto todo |
| 10 dobras no BERT 256 (cada modelo treina com 90% dos dados; hipótese levantada em aula) | BERT sozinho **46,72%** contra 46,23% do de 5 dobras (melhor em 4 das 5 dobras; McNemar p = 0,083) e acima do 512 (46,42%); no lugar do 256 de 5 dobras na v4: 47,73% (contra 47,91%; melhor em 2 das 5 dobras) | não entra: o ganho do BERT sozinho não chega ao stacking |
| BERT ordinal e BERT de extremos (2 dobras) como blocos extras | efeito médio nas 32.767 combinações: −0,01 e +0,00 ponto | não entram |

**Busca de combinações e otimismo da seleção.** Quando se escolhe a melhor entre muitas configurações, a média dela na validação cruzada é otimista. Por isso estimamos, com **validação cruzada aninhada**, quanto rende o procedimento "escolher a configuração com a maior média": em cada dobra externa, a escolha é feita só com as outras 4 e avaliada na que ficou de fora.
- **24 configurações escolhidas à mão** (blocos, `C` do meta, meta não linear, vieses): o procedimento rendeu 47,60%, abaixo dos 47,74% da v3 fixa.
- **Todos os 32.767 subconjuntos de 15 blocos**, depois que o BERT de 10 dobras terminou (`e22_todas_combinacoes.py`, ~4 h em 8 processos):
  - a melhor combinação teve **48,12%** (+0,21 sobre a v4) com 9 blocos: RL e SVM sobre TF-IDF, extremos, MiniLM, Qwen3, BERT 128 de 2 dobras, BERT 256 de 5 dobras, BERT 512 e BERT de extremos;
  - ela ganha da v4 em só 3 das 5 dobras, e a diferença nas previsões fora da dobra não é significativa (494 × 452 acertos exclusivos; p = 0,18);
  - a v4 fica em 208º lugar; as 25 melhores ficam entre 47,98 e 48,12%;
  - o procedimento aninhado rendeu de novo **47,60%**, abaixo dos 47,91% da v4 fixa, e cada dobra externa escolheu uma combinação diferente;
  - pela regra fixada antes (ganho ≥ 0,2 ponto, em pelo menos 4 das 5 dobras, aninhado ≥ v4), a v4 foi mantida.
- **Efeito médio de cada bloco** (média das combinações com ele menos a das sem ele): RL sobre TF-IDF +0,35; SVM +0,27; BERT 512 +0,20; BERT 256 de 10 dobras +0,18; de 5 dobras +0,16; kNN sobre MiniLM +0,12; Qwen3 +0,10. BERT 128, BERT ordinal, BERT de extremos, MiniLM e atributos ficam perto de zero, e os 15 blocos juntos dão 47,69%.

Ou seja, diferenças de 0,1–0,2 ponto entre configurações são da ordem do ruído de seleção. Por isso as trocas de versão exigiram ganho em quase todas as dobras.

Um LLM via API (Claude), com várias estratégias de prompt e garantias contra vazamento, ficou implementado (`ep2/experimentos/codigo/e06_llm.py`), mas não foi executado nesta versão.

## 5. Valores ótimos

| Componente | Parâmetro | Valor |
|---|---|---|
| `bert256` | Modelo | `neuralmind/bert-base-portuguese-cased` |
| | Comprimento máximo / truncamento | 256 subpalavras, cabeça + cauda (127 + 127) |
| | Camadas congeladas | embeddings + 4 camadas inferiores (8 camadas ajustadas) |
| | Taxa de aprendizado / épocas / lote | 3e-5 / 2 / 16 |
| | Dobras / entrada no stacking | 5 / escore `logit(c5) − logit(c1)` |
| `bert` | Comprimento máximo / truncamento | 128 subpalavras, cabeça + cauda (63 + 63) |
| | Demais parâmetros | iguais ao `bert256`, mas com 2 dobras |
| `qwen3` | Modelo / comprimento / `C` | Qwen3-Embedding-0.6B / 256 tokens, com instrução / 0,003 |
| `lr_pal` | Representação / `C` | palavras 1–2-gram / 1 |
| `svc` | Representação / `C` | palavras 1–2-gram + caracteres 2–5-gram / 0,1 |
| `extremos` | Representação / `C` | palavras + caracteres / 1 |
| Meta-classificador | Tipo / `C` | regressão logística / 0,1 |

**Justificativa das escolhas:**
- **BERTimbau, fine-tuning completo:** foi pré-treinado em português brasileiro. O LoRA treina menos parâmetros, mas acertou significativamente menos na mesma metade (p = 0,018).
- **Comprimento 256:** com 128, 63% das respostas são cortadas; com 256, 36%. O 512 é um pouco melhor sozinho, mas não melhora o stacking, e os escores do 256 e do 512 têm correlação de 0,91. A escolha seguiu a regra fixada antes de ver os resultados.
- **Dois BERTs no stacking:** o de 128 e o de 256 têm correlação de 0,88. Tirar o de 128 custa 0,05 ponto, e a média dos dois num bloco só custa 0,36 ponto.
- **Camadas congeladas:** congelar os embeddings e as 4 camadas inferiores funciona como regularização para um conjunto pequeno e com rótulos ruidosos.
- **Épocas:** 2. Com 1 época, o BERT 128 de 5 dobras perde 1,8 ponto sozinho e não ajuda o stacking.
- **Qwen3 no lugar do MiniLM:** é a configuração de maior média entre as 24 testadas e ganha em 4 das 5 dobras (+0,17 ponto), mas a vantagem sobre a v3 não é estatisticamente significativa. A troca e o critério estão discutidos na seção 8.

## 6. Procedimento

**Etapas do EP2, em ordem.**
1. **Ponto de partida, EP1 (até 04/10):** stacking de quatro blocos (seção 2).
2. **Preparação (29/10):**
   - leitura dos arquivos do EP2 e reversão dos escapes do Excel (`_x0096_`), que deixa os textos idênticos aos do EP1;
   - contagem dos textos dos testes repetidos no treino;
   - recálculo do stacking do EP1 no ambiente do EP2 (46,92%).
3. **v2 (29/10):**
   - fine-tuning do BERTimbau com 128 subpalavras em 2 dobras (~6 h), com o modelo de partida e as camadas congeladas escolhidos por treinos curtos;
   - escolha da forma de entrada do BERT no stacking (o escore);
   - resultado: 47,29%. O McNemar contra o EP1 no test1 deu p = 0,047, no limite, e por isso seguimos buscando um ganho mais firme.
4. **Busca de melhorias (29 e 30/10):**
   - kNN, 34 atributos escritos à mão, variações do meta-classificador e vieses de classe;
   - embeddings do Qwen3 e do e5;
   - LoRA × fine-tuning completo, objetivo ordinal, 256 subpalavras numa metade dos dados e TAPT;
   - LLM local em zero-shot; o LLM via API ficou implementado, mas não foi executado.

   As variações do BERT foram comparadas na mesma metade de validação.
5. **Infraestrutura (30/10 em diante):** previsões na GPU integrada, checkpoints retomáveis (o Windows reiniciou sozinho no dia 30) e, depois, lotes agrupados por tamanho.
6. **BERTs de 5 dobras (31/10 a 03/11):** BERT 128 (31/10), BERT 256 (~39 h, pronto em 02/11) e BERT 512 (03/11).
7. **v3 (03/11):** pela regra fixada antes (maior média no stacking entre 256, 512 e 256 + 512), entrou o BERT 256. Resultado: 47,74%, com p = 0,036 contra o EP1.
8. **Combinações e v4 (03/11):**
   - 24 configurações com validação aninhada; a regra mandava manter a v3;
   - com o Qwen3 no lugar do MiniLM, a v4 deu 47,91% e p = 0,015, e foi adotada (decisão discutida na seção 8);
   - três variantes com o BERT 512, vistas depois, não mudaram a escolha;
   - as versões ficaram registradas em `ep2/versoes/`.
9. **10 dobras (03 a 06/11):** BERT 256 em 10 dobras (~58 h), hipótese levantada em aula, seguido do BERT ordinal e do BERT de extremos, em 2 dobras.
10. **Busca exaustiva (06/11):**
    - todas as 32.767 combinações de 15 blocos, com validação aninhada (~4 h); pela regra, a v4 foi mantida;
    - a melhor combinação, vista depois no test1, ficou abaixo da v4 (seção 8).
11. **Análise e teste final:** eli5 e análise de erros (seção 9), McNemar (seção 8) e rotulação do test2 pela v4.

**Como cada etapa foi conduzida:**

- **Semente única:** 123 em todas as divisões e modelos.
- **Avaliação:** validação cruzada estratificada de 5 dobras sobre as 20.092 respostas do `train.xlsx`; o baseline oficial (TF-IDF + regressão logística) é medido nas mesmas dobras.
- **Stacking sem vazamento:**
  - cada modelo-base gera previsões "fora da dobra" para todo o treino, e o meta-classificador aprende sobre essas previsões;
  - os modelos sobre TF-IDF, os embeddings do Qwen3 e o BERT 256 usam as mesmas 5 dobras;
  - o BERT 128 usa 2 dobras: treina em metade do treino e prevê a outra metade.
- **Rotulação do test2:**
  - os modelos sobre TF-IDF e embeddings são retreinados com todo o treino;
  - cada BERT prevê pela média dos logits dos modelos de cada dobra;
  - o meta-classificador é treinado sobre as previsões fora da dobra.
- **Custo e infraestrutura** (notebook com i5-1135G7 de 4 núcleos, 16 GB de RAM e GPU integrada Iris Xe):
  - o fine-tuning roda na CPU: o BERT 256 de 5 dobras levou cerca de 39 h, e o de 10 dobras (já com lotes agrupados), cerca de 58 h;
  - agrupamos os lotes por tamanho do texto, como o `group_by_length` do Hugging Face: lotes de textos de tamanho parecido precisam de menos preenchimento;
  - isso não muda o que o modelo aprende, mas acelera o BERT 256 de 1,16 para 1,72 textos/s (+48%), e com isso o BERT 512 custou o mesmo que o 256 sem agrupamento;
  - as previsões (validação, test1, test2) rodam na GPU integrada via OpenVINO em FP16, a 22 textos/s contra ~7 na CPU, e a CPU segue treinando a dobra seguinte;
  - conferimos a GPU com a CPU: diferença máxima de 0,003 nos logits e a mesma classe prevista;
  - os treinos gravam checkpoints a cada 20 min e retomam de onde pararam (o Windows reiniciou sozinho uma vez durante um treino).
- **Regras de decisão fixadas antes de ver os resultados:** para o TAPT, para a escolha entre BERT 256 e 512 e para a busca de combinações (seção 4). As decisões foram tomadas só com a validação cruzada no treino.
- **Uso dos conjuntos de teste:**
  - o test2 e o test1 entram sem rótulos apenas no vocabulário do TF-IDF e na inferência;
  - **o test2 não foi usado em nenhuma comparação**;
  - os rótulos do test1 foram usados somente no teste de McNemar, calculado para mais de um candidato ao longo do trabalho (seção 8).

## 7. Resultado

Acurácia em validação cruzada estratificada de 5 dobras (semente 123) sobre as 20.092 respostas do treino:

| Modelo | Dobra 1 | Dobra 2 | Dobra 3 | Dobra 4 | Dobra 5 | Média |
|---|---|---|---|---|---|---|
| Baseline oficial (TF-IDF + regressão logística) | 45,91% | 45,28% | 45,52% | 44,77% | 46,22% | 45,54% |
| BERTimbau 128, sozinho | 43,89% | 45,16% | 45,52% | 43,73% | 46,09% | 44,88% |
| BERTimbau 256, sozinho | 45,36% | 46,73% | 46,76% | 45,74% | 46,57% | 46,23% |
| Stacking do EP1 (4 blocos) | 46,68% | 47,23% | 47,39% | 45,79% | 47,54% | 46,92% |
| Stacking do EP2 v2 (EP1 + BERT 128) | 47,52% | 47,35% | 48,03% | 46,12% | 47,44% | 47,29% |
| Stacking do EP2 v3 (v2 + BERT 256) | 47,95% | 47,72% | 48,26% | 46,49% | 48,28% | 47,74% |
| **Stacking do EP2 v4 (v3 com Qwen3 no lugar do MiniLM)** | **48,62%** | **47,92%** | **48,08%** | **46,54%** | **48,41%** | **47,91%** |

- **Cálculo dos BERTs sozinhos:** acurácia (classe de maior logit) das previsões fora da dobra, recortadas nas mesmas 5 dobras das outras linhas. O BERT 128 vem de 2 dobras (44,66% e 45,09%).
- **Stacking do EP1 nesta tabela:** é o mesmo modelo do EP1, recalculado no ambiente do EP2. O vocabulário do TF-IDF agora inclui também os textos do test2, por isso o valor difere pouco dos 46,98% relatados no EP1.

**Resultado:**
- O stacking final (v4) supera o do EP1 em todas as 5 dobras (+0,99 ponto na média) e o baseline oficial em todas (+2,37 pontos na média).
- Cada versão melhorou a anterior na média. O maior salto veio do BERT 256 (+0,45 ponto, melhor nas 5 dobras).
- Distribuição prevista no test2 (`ep2/entrega/test2.xlsx`, 897 respostas): `c1` 309, `c234` 233, `c5` 355.

## 8. Evolução em relação ao EP1

Para a comparação, o modelo do EP2 (treinado só com o `train.xlsx`) rotulou o test1. Esses rótulos foram comparados com os entregues no EP1, usando os rótulos verdadeiros do test1 fornecidos pelo professor. O test2 não foi usado em nenhuma comparação.

| Estratégia | Teste | Acurácia |
|---|---|---|
| Modelo EP1 (entregue) | test1 (900 respostas) | 48,56% |
| **Modelo EP2 (v4)** | test1 (900 respostas) | **51,56%** |

Tabela de contingência:

| | EP2 acerta | EP2 erra |
|---|---|---|
| **EP1 acerta** | 393 | 44 |
| **EP1 erra** | 71 | 392 |

**Teste de McNemar** (qui-quadrado com correção de continuidade, 1 grau de liberdade): estatística χ² = (|44 − 71| − 1)² / (44 + 71) = **5,878**, **p = 0,0153**. Na versão exata (binomial), p = 0,0150.

Com p < 0,05, a vantagem do modelo do EP2 sobre o do EP1 é **estatisticamente significativa** no test1. Das 115 respostas em que os dois modelos discordam quanto ao acerto, o EP2 acerta 71 e o EP1, 44.

**Transparência sobre o uso do test1.** Os rótulos do test1 só entraram no McNemar, mas calculamos o teste para sete candidatos ao longo do trabalho:

| Modelo | Validação cruzada | Test1 | McNemar contra o EP1 |
|---|---|---|---|
| v2 | 47,29% | 50,44% | p = 0,047 |
| v3 | 47,74% | 51,00% | p = 0,036 |
| **v4 (entregue)** | **47,91%** | **51,56%** | **p = 0,015** |
| v4 com BERT 512 no lugar do 256 | 47,86% | 50,89% | p = 0,060 |
| v4 + BERT 512 | 47,87% | 50,67% | p = 0,102 |
| v2 + Qwen3 + BERT 512 | 47,88% | 50,33% | p = 0,149 |
| Melhor das 32.767 combinações (9 blocos; seção 4) | 48,12% | 50,11% | p = 0,235 |

Leituras:
- **Critério de escolha.** Todos os modelos foram escolhidos pela validação cruzada no treino. As três variantes com o BERT 512 foram avaliadas depois da escolha da v4 e não a mudaram.
- **A melhor das 32.767 combinações foi reprovada antes de ser vista no test1.** Ela tem a maior média na validação cruzada (48,12%), mas ganhou da v4 em só 3 das 5 dobras, e o procedimento de escolhê-la rendeu 47,60% na validação aninhada. Avaliada no test1 só depois, a pedido do grupo, ficou em 50,11%, abaixo da v4: a v4 acerta 20 respostas que ela erra, e ela acerta 7 que a v4 erra (p = 0,021). Isso ilustra o otimismo de escolher o máximo entre muitas configurações, que a regra fixada e a validação aninhada já indicavam.
- **A troca v3 → v4 não seguiu a regra fixada.** A regra da busca de combinações exigia ganho de 0,2 ponto, 4 das 5 dobras melhores e estimativa aninhada acima da v3, e mandava manter a v3: o ganho foi de 0,17 ponto, e a estimativa aninhada ficou em 47,60%. Adotamos a v4 por ter a maior média e ganhar em 4 das 5 dobras, já conhecendo o resultado das duas no test1.
- **A v4 não é significativamente melhor que a v3.** Na validação cruzada, p = 0,26 (481 × 446 respostas); no test1, p = 0,54.
- **Correção para múltiplas comparações.** Com Bonferroni para os três candidatos vistos no test1 antes da escolha da v4 (v2, v3, v4), o p da v4 fica em 0,046. Para os sete, fica em 0,107.
- **Conclusão.** A superioridade do EP2 sobre o EP1 é consistente: as três versões vão na mesma direção, todas com p < 0,05 sem correção, e a validação cruzada mostra ganho em todas as dobras. A ordem exata entre v3 e v4 está dentro do ruído.

## 9. Análise

Como pede o modelo de relatório, a interpretação usa o **eli5** (0.16.0) sobre um classificador simples baseado em palavras: regressão logística (`C=1`) sobre TF-IDF de 1–2-gramas, minúsculas, `min_df=20`. Esse classificador acerta 45,88% em validação cruzada (dobras: 45,68 / 46,35 / 45,45 / 45,32 / 46,62%), perto do baseline, e é muito mais legível que o stacking. Saídas completas em `ep2/resultados/analise/` (`global_eli5.html`, `local_eli5.html`, `erros_casos.txt`). Código: `ep2/codigo/03_analise.py`.

### (i) Interpretação global: atributos mais importantes por classe

| Classe | Termos com maior peso **a favor** | Termos com maior peso **contra** |
|---|---|---|
| `c1` (pouco clara) | solicitação, essa informação, processo, demanda, solicitação em, informações, anteriormente, que solicitação, inss | prezada, questionário, seguem, encaminhamos, anexo os, vaga |
| `c234` (intermediária) | sistemas, requisitos, de acompanhamento, em análise, documento anexo, referente sua, região, quanto | segue as, solicitação encaminhamos, dados para, solicitado |
| `c5` (muito clara) | questionário, olá, sim, documento solicitado, anexo os, solicitação encaminhamos, vaga disponível, ainda não, nenhum | essa informação, providências, demanda, levantamento, ainda que, informações |

Leituras:
- **`c1` é a resposta que fala *sobre o pedido* em vez de respondê-lo:** "solicitação", "demanda", "essa informação", "anteriormente" (remete a pedido anterior), "processo". Os mesmos termos são os mais fortes **contra** `c5`.
- **`c5` é a resposta direta:** saudação próxima ("olá"), "sim", "documento solicitado", "anexo os". Respostas negativas *objetivas* também são vistas como claras: "nenhum" e "ainda não" pesam a favor de `c5`. Ou seja, o cidadão premia a objetividade, não só o atendimento do pedido.
- **`c234` não tem vocabulário próprio:** os pesos são menores e espalhados ("em análise", "documento anexo", "quanto"). É coerente com ser a classe mais confundida (abaixo).
- **Atributos espúrios:** aparecem nomes próprios (`luis`, `sr luis`, `senhora odila`) e siglas de órgãos (`dnpm`, `inss`, `conab`, `proad`).
  - O modelo aprende *quem* respondeu: certos servidores e órgãos usam modelos de resposta que se repetem dezenas de vezes, sempre com avaliações parecidas.
  - A correlação existe nos dados, mas não é uma propriedade da clareza do texto e tende a não generalizar para órgãos que não estão no treino.
  - O `min_df=20` já remove os nomes raros, mas esses modelos de resposta se repetem mais que isso.

### (ii) Interpretação local: exemplos

Classificador treinado nas dobras 2–5, explicando respostas da dobra 1, que ele não viu. Entre parênteses, a contribuição do termo para a classe prevista.

| Caso | Real → previsto (prob.) | Trecho da resposta | Termos que mais pesaram |
|---|---|---|---|
| Acerto | `c1` → `c1` (0,67) | CAIXA: "não foi possível compreender o seu pedido… orientamos gerar novo pedido" | solicitação (+0,09), reclamações (+0,07), compreender (+0,07) |
| Acerto | `c5` → `c5` (0,68) | "Olá Dayanne, boa tarde! … esclarecemos que … vacância de cargos" | olá (+0,19), para cargo (+0,11), sim (+0,10) |
| Acerto | `c234` → `c234` (0,62) | Conab: "segue anexo resposta … poderá interpor recurso" | vários termos fracos (+0,04 cada): decreto 7724, conab, pela gerência |
| Erro | `c234` → `c1` (0,75) | IBGE: "os dados … estão em nosso banco SIDRA. Peço que realize sua pesquisa através do link …" | em nosso (+0,14), realize (+0,09), http www2 (+0,08) |
| Erro | `c5` → `c234` (0,63) | "Em anexo, encaminhamos Memorando nº 033/2017 - CCON." | memorando (+0,42), memorando nº (+0,25), anexo (+0,13) |
| Erro | `c234` → `c5` (0,73) | "A documentação solicitada segue em anexo. Conforme prevê o artigo 15 …" | documentação solicitada (+0,06), institucional (+0,08) |

Os três erros têm a mesma causa: **o conteúdo da resposta não está no texto, e sim num anexo ou num link.** O texto é só uma mensagem de encaminhamento, e o cidadão avaliou o que abriu.
- No caso do IBGE, o modelo trata "faça a pesquisa no link" como resposta evasiva, mas o cidadão deu nota intermediária.
- No memorando, o modelo não tem como saber que o anexo respondia bem.

Nos acertos, o modelo usa sinais com sentido: "não foi possível compreender" para `c1`; saudação pessoal e explicação para `c5`. O acerto `c234` sai de muitos sinais fracos, o que mostra de novo a falta de identidade lexical dessa classe.

### (iii) Análise manual de erros do modelo final

Matriz de confusão do stacking final (v4), com previsões fora da dobra para as 20.092 respostas do treino:

| Real \ previsto | `c1` | `c234` | `c5` | Revocação |
|---|---|---|---|---|
| `c1` | **3.367** | 1.565 | 1.415 | 53,0% |
| `c234` | 2.138 | **2.259** | 2.456 | 33,0% |
| `c5` | 1.270 | 1.621 | **4.001** | 58,1% |

- **`c234` é a classe mais difícil** (33% de revocação). Ela junta as notas 2, 3 e 4 e fica "entre" as outras duas, sem vocabulário próprio.
- **26% dos erros são trocas entre os extremos** (`c1` ↔ `c5`: 2.685 respostas). A mesma resposta pode ser "muito clara" para um cidadão e "nada clara" para outro.
- **Ruído de rótulo medido diretamente:** 1.935 respostas do treino (9,6%) têm texto **idêntico** ao de outra resposta, mas rótulo diferente. Nenhum modelo que lê só a resposta consegue acertar todas. Nelas, o modelo acerta 42,7%; nas demais, 48,5%.

Lemos 13 erros sorteados (semente 123, sem textos repetidos; lista em `erros_casos.txt`). Eles se agrupam em três padrões:

1. **Resposta no anexo:** "segue resposta em anexo", "segue, em anexo, a resposta…", "arquivo contendo resposta foi inserido no e-SIC".
   - São textos quase iguais, com rótulos `c1`, `c234` e `c234`, e o modelo previu `c5` nos três.
   - O modelo aprendeu que "segue em anexo" costuma acompanhar a entrega do que foi pedido. Mas o rótulo depende do conteúdo do anexo, que não está nos dados.
2. **Negativas e recusas justificadas, com qualquer rótulo:**
   - "o pedido não é solicitação de informação, é um questionário de opinião" (`c234`, previsto `c1`);
   - "a LAI não ampara consultas; procure os canais do INSS" (`c1`, previsto `c234`);
   - "solicitamos que esclareça o ano e abra uma nova demanda" (`c1`, previsto `c234`);
   - "a secretaria não faz esse levantamento" (`c234`, previsto `c1`);
   - "gere o extrato previdenciário pelo link …" (`c5`, previsto `c1`).

   É o mesmo tipo de resposta, com os três rótulos. O que decide a nota é se o cidadão aceitou a justificativa ou se o caminho indicado resolveu o problema, e isso não aparece no texto.
3. **Respostas objetivas com avaliação diferente da esperada:**
   - Convênio da PRF com a PM de Minas, com datas e prazos: `c1`, previsto `c5`. Provavelmente não era o que o cidadão perguntou.
   - Ifes citando acórdãos do TCU para negar um aproveitamento de concurso: `c234`, previsto `c5`.
   - CODESP liberando documento "de forma parcial", com vistas por agendamento: `c1`, previsto `c5`. O modelo vê sinais de entrega; o cidadão avaliou a parte negada.
   - No sentido contrário, duas respostas `c5` previstas como `c1`:
     - a CAIXA, cujos dados (números de processos) vêm cercados de texto padrão do SAC e da Ouvidoria;
     - uma explicação da legislação de provimento de cargos que começa com "acerca da demanda", termo que pesa a favor de `c1`.

**Conclusão da análise.**
- O rótulo mede a percepção do cidadão sobre a resposta *em relação ao que ele perguntou*, e muitas vezes depende de anexos e links que não estão no texto.
- Com apenas a resposta, há um teto de acerto, e o ruído dos textos idênticos com rótulos diferentes mostra que ele é baixo.
- Os BERTs ajudam porque leem mais contexto: o de 256 subpalavras vê a resposta inteira em 64% dos casos. Mesmo assim, não resolvem os três padrões acima.
- O ganho mais promissor para trabalhos futuros seria dispor também do **texto do pedido**, para modelar a adequação da resposta à pergunta.

## 10. Repositório e reprodução

Código: https://github.com/digogc/ep1-pln-clareza (pasta `ep2/`; o modelo final está em `ep2/codigo/` e os experimentos da seção 4 em `ep2/experimentos/`).

Passo a passo, a partir da raiz do repositório (detalhes em `ep2/README.md`):

1. Copiar `train.xlsx`, `test2.xlsx` e `test1_rotulado.xlsx` para `ep2/dados/`.
2. Instalar as dependências: `pip install -r modelo_final/requirements.txt` e `pip install eli5==0.16.0 transformers`. Versões usadas: torch 2.14, transformers 5.17, sentence-transformers 6.1, scikit-learn 1.7.
3. (Opcional, dezenas de horas em CPU) Refazer os modelos neurais. As previsões deles já estão versionadas em `ep2/resultados/`:
   - BERT 128: `python ep2/codigo/01_bert_finetune.py` (~6 h) gera os arquivos `bert_*`;
   - BERT 256: `python ep2/experimentos/codigo/e02_bert.py --nome bert256_5d --max-len 256 --dobras 5 --threads 8` (~39 h; com `--agrupar`, ~26 h, mas a ordem dos lotes muda e os números saem um pouco diferentes). Depois, copiar `bert256_5d_oof.npy`, `bert256_5d_test1.npy` e `bert256_5d_test2.npy` de `ep2/experimentos/resultados/` para `ep2/resultados/`, sem o `_5d` no nome;
   - Qwen3: `python ep2/experimentos/codigo/e04_embeddings.py` gera o `bloco_emb_qwen3.npz`, que deve ser copiado para `ep2/resultados/bloco_qwen3.npz`.
4. `python ep2/codigo/02_stacking_ep2.py`: mostra a acurácia por dobra de todas as versões e grava `ep2/entrega/test2.xlsx`.
5. `python ep2/codigo/03_analise.py`: gera a análise da seção 9 em `ep2/resultados/analise/`.
6. `python ep2/codigo/04_mcnemar.py`: roda o teste de McNemar da seção 8.

As versões anteriores do modelo (código, planilha do test2 e resultados) estão em `ep2/versoes/`.
