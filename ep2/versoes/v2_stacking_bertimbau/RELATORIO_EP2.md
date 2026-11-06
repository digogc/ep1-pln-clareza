# ACH2118 Introdução ao Processamento de Língua Natural — Relatório do EP2

## 1. Integrantes

| Nome | Número USP |
|---|---|
| Dimitri Prado | 14746022 |
| Lucas de Lyra Monteiro | 15471435 |
| Rodrigo Gonçalves Cardoso | 14658330 |
| Rodrigo Lyusei Suguimoto | 14670877 |

## 2. Estratégia

A tarefa é a mesma do EP1: classificar a clareza de respostas do e-SIC em {`c1`, `c234`, `c5`}. A evolução em relação ao EP1 é o **fine-tuning de um BERT em português (BERTimbau base, `neuralmind/bert-base-portuguese-cased`)**, avaliado sozinho e incorporado como quinto modelo-base do stacking do EP1.

**O modelo final é um stacking de cinco modelos-base**, combinados por um meta-classificador de regressão logística:

1. `lr_pal`: regressão logística sobre TF-IDF de palavras (1–2-gramas);
2. `svc`: SVM linear sobre TF-IDF de palavras (1–2-gramas) e de caracteres (2–5-gramas);
3. `emb`: regressão logística sobre embeddings de sentença do `paraphrase-multilingual-MiniLM-L12-v2` (congelado);
4. `extremos`: regressão logística `c1` × `c5` treinada apenas nos extremos (eixo "ruim–bom");
5. `bert`: **BERTimbau ajustado (fine-tuning) para a tarefa**, que lê a resposta como sequência de subpalavras com atenção contextual.

No stacking, o BERT entra resumido num único escore ordinal, `logit(c5) − logit(c1)`, em vez dos três logits. É o mesmo eixo "ruim → bom" do bloco `extremos`, e essa forma foi a melhor na validação cruzada (seção 4).

**Por que o BERT ajuda, mesmo sendo mais fraco sozinho.** Sozinho, o BERTimbau acerta 44,9%, menos que o baseline (45,5%). Ele lê a resposta de outro jeito (subpalavras, ordem e contexto) e erra em respostas diferentes das que os modelos de TF-IDF erram. No stacking, isso vira informação complementar: o EP2 passa a acertar 724 respostas do treino que o EP1 errava e passa a errar 650 que o EP1 acertava.

**Dados.** Usamos apenas os arquivos fornecidos: o treino é o mesmo do EP1 (`train.xlsx`), como pede o enunciado, e nenhum dado externo foi usado. Nenhum dado dos conjuntos de teste é usado com rótulo. Cerca de 11% dos textos do test1 (100) e do test2 (106) são idênticos a algum texto do treino. São modelos padrão de resposta dos órgãos (mediana de 13 cópias no treino por texto), ou seja, pedidos diferentes respondidos com o mesmo texto, e não as mesmas linhas.

## 3. Pré-processamento

- **Todos os modelos:** desfaz os escapes de caracteres de controle do Excel presentes nos arquivos do EP2 (por exemplo `_x0096_`), o que torna os textos idênticos aos do EP1, e junta espaços e quebras de linha repetidos. Sem remoção de stopwords, stemming ou lematização.
- **BERTimbau:** tokenizador WordPiece do próprio modelo (mantém maiúsculas e acentos); truncamento em 128 subpalavras com **cabeça + cauda** (as 63 primeiras e as 63 últimas), porque o início traz a saudação e o assunto e o fim traz a conclusão, o encaminhamento e a assinatura do órgão.
- **TF-IDF:** minúsculas, `sublinear_tf=True`, `min_df=2` (palavras) e `min_df=3` (caracteres `char_wb`); vocabulário e IDF calculados sobre treino + test1 + test2, sem rótulos.
- **Embeddings de sentença:** truncamento em 256 subpalavras, vetores normalizados e padronizados.

## 4. Parâmetros avaliados na busca em grade

**BERTimbau (fine-tuning)**

| Parâmetro | Valores |
|---|---|
| Modelo | {BERTimbau base, multilingual-e5-small} (velocidade de treino medida em CPU: 1,5 e 3,5 textos/s) |
| Comprimento máximo | {128, 256} subpalavras (256 custa ~1,4 vez mais) |
| Camadas congeladas | {0, 4, 6, 8} (além dos embeddings) |
| Taxa de aprendizado | 3e-5, com aquecimento linear em 10% dos passos e decaimento linear |
| Épocas | {1, 2} (acompanhadas na metade de validação ao fim de cada época) |
| Lote | 16 |
| Entrada do BERT no stacking | {3 logits, log-softmax, 3 logits padronizados por metade, escore `logit(c5) − logit(c1)`}: 47,06 / 47,05 / 47,15 / **47,29%** em validação cruzada |
| `C` do meta-classificador com o BERT | {0,01; 0,1; 1}: 47,00 / **47,06** / 47,10% (com os 3 logits; diferença desprezível, mantido 0,1 do EP1) |

Cada fine-tuning completo leva cerca de 3 h em CPU, por isso a grade do BERT não foi uma busca exaustiva por acurácia. Modelo, comprimento e camadas congeladas foram comparados pelo custo medido em treinos curtos. Só a configuração escolhida foi treinada até o fim, nas duas dobras; a justificativa está na seção 5.

**Stacking e demais modelos-base** (herdados do EP1)

| Parâmetro | Valores |
|---|---|
| Representação lexical | {palavras 1-gram, palavras 1–2-gram, caracteres 2–5-gram, palavras + caracteres} |
| Classificador lexical | {regressão logística, LinearSVC, Complement Naive Bayes} |
| `C` (regressão logística / LinearSVC) | {0,3; 1; 3} / {0,01; 0,03; 0,1; 0,3; 1} |
| Embeddings de sentença | {MiniLM, e5-small, os dois, concatenados}, `C` ∈ {0,003; 0,01; 0,03; 0,1} |
| Blocos do stacking | seleção passo a passo entre os candidatos |
| `C` do meta-classificador | {0,01; 0,1; 1} |

## 5. Valores ótimos

| Componente | Parâmetro | Valor |
|---|---|---|
| `bert` | Modelo | `neuralmind/bert-base-portuguese-cased` |
| | Comprimento máximo / truncamento | 128 subpalavras, cabeça + cauda |
| | Camadas congeladas | embeddings + 4 camadas inferiores (8 camadas ajustadas) |
| | Taxa de aprendizado / épocas / lote | 3e-5 / 2 / 16 |
| | Entrada no stacking | escore `logit(c5) − logit(c1)` |
| `lr_pal` | Representação / `C` | palavras 1–2-gram / 1 |
| `svc` | Representação / `C` | palavras 1–2-gram + caracteres 2–5-gram / 0,1 |
| `emb` | Modelo / `C` | paraphrase-multilingual-MiniLM-L12-v2 / 0,01 |
| `extremos` | Representação / `C` | palavras + caracteres / 1 |
| Meta-classificador | `C` | 0,1 |

**Justificativa das escolhas do BERT:**
- **Modelo:** o BERTimbau foi pré-treinado em português brasileiro. O `multilingual-e5-small` seria cerca de 2 vezes mais rápido, mas o objetivo do EP2 era justamente um BERT ajustado para o português.
- **Comprimento:** 128 subpalavras com cabeça + cauda cabem no orçamento de duas dobras × duas épocas (~6 h); 256 subpalavras levaria ~8 h.
- **Camadas congeladas:** congelar os embeddings e as 4 camadas inferiores funciona como regularização para um conjunto pequeno e com rótulos ruidosos; o ganho de velocidade medido foi pequeno.
- **Épocas:** 2. Na amostra de 600 respostas de validação, a dobra 0 foi de 44,8% para 41,8% da 1ª para a 2ª época, e a dobra 1 de 42,8% para 45,3%. O resultado é inconclusivo, e a amostra é pequena (±2 pontos), por isso ficaram as 2 épocas planejadas.

## 6. Procedimento

- **Semente única:** 123 em todas as divisões e modelos.
- **Avaliação:** validação cruzada estratificada de 5 dobras sobre as 20.092 respostas do `train.xlsx`; o baseline oficial (TF-IDF + regressão logística) é medido nas mesmas dobras.
- **Stacking sem vazamento:** cada modelo-base gera previsões "fora da dobra" para todo o treino. Os quatro modelos do EP1 usam 5 dobras. O BERTimbau usa **2 dobras** (treina em metade do treino e prevê a outra metade), porque cada fine-tuning em CPU leva horas; as previsões do BERT para os conjuntos de teste são a média dos dois modelos. O meta-classificador aprende sobre essas previsões.
- **Rotulação do test2:** modelos-base do EP1 retreinados com todo o treino, BERT pela média das duas dobras, meta-classificador treinado sobre as previsões fora da dobra.
- **Uso dos conjuntos de teste:** o test2 e o test1 entram sem rótulos apenas no vocabulário do TF-IDF e na inferência. Os rótulos do test1 são usados **somente** no teste de McNemar (seção 8); nenhuma escolha de modelo foi feita com o test1 ou o test2.

## 7. Resultado

Acurácia em validação cruzada estratificada de 5 dobras (semente 123) sobre as 20.092 respostas do treino:

| Modelo | Dobra 1 | Dobra 2 | Dobra 3 | Dobra 4 | Dobra 5 | Média |
|---|---|---|---|---|---|---|
| Baseline oficial (TF-IDF + regressão logística) | 45,91% | 45,28% | 45,52% | 44,77% | 46,22% | 45,54% |
| BERTimbau ajustado, sozinho | 43,89% | 45,16% | 45,52% | 43,73% | 46,09% | 44,88% |
| Stacking do EP1 (4 blocos) | 46,68% | 47,23% | 47,39% | 45,79% | 47,54% | 46,92% |
| **Stacking do EP2 (4 blocos + BERTimbau)** | **47,52%** | **47,35%** | **48,03%** | **46,12%** | 47,44% | **47,29%** |

- **Cálculo do BERT sozinho:** acurácia das previsões fora da dobra (2 dobras: 44,66% e 45,09%), recortadas nas mesmas 5 dobras das outras linhas para a tabela ficar comparável.
- **Stacking do EP1 nesta tabela:** é o mesmo modelo do EP1, recalculado no ambiente do EP2. O vocabulário do TF-IDF agora inclui também os textos do test2, por isso o valor difere pouco dos 46,98% relatados no EP1.

**Resultado:**
- O stacking do EP2 supera o do EP1 em 4 das 5 dobras (+0,37 ponto na média) e o baseline oficial em todas (+1,75 ponto na média).
- O BERTimbau sozinho fica abaixo do baseline, mas no stacking é o bloco que traz o ganho.
- Distribuição prevista no test2 (`ep2/entrega/test2.xlsx`, 897 respostas): `c1` 310, `c234` 231, `c5` 356.

## 8. Evolução em relação ao EP1

Para a comparação, o modelo do EP2 (treinado só com o `train.xlsx`) rotulou o test1. Esses rótulos foram comparados com os entregues no EP1, usando os rótulos verdadeiros do test1 fornecidos pelo professor. Esses rótulos só foram lidos depois que o modelo estava escolhido, e nenhuma decisão foi tomada com eles. O test2 não foi usado em nenhuma comparação.

| Test1 (900 respostas) | Acurácia |
|---|---|
| EP1 (entregue) | 48,56% |
| **EP2** | **50,44%** |

Tabela de contingência:

| | EP2 acerta | EP2 erra |
|---|---|---|
| **EP1 acerta** | 413 | 24 |
| **EP1 erra** | 41 | 422 |

**Teste de McNemar** (qui-quadrado com correção de continuidade, 1 grau de liberdade): estatística χ² = (|24 − 41| − 1)² / (24 + 41) = **3,938**, **p = 0,047**. Na versão exata (binomial), p = 0,046.

Com p < 0,05, o modelo do EP2 é **significativamente melhor** que o do EP1 no test1. Das 65 respostas em que os dois modelos discordam quanto ao acerto, o EP2 acerta 41 e o EP1, 24.

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
- **Atributos espúrios:** aparecem nomes próprios (`luis`, `sr luis`, `senhora odila`) e siglas de órgãos (`dnpm`, `inss`, `conab`, `proad`). O modelo aprende *quem* respondeu: certos servidores e órgãos usam modelos de resposta que se repetem dezenas de vezes, sempre com avaliações parecidas. A correlação existe nos dados, mas não é uma propriedade da clareza do texto e tende a não generalizar para órgãos que não estão no treino. O `min_df=20` já remove os nomes raros, mas esses modelos de resposta se repetem mais que isso.

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

Matriz de confusão do stacking do EP2, com previsões fora da dobra para as 20.092 respostas do treino:

| Real \ previsto | `c1` | `c234` | `c5` | Revocação |
|---|---|---|---|---|
| `c1` | **3.284** | 1.572 | 1.491 | 51,7% |
| `c234` | 2.131 | **2.203** | 2.519 | 32,1% |
| `c5` | 1.271 | 1.606 | **4.015** | 58,3% |

- **`c234` é a classe mais difícil** (32% de revocação). Ela junta as notas 2, 3 e 4 e fica "entre" as outras duas, sem vocabulário próprio.
- **26% dos erros são trocas entre os extremos** (`c1` ↔ `c5`: 2.762 respostas). A mesma resposta pode ser "muito clara" para um cidadão e "nada clara" para outro.
- **Ruído de rótulo medido diretamente:** 1.935 respostas do treino (9,6%) têm texto **idêntico** ao de outra resposta, mas rótulo diferente. Nenhum modelo que lê só a resposta consegue acertar todas. Nelas, o modelo acerta 42,1%; nas demais, 47,8%.

Lemos 13 erros sorteados (semente 123, sem textos repetidos; lista em `erros_casos.txt`) e eles se agrupam em três padrões:

1. **Encaminhamentos ("procure outro órgão") com qualquer rótulo:** a mesma estrutura de resposta aparece com os três rótulos.
   - Petrobras informando que não houve pedido de informação, só de cancelamento; Suframa indicando o e-mail da Superintendência; SUSEP indicando o formulário de vistas. Todos são **`c5`** e o modelo previu `c1`.
   - MEC indicando a Secretaria de Educação de SP é **`c1`**, e ANEEL indicando o ONS é **`c234`**.

   O modelo aprendeu que "encaminhar" é pouco claro, o que vale na média. Mas, quando o encaminhamento é preciso (link, e-mail, passo a passo), muitos cidadãos o julgam claro.
2. **Respostas objetivas com dados avaliadas como intermediárias:** tabela de egressos de doutorado; quadro de funcionários do Serpro, com números; quantidade de terceirizados de um cargo numa cidade. São **`c234`** e o modelo previu `c5`. O texto parece completo, e o rótulo provavelmente reflete se os dados eram *os que o cidadão queria*, o que só se saberia lendo o pedido.
3. **Entregas parciais com ressalva avaliadas como pouco claras:**
   - "o Plano Diretor encontra-se sob apreciação… contudo anexamos cópia";
   - planilha enviada por link, "contudo, o e-mail dos alunos é informação pessoal e não pode ser divulgado";
   - pedido redirecionado para atendimento presencial.

   São **`c1`** e o modelo previu `c5`. O modelo vê sinais de entrega ("anexamos", "planilha", link); o cidadão avaliou a parte negada.

**Conclusão da análise.** O rótulo mede a percepção do cidadão sobre a resposta *em relação ao que ele perguntou*, e muitas vezes depende de anexos e links que não estão no texto. Com apenas a resposta, há um teto de acerto, e o ruído dos textos idênticos com rótulos diferentes mostra que ele é baixo. O ganho mais promissor para trabalhos futuros seria dispor também do **texto do pedido**, para modelar a adequação da resposta à pergunta.

## 10. Repositório e reprodução

Código: https://github.com/digogc/ep1-pln-clareza (pasta `ep2/`).

Passo a passo, a partir da raiz do repositório (detalhes em `ep2/README.md`):

1. Copiar `train.xlsx`, `test2.xlsx` e `test1_rotulado.xlsx` para `ep2/dados/`.
2. `pip install -r modelo_final/requirements.txt` e `pip install eli5==0.16.0`.
3. (Opcional, cerca de 6 h em CPU) `python ep2/codigo/01_bert_finetune.py`, que refaz o fine-tuning do BERTimbau. As previsões dele já estão versionadas em `ep2/resultados/bert_*.npy`.
4. `python ep2/codigo/02_stacking_ep2.py`: mostra a acurácia por dobra e grava `ep2/entrega/test2.xlsx` (cerca de 7 min).
5. `python ep2/codigo/03_analise.py`: gera a análise da seção 9 em `ep2/resultados/analise/`.
6. `python ep2/codigo/04_mcnemar.py`: roda o teste de McNemar da seção 8.
