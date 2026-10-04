# EP1 — ACH2118 Introdução ao Processamento de Língua Natural

**Professor:** Ivandré Paraboni (EACH-USP)
**Resumo feito a partir de:** `ep1-enunciado.pdf`, `ep1-modelo-relatorio.pdf`, `train.xlsx` e `test1.xlsx`

> ⚠️ **Prazo de entrega: domingo, 04/10** (o enunciado não informa horário).
> Apresentações em aula: **06/10 e 13/10**.

---

## 1. A tarefa em uma frase

Treinar um classificador que diga quão **clara** é uma resposta dada a um cidadão no e-SIC (pedidos de acesso à informação), em 3 classes — `c1`, `c234`, `c5` — e usar esse modelo para rotular o conjunto de teste.

As classes vêm de notas de 1 a 5 dadas pelos próprios usuários: `c1` = nota 1, `c5` = nota 5 e `c234` = notas intermediárias agrupadas.

## 2. O que precisa ser feito

1. Desenvolver um modelo **melhor que o baseline oficial** (regressão logística + TF-IDF).
2. Usar o modelo para preencher a coluna `clarity` do `test1.xlsx`.
3. Escrever o relatório seguindo o modelo do professor.
4. Preparar a apresentação (até 10 min) em PDF.
5. Juntar tudo em um `.zip` e entregar (um único integrante entrega pelo grupo).

## 3. O que entregar (pasta `.zip`)

| # | Item | Formato | Observação |
|---|------|---------|------------|
| 1 | Relatório | conforme `ep1-modelo-relatorio.pdf` | precisa conter o link do código |
| 2 | Apresentação | **PDF** (não ppt/pptx) | até 10 minutos |
| 3 | Conjunto de teste rotulado | **.xlsx** | mesmo formato do `test1.xlsx` |

### Cuidados com a planilha de teste (erro aqui = −20%)

- Manter exatamente **900 linhas** de dados e **2 colunas** (`resp_text`, `clarity`).
- Não reordenar as linhas nem alterar os textos.
- Rótulos escritos exatamente como no treino: `c1`, `c234`, `c5`.
- Salvar como `.xlsx` (não csv).

## 4. Como a nota é calculada

| Critério | Peso |
|----------|------|
| Acurácia média no teste | 70% |
| Método inovador / bem elaborado | 20% |
| Relatório | 5% |
| Apresentação em aula | 5% |

- A nota-base é **normalizada pelo ranking da turma** (o melhor resultado vale 10).
- O ranking define a ordem das apresentações, **começando pelo último colocado**.
- **Não pontuam em "inovação":** TF-IDF/BoW com regressão logística, SVM, Naive Bayes e afins.

### Penalidades

- **Acurácia abaixo do baseline de classe majoritária:** nota 3,0, trabalho não é avaliado e não concorre em inovação.
- **−20% cumulativo** para cada item:
  - resultado inferior ao baseline oficial (regressão logística + TF-IDF);
  - não entregar relatório ou código no prazo;
  - não entregar o PDF da apresentação;
  - arquivo de teste em formato inválido (linhas/colunas a mais ou a menos, rótulos diferentes ou fora de posição, arquivo que não seja xlsx).
- Faltar na hora da apresentação zera só a nota da apresentação.

### Bônus

- **+10% na nota final** para cada integrante que desistiu/sumiu, se isso for indicado no relatório.

## 5. Estrutura do relatório (modelo do professor)

1. **Nome e número USP** de todos os integrantes efetivos.
2. **Parágrafo introdutório:** estratégia principal, tipo de representação textual (palavras, caracteres, embeddings…) e tipo de classificador. Pode descrever mais de um modelo, desde que o **modelo final** fique claramente indicado.
3. **Pré-processamento:** como os textos foram tratados.
4. **Parâmetros avaliados na busca em grade:** tabela com parâmetro e faixa de valores.
5. **Valores ótimos** encontrados para cada parâmetro.
6. **Procedimento:** como foi a separação treino/validação e demais passos para reproduzir.
7. **Resultado:** tabela com a métrica (acurácia) **apenas do modelo final**.
8. **Link do repositório** de código (GitHub, Colab ou similar).
9. **Instruções de reprodução:** passo a passo, bibliotecas e dependências.

## 6. O que encontrei nos dados

| | `train.xlsx` | `test1.xlsx` |
|---|---|---|
| Linhas | 20.092 | 900 |
| Colunas | `resp_text`, `clarity` | `resp_text`, `clarity` (vazia) |
| Tamanho mediano do texto | 101 palavras | 101 palavras |
| Texto mais longo | 1.821 palavras | 930 palavras |

- **Classes balanceadas no treino:** `c5` 6.892 · `c234` 6.853 · `c1` 6.347.
- **Sem valores nulos.**
- **Textos repetidos no treino:** 1.660 linhas são repetições. Dos 463 textos que se repetem, **291 aparecem com rótulos diferentes** — ou seja, os rótulos são ruidosos (a mesma resposta recebeu notas diferentes de usuários diferentes). Isso limita a acurácia máxima possível.
- **88 das 900 respostas de teste** têm texto idêntico a alguma resposta do treino.

### Referências que medi (validação cruzada 5-fold no treino)

| Modelo | Acurácia |
|--------|----------|
| Classe majoritária | ~34,3% |
| TF-IDF + regressão logística (parâmetros padrão do scikit-learn) | ~45,4% |

A configuração exata do baseline do professor não está no enunciado, então trate os ~45% como aproximação da barra a superar. É um problema difícil: não espere acurácias altas. Nos experimentos finais (ver `RESULTADOS.md`), com a divisão de dobras definitiva, esse baseline mediu 44,56% e o modelo final chegou a 46,37%. Na versão 2 (semente 123, protocolo do baseline oficial), o baseline mede 45,54% em validação cruzada e 46,75% no holdout, e o modelo final 46,80% e 47,50%.

## 7. Recomendações do enunciado

- **Começar simples:** garantir um resultado rápido antes de partir para algo sofisticado.
- Usar **validação cruzada** e **grid search** para evitar overfitting — uma queda grande entre treino e teste "não é um resultado real".
- Não subestimar o tempo de treino.
- Montar um pipeline de experimentação reaproveitável (**haverá EP2**).
- Conferir a entrega antes de enviar.

## 8. Sugestão de caminho (minha, não está no enunciado)

1. **Reproduzir o baseline** TF-IDF + regressão logística para ter a barra medida.
2. **Modelo principal que conte como inovação**, por exemplo:
   - fine-tuning de um BERT em português (ex.: BERTimbau);
   - embeddings de sentença + classificador (regressão logística, MLP ou gradient boosting);
   - combinação (ensemble) de embeddings com atributos manuais: tamanho da resposta, presença de anexos/links, citações de leis, termos de negativa ("não é possível", "sigilo") etc.
3. **Validação cruzada agrupando textos repetidos** (ex.: `StratifiedGroupKFold`), para a mesma resposta não cair em treino e validação ao mesmo tempo e inflar o resultado.
4. Anotar os parâmetros testados e os melhores valores **enquanto roda** — o relatório pede essas tabelas.
5. Gerar o `test1.xlsx` rotulado e conferir o formato (seção 3).
6. Relatório → slides em PDF → `.zip`.

Atenção ao limite de 512 tokens dos modelos tipo BERT: cerca de 8% das respostas do treino têm mais de 350 palavras (que é mais ou menos onde esse limite cai em português) e serão truncadas.

## 9. Preciso usar o Claude Code?

**Não.** O enunciado não exige nem menciona nenhuma ferramenta específica — nem Claude Code, nem qualquer outra IA. O que ele exige é o resultado: modelo, planilha rotulada, relatório com link do código e apresentação.

O que você realmente precisa:

- **Python** com `pandas`, `scikit-learn` e, se for usar transformers, `torch` + `transformers`.
- **GPU** se for fazer fine-tuning de BERT (Google Colab resolve; em CPU fica inviável no prazo).
- Um **repositório** (GitHub ou Colab) para colocar o link no relatório.

O Claude Code é uma opção útil, não um requisito: ajuda a montar o repositório, o pipeline de experimentos e os scripts de treino/predição mais rápido. Para um experimento único, um notebook no Colab dá conta sozinho.

Dois pontos a confirmar por fora, porque o enunciado não diz:

- se a disciplina tem alguma **regra sobre uso de IA** nos EPs;
- se existe **mais de um conjunto de teste** — o enunciado fala em "acurácia média" e o arquivo se chama `test1.xlsx`, mas só há esse na pasta.

## 10. Checklist final de entrega

- [ ] Acurácia em validação cruzada acima do baseline TF-IDF + regressão logística
- [ ] `test1.xlsx` rotulado: 900 linhas, 2 colunas, rótulos `c1`/`c234`/`c5`, formato xlsx
- [ ] Relatório com os 9 itens do modelo
- [ ] Nome e número USP de todos os integrantes (e desistências, se houver)
- [ ] Link do repositório funcionando e código reproduzível
- [ ] Apresentação em PDF, até 10 minutos
- [ ] Tudo em um único `.zip`, entregue por um integrante até 04/10
