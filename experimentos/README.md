# Experimentos (histórico)

Nada desta pasta é necessário para reproduzir o resultado final; o modelo final está em [`../modelo_final/`](../modelo_final/). Aqui ficam os testes feitos no caminho, com o que cada um mostrou.

Para rodar estes scripts, copie `train.xlsx` e `test1.xlsx` para dentro de `experimentos/` e execute a partir de `experimentos/codigo/`.

## Índice dos testes

Acurácias em validação cruzada de 5 dobras. As duas fases usam divisões diferentes e não são comparáveis entre si.

### Fase 1 — dobras que agrupam textos idênticos (baseline nessa divisão: 44,56%)

| Script | O que testa | Resultado | Entrou no modelo final? |
|---|---|---|---|
| `01_tfidf.py` | grade de TF-IDF (palavras, caracteres) × regressão logística, SVM, Naive Bayes | melhor: 45,66% | a representação e o SVM, sim |
| `02_atributos_lgbm.py`, `atributos.py` | 87 atributos manuais (tamanho, legibilidade, marcadores) + LightGBM | 42,79% | não |
| `03_embeddings.py` | embeddings estáticos (spaCy, Word2Vec, Doc2Vec, LSA) + classificador | melhor: 44,65% | não |
| `03_textcnn.py` | rede convolucional sobre palavras | 44,94% | não |
| `04_ensemble.py` | primeiro stacking (TF-IDF + atributos + memória de respostas repetidas) | 46,37% | substituído |

### Fase 2 — dobras estratificadas comuns, semente 123 (baseline oficial: 45,54%)

| Script | O que testa | Resultado | Entrou no modelo final? |
|---|---|---|---|
| `00_baseline.py` | baseline oficial sob vários protocolos (explica por que o número muda de 44,6% para ~46%) | — | referência |
| `05_embeddings_sentenca.py` | extração de embeddings de sentença (MiniLM, e5-small) | — | sim (MiniLM) |
| `06_vizinhos.py` | "memória difusa": rótulos das respostas mais parecidas | sem ganho depois de corrigido um vazamento | não |
| `pipeline.py`, `07_avaliar.py` | stacking com seleção passo a passo dos blocos, comparado ao baseline | 46,5% a 46,8% | base do modelo final |
| `08_final.py` | versão de desenvolvimento do modelo final | 46,98% | sim (reescrito em `../modelo_final/`) |
| `09_so_tfidf_svm.py` | só TF-IDF + SVM, com grade de `C` | melhor: 46,45% | não |
| `10_comparar_versoes.py` | SVM só de palavras no lugar do SVM de palavras + caracteres | 46,65% | não |
| `11_melhorias_rapidas.py`, `12_minilm_holdout.py` | outros valores de `C`; MiniLM no lugar do e5-small | MiniLM: 46,98% | sim (MiniLM) |
| `15_busca_final.py` | 11 variações em torno do modelo final | todas entre 46,8% e 47,0% | não |
| `13_bert_dimitri_inferencia.py`, `14_stacking_com_bert.py` | juntar um BERTimbau ajustado ao stacking | não concluído (inferência em CPU interrompida por tempo) | não |

## Pastas

| Pasta | Conteúdo |
|---|---|
| `codigo/` | os scripts acima |
| `resultados/` | tabelas (`*_grid.csv`) e logs de cada script |
| `predicoes/` | planilhas de teste rotuladas por versões anteriores (não são a entrega) |
| `dados/` | divisão de dobras da fase 1 e caches |
| `RESULTADOS.md` | relato detalhado de todos os experimentos |
| `RESUMO-EP1.md` | resumo do enunciado |
