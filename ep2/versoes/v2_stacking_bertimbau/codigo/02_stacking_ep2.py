"""EP2 - stacking do EP1 + BERTimbau ajustado (01_bert_finetune.py) como quinto modelo-base.

Modelos-base (previsoes "fora da dobra", 5 dobras estratificadas, semente 123):
    lr_pal    regressao logistica sobre TF-IDF de palavras (1-2-gramas)
    svc       SVM linear sobre TF-IDF de palavras (1-2) + caracteres (2-5)
    emb       regressao logistica sobre embeddings do MiniLM (congelado)
    extremos  regressao logistica c1 x c5 treinada so nos extremos
    bert      BERTimbau ajustado (previsoes fora da dobra de 2 dobras, vindas de 01_bert_finetune.py),
              resumido no escore logit(c5) - logit(c1)
Meta-classificador: regressao logistica (C=0,1) sobre as saidas padronizadas.

Mostra, nas mesmas 5 dobras: baseline oficial, BERTimbau sozinho, stacking do EP1 (4 blocos) e
stacking do EP2 (5 blocos). Grava o test2 rotulado em ep2/entrega/test2.xlsx e as previsoes do
modelo EP2 para o test1 em ep2/resultados/test1_ep2.xlsx (so para o teste de McNemar).
Uso:  python ep2/codigo/02_stacking_ep2.py
"""
import time
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from openpyxl import load_workbook
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from dados import DADOS, EP2, RES, ROTULOS, SEED, carregar

warnings.filterwarnings("ignore")
CACHE = EP2 / "cache"
N_DOBRAS = 5


def embeddings(textos):
    """Embeddings MiniLM de treino + test1 + test2. Reaproveita o cache do EP1 (treino + test1, mesma
    ordem e mesmos textos) e so calcula os vetores que faltam (o test2)."""
    arq = CACHE / "emb_minilm.npy"
    if arq.exists():
        E = np.load(arq)
        if len(E) == len(textos):
            return E
    ep1 = EP2.parent / "modelo_final" / "cache" / "emb_minilm.npy"
    E = np.load(ep1).astype(np.float32) if ep1.exists() else np.zeros((0, 384), np.float32)
    if len(E) > len(textos):
        E = np.zeros((0, 384), np.float32)
    if len(E) < len(textos):
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
        m.max_seq_length = 256
        novos = m.encode(list(textos[len(E):]), batch_size=16, normalize_embeddings=True, convert_to_numpy=True)
        E = np.r_[E, novos.astype(np.float32)]
    CACHE.mkdir(exist_ok=True)
    np.save(arq, E)
    return E


def bases(X):
    return {
        "lr_pal": lambda a, ya, b: LogisticRegression(C=1.0, max_iter=300).fit(X["pal"][a], ya).predict_log_proba(X["pal"][b]),
        "svc": lambda a, ya, b: LinearSVC(C=0.1, random_state=SEED).fit(X["tfidf"][a], ya).decision_function(X["tfidf"][b]),
        "emb": lambda a, ya, b: np.log(make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=300))
                                       .fit(X["emb"][a], ya).predict_proba(X["emb"][b]) + 1e-6),
        "extremos": lambda a, ya, b: LogisticRegression(C=1.0, max_iter=300).fit(X["tfidf"][a[ya != 1]], ya[ya != 1])
                                     .decision_function(X["tfidf"][b]).reshape(-1, 1),
    }


def meta():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))


def baseline_oficial(textos, y, a, b):
    v = TfidfVectorizer()
    m = LogisticRegression(class_weight="balanced", max_iter=100, random_state=SEED).fit(v.fit_transform(textos[a]), y[a])
    return m.predict(v.transform(textos[b]))


def gravar(modelo_xlsx, pred, saida):
    wb = load_workbook(modelo_xlsx)
    ws = wb.active
    assert ws.max_row - 1 == len(pred)
    for i, p in enumerate(pred):
        ws.cell(row=i + 2, column=2, value=ROTULOS[int(p)])
    saida.parent.mkdir(exist_ok=True)
    wb.save(saida)


def main():
    t0 = time.time()
    d = carregar()
    y, n, n1 = d["y"], len(d["treino"]), len(d["test1"])
    textos = np.r_[d["treino"], d["test1"], d["test2"]]
    tr, te = np.arange(n), np.arange(n, len(textos))
    dobra = np.zeros(n, dtype=int)
    for k, (_, v) in enumerate(StratifiedKFold(N_DOBRAS, shuffle=True, random_state=SEED).split(y, y)):
        dobra[v] = k

    cache = RES / "bases_stacking.npz"
    if cache.exists():
        z = np.load(cache)
        fora = {k[5:]: z[k] for k in z.files if k.startswith("fora_")}
        teste = {k[6:]: z[k] for k in z.files if k.startswith("teste_")}
        baseline = z["baseline"]
        print("modelos-base lidos de", cache.name, flush=True)
    else:
        pal = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(textos)
        car = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True).fit_transform(textos)
        X = {"pal": pal.tocsr().astype(np.float32), "tfidf": sp.hstack([pal, car]).tocsr().astype(np.float32),
             "emb": embeddings(textos)}
        print(f"representacoes prontas ({time.time()-t0:.0f}s)", flush=True)
        fora, teste = {}, {}
        for nome, f in bases(X).items():
            t1 = time.time()
            partes = [f(tr[dobra != k], y[dobra != k], tr[dobra == k]) for k in range(N_DOBRAS)]
            o = np.zeros((n, partes[0].shape[1]))
            for k in range(N_DOBRAS):
                o[dobra == k] = partes[k]
            fora[nome], teste[nome] = o, f(tr, y, te)
            print(f"  {nome} pronto ({time.time()-t1:.0f}s)", flush=True)
        baseline = np.zeros(n, dtype=int)
        for k in range(N_DOBRAS):
            baseline[dobra == k] = baseline_oficial(textos, y, tr[dobra != k], tr[dobra == k])
        np.savez(cache, baseline=baseline, **{f"fora_{k}": v for k, v in fora.items()}, **{f"teste_{k}": v for k, v in teste.items()})

    tem_bert = (RES / "bert_oof.npy").exists()
    if tem_bert:
        bert_oof = np.load(RES / "bert_oof.npy")
        bert_teste = np.r_[np.load(RES / "bert_test1.npy"), np.load(RES / "bert_test2.npy")]
        # no stacking o BERT entra como um escore ordinal "ruim -> bom": logit(c5) - logit(c1)
        # (melhor em validacao cruzada que os 3 logits, o log-softmax ou os logits padronizados)
        fora["bert"] = (bert_oof[:, 2] - bert_oof[:, 0]).reshape(-1, 1)
        teste["bert"] = (bert_teste[:, 2] - bert_teste[:, 0]).reshape(-1, 1)
    else:
        print("AVISO: previsoes do BERT ainda nao existem (rode 01_bert_finetune.py); seguindo sem ele", flush=True)

    ep1 = ["lr_pal", "svc", "emb", "extremos"]
    versoes = {"stacking EP1 (4 blocos)": ep1}
    if tem_bert:
        versoes["stacking EP2 (4 blocos + BERTimbau)"] = ep1 + ["bert"]
    linhas = {"baseline oficial": [float((baseline[dobra == k] == y[dobra == k]).mean()) for k in range(N_DOBRAS)]}
    if tem_bert:
        linhas["BERTimbau sozinho"] = [float((bert_oof[dobra == k].argmax(1) == y[dobra == k]).mean()) for k in range(N_DOBRAS)]
    preds = {}
    for nome, blocos in versoes.items():
        Xm = np.hstack([fora[b] for b in blocos])
        p = np.zeros(n, dtype=int)
        for k in range(N_DOBRAS):
            p[dobra == k] = meta().fit(Xm[dobra != k], y[dobra != k]).predict(Xm[dobra == k])
        preds[nome] = p
        linhas[nome] = [float((p[dobra == k] == y[dobra == k]).mean()) for k in range(N_DOBRAS)]
    tab = pd.DataFrame(linhas, index=[f"dobra {k+1}" for k in range(N_DOBRAS)]).T
    tab["media"] = tab.mean(1)
    tab.to_csv(RES / "02_stacking_dobras.csv")
    print(f"\nacuracia em validacao cruzada de {N_DOBRAS} dobras (semente {SEED}):")
    print((tab * 100).round(2).to_string())
    if tem_bert:
        a, b = preds["stacking EP2 (4 blocos + BERTimbau)"], preds["stacking EP1 (4 blocos)"]
        print(f"respostas do treino em que so um acerta: EP2 {int(((a == y) & (b != y)).sum())} x EP1 {int(((a != y) & (b == y)).sum())}")

    final = list(versoes.values())[-1]
    np.save(RES / "02_oof_pred_final.npy", preds[list(versoes)[-1]])
    m = meta().fit(np.hstack([fora[b] for b in final]), y)
    pt = m.predict(np.hstack([teste[b] for b in final]))
    gravar(DADOS / "test2.xlsx", pt[n1:], EP2 / "entrega" / "test2.xlsx")
    gravar(DADOS / "test1_rotulado.xlsx", pt[:n1], RES / "test1_ep2.xlsx")
    print(f"\nmodelo usado nas planilhas: {final}")
    print("test2 rotulado em ep2/entrega/test2.xlsx | distribuicao:",
          {ROTULOS[i]: int(q) for i, q in enumerate(np.bincount(pt[n1:], minlength=3))})
    print(f"tempo total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
