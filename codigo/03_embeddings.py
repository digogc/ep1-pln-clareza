"""Abordagem 3a - embeddings estaticos: cada resposta vira um vetor denso + classificador.

Representacoes (nenhuma usa rotulos; sao ajustadas sobre treino + teste):
  spacy_media - media dos vetores pre-treinados do spaCy (pt_core_news_lg, 300d)
  spacy_idf   - idem, ponderando cada palavra pelo IDF
  w2v_idf     - Word2Vec skip-gram (200d) treinado no proprio corpus, media ponderada por IDF
  doc2vec     - Doc2Vec PV-DBOW (200d) treinado no proprio corpus
  lsa         - TF-IDF de palavras (1-2-gram) reduzido a 300 dimensoes por SVD
  tudo        - concatenacao de spacy_idf + w2v_idf + doc2vec + lsa (1000d)
Classificadores: regressao logistica e rede MLP, com busca em grade.
Uso:  python codigo/03_embeddings.py     (precisa: python -m spacy download pt_core_news_lg)
"""
import re
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import load_data, save_test_xlsx, ROOT, N_FOLDS

warnings.filterwarnings("ignore")
RES, DADOS = ROOT / "resultados", ROOT / "dados"
TOKEN = re.compile(r"[A-Za-zÀ-ÿ]+|\d+")


def tokens(t):
    return TOKEN.findall(t.lower())


def idf(T):
    v = TfidfVectorizer(tokenizer=tokens, token_pattern=None, lowercase=False, min_df=2).fit(T)
    return dict(zip(v.get_feature_names_out(), v.idf_))


def media(T, vet, dim, pesos=None):
    X = np.zeros((len(T), dim), dtype=np.float32)
    for i, t in enumerate(T):
        soma, tot = np.zeros(dim), 0.0
        for p in tokens(t):
            v = vet.get(p)
            if v is None:
                continue
            w = 1.0 if pesos is None else pesos.get(p, 1.0)
            soma += w * v
            tot += w
        if tot > 0:
            X[i] = soma / tot
    return X


def vetores_spacy(T):
    import pt_core_news_lg
    nlp = pt_core_news_lg.load(disable=["tagger", "parser", "ner", "lemmatizer", "morphologizer",
                                        "attribute_ruler"])
    vocab = {p for t in T for p in tokens(t)}
    vet = {p: nlp.vocab.get_vector(p) for p in vocab if nlp.vocab.has_vector(p)}
    print(f"   spaCy: {len(vet)} de {len(vocab)} palavras do corpus tem vetor pre-treinado", flush=True)
    return vet


def emb_w2v(T):
    from gensim.models import Word2Vec
    m = Word2Vec([tokens(t) for t in T], vector_size=200, window=5, min_count=2, sg=1, negative=10,
                 epochs=10, workers=2, seed=42)
    return media(T, {p: m.wv[p] for p in m.wv.index_to_key}, 200, idf(T))


def emb_doc2vec(T):
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    m = Doc2Vec([TaggedDocument(tokens(t), [i]) for i, t in enumerate(T)], vector_size=200, dm=0,
                min_count=2, negative=10, epochs=20, workers=2, seed=42)
    return np.vstack([m.dv[i] for i in range(len(T))]).astype(np.float32)


def emb_lsa(T):
    X = TfidfVectorizer(min_df=2, max_df=0.9, ngram_range=(1, 2), sublinear_tf=True).fit_transform(T)
    return TruncatedSVD(n_components=300, random_state=42).fit_transform(X).astype(np.float32)


def embedding(nome, T, _cache={}):
    arq = DADOS / f"emb_{nome}.npy"
    if arq.exists():
        return np.load(arq)
    t0 = time.time()
    if nome.startswith("spacy"):
        if "sp" not in _cache:
            _cache["sp"] = vetores_spacy(T)
        X = media(T, _cache["sp"], 300, idf(T) if nome == "spacy_idf" else None)
    elif nome == "tudo":
        X = np.hstack([embedding(n, T) for n in ("spacy_idf", "w2v_idf", "doc2vec", "lsa")])
    else:
        X = {"w2v_idf": emb_w2v, "doc2vec": emb_doc2vec, "lsa": emb_lsa}[nome](T)
    np.save(arq, X)
    print(f"   embedding {nome}: {X.shape} em {time.time()-t0:.0f}s", flush=True)
    return X


def fabrica(clf, p):
    if clf == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(C=p["C"], max_iter=500))
    return make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=p["camadas"], alpha=p["alpha"], early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=5, max_iter=60, batch_size=128, random_state=42))


GRADE_LR = [{"C": C} for C in (0.001, 0.01, 0.1, 1)]
GRADE_MLP = [{"camadas": (256,), "alpha": 0.01}, {"camadas": (256,), "alpha": 1.0},
             {"camadas": (512, 256), "alpha": 1.0}]
REPS = ["spacy_media", "spacy_idf", "w2v_idf", "doc2vec", "lsa", "tudo"]
REPS_MLP = ["spacy_idf", "w2v_idf", "tudo"]


def um_fold(clf, p, Xa, ya, Xb, yb):
    m = fabrica(clf, p).fit(Xa, ya)
    pb = m.predict_proba(Xb)
    return float((pb.argmax(1) == yb).mean()), float((m.predict(Xa) == ya).mean()), pb


def main():
    tr, te = load_data()
    y, fold, N = tr["y"].values, tr["fold"].values, len(tr)
    T = list(tr["text"]) + list(te["text"])
    linhas, melhor = [], None
    for rep in REPS:
        X = embedding(rep, T)
        Xtr = X[:N]
        cand = [("LR", p) for p in GRADE_LR] + ([("MLP", p) for p in GRADE_MLP] if rep in REPS_MLP else [])
        for clf, p in cand:
            t0 = time.time()
            res = Parallel(n_jobs=2)(delayed(um_fold)(clf, p, Xtr[fold != k], y[fold != k],
                                                      Xtr[fold == k], y[fold == k]) for k in range(N_FOLDS))
            oof = np.zeros((N, 3), dtype=np.float32)
            for k in range(N_FOLDS):
                oof[fold == k] = res[k][2]
            va = [r[0] for r in res]
            l = dict(representacao=rep, dim=X.shape[1], classificador=clf, parametros=str(p),
                     acc_val_media=np.mean(va), acc_val_desvio=np.std(va),
                     acc_treino_media=np.mean([r[1] for r in res]),
                     **{f"acc_dobra{k}": v for k, v in enumerate(va)})
            linhas.append(l)
            print(f"{rep:12s} {clf:3s} {str(p):38s} val={l['acc_val_media']:.4f} "
                  f"treino={l['acc_treino_media']:.4f} ({time.time()-t0:.0f}s)", flush=True)
            if melhor is None or l["acc_val_media"] > melhor[0]["acc_val_media"]:
                melhor = (l, rep, clf, p, oof)
        pd.DataFrame(linhas).to_csv(RES / "03_embeddings_grid.csv", index=False)
    pd.DataFrame(linhas).sort_values("acc_val_media", ascending=False).to_csv(
        RES / "03_embeddings_grid.csv", index=False)
    l, rep, clf, p, oof = melhor
    print(f"\nMELHOR: {rep} + {clf} {p} val={l['acc_val_media']:.4f}", flush=True)
    np.save(RES / "03_embeddings_oof_melhor.npy", oof)
    X = embedding(rep, T)
    pt = fabrica(clf, p).fit(X[:N], y).predict_proba(X[N:])
    np.save(RES / "03_embeddings_teste_melhor.npy", pt)
    out = save_test_xlsx(pt.argmax(1), "test1_embeddings.xlsx")
    print("teste rotulado em", out, "| distribuicao:",
          pd.Series(pt.argmax(1)).value_counts().sort_index().to_dict(), flush=True)


if __name__ == "__main__":
    main()
