"""Pipeline v2 do EP1: modelos-base + memoria difusa + stacking, com semente unica (123).

Tudo o que nao usa rotulos (TF-IDF, atributos manuais, embeddings) e calculado uma unica vez
sobre treino + teste. O que usa rotulos e sempre ajustado so nas linhas de treino do momento:
    - modelos-base: previsoes "fora da dobra" (5 dobras estratificadas, semente 123);
    - vizinhos e memoria: tambem fora da dobra (so consultam rotulos das outras dobras).
A mesma funcao `empilhar` serve para rotular o teste e para repetir o protocolo do baseline
oficial (holdout de 20%, semente 123) sem vazamento.
"""
import importlib
import time
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from atributos import CATEGORICAS, tabela_atributos
from common import LABELS, ROOT, clean

warnings.filterwarnings("ignore")
SEED = 123
N_FOLDS = 5
DADOS, RES = ROOT / "dados", ROOT / "resultados"
viz = importlib.import_module("06_vizinhos")


class Corpus:
    """Treino + teste empilhados (treino primeiro), com todas as representacoes sem rotulo."""

    def __init__(self, embeddings=("e5small", "minilm")):
        t0 = time.time()
        tr, te = pd.read_excel(ROOT / "train.xlsx"), pd.read_excel(ROOT / "test1.xlsx")
        self.n_treino = len(tr)
        self.bruto = np.array([str(t) for t in tr["resp_text"]] + [str(t) for t in te["resp_text"]])
        self.texto = np.array([clean(t) for t in self.bruto])
        self.y = tr["clarity"].map({l: i for i, l in enumerate(LABELS)}).values.astype(int)
        cache = DADOS / "v2_tfidf.npz"
        if cache.exists():
            z = sp.load_npz(cache)
            self.n_pal = int(np.load(DADOS / "v2_tfidf_npal.npy"))
        else:
            pal = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(self.texto)
            car = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                  sublinear_tf=True).fit_transform(self.texto)
            z, self.n_pal = sp.hstack([pal, car]).tocsr().astype(np.float32), pal.shape[1]
            sp.save_npz(cache, z)
            np.save(DADOS / "v2_tfidf_npal.npy", self.n_pal)
        self.tfidf = z                                  # palavras 1-2 + caracteres 2-5
        self.tfidf_pal = z[:, :self.n_pal].tocsr()      # so palavras (espaco de vizinhanca)
        cache = DADOS / "v2_atributos.pkl"
        if cache.exists():
            self.atrib = pd.read_pickle(cache)
        else:
            self.atrib = tabela_atributos(list(self.bruto))
            self.atrib.to_pickle(cache)
        self.emb = {n: np.load(DADOS / f"emb_{n}.npy") for n in embeddings
                    if (DADOS / f"emb_{n}.npy").exists()}
        print(f"corpus pronto em {time.time()-t0:.0f}s | tfidf {self.tfidf.shape} | "
              f"{self.atrib.shape[1]} atributos | embeddings {list(self.emb)}", flush=True)


# ----------------------------------------------------------------------------- modelos-base
def base_svc(c, a, ya, b):
    m = LinearSVC(C=0.1, random_state=SEED).fit(c.tfidf[a], ya)
    return m.decision_function(c.tfidf[b])


def base_svc_pal(c, a, ya, b):
    """LinearSVC so nos n-gramas de palavras (melhor SVM isolado na grade de 09_so_tfidf_svm.py)."""
    m = LinearSVC(C=0.1, random_state=SEED).fit(c.tfidf_pal[a], ya)
    return m.decision_function(c.tfidf_pal[b])


def base_lgbm(c, a, ya, b):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(objective="multiclass", n_estimators=100, learning_rate=0.03, num_leaves=31,
                           min_child_samples=60, colsample_bytree=0.3, subsample=0.7, subsample_freq=1,
                           reg_lambda=1.0, verbose=-1, n_jobs=4, random_state=SEED)
    m.fit(c.atrib.iloc[a], ya, categorical_feature=CATEGORICAS)
    return np.log(m.predict_proba(c.atrib.iloc[b]) + 1e-6)


def base_emb(nome, C=1.0):
    def f(c, a, ya, b):
        E = c.emb[nome]
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=300)).fit(E[a], ya)
        return np.log(m.predict_proba(E[b]) + 1e-6)
    return f


def base_lr_pal(c, a, ya, b):
    """Regressao logistica so nos n-gramas de palavras (erra em lugares diferentes do SVC)."""
    m = LogisticRegression(C=1.0, max_iter=300).fit(c.tfidf_pal[a], ya)
    return m.predict_log_proba(c.tfidf_pal[b])


def base_ordinal(c, a, ya, b):
    """Trata as classes como escala (c1 < c234 < c5): regressao ridge da nota."""
    m = Ridge(alpha=3.0, solver="sag", max_iter=50, random_state=SEED).fit(c.tfidf[a], ya)
    return m.predict(c.tfidf[b]).reshape(-1, 1)


def base_extremos(c, a, ya, b):
    """Classificador c1 x c5 treinado so nos extremos; a pontuacao vira um eixo ruim-bom."""
    e = ya != 1
    m = LogisticRegression(C=1.0, max_iter=300).fit(c.tfidf[a[e]], ya[e])
    return m.decision_function(c.tfidf[b]).reshape(-1, 1)


def bases_disponiveis(c):
    d = {"svc": base_svc, "lgbm": base_lgbm, "lr_pal": base_lr_pal, "ordinal": base_ordinal,
         "extremos": base_extremos, "svc_pal": base_svc_pal}
    for n in c.emb:
        d[f"emb_{n}"] = base_emb(n, 0.01)
    return d


# ----------------------------------------------------------------------------- memoria
def memoria_exata(c, a, ya, b):
    """Para cada linha de `b`: proporcao de rotulos entre as linhas de `a` com texto identico
    (+ log da quantidade). Zeros quando nao ha copia."""
    cont = {}
    for t, r in zip(c.texto[a], ya):
        cont.setdefault(t, np.zeros(3))[r] += 1

    def linha(v):
        n = v.sum()
        return np.r_[v / n if n > 0 else np.zeros(3), np.log1p(n)]

    return np.array([linha(cont.get(t, np.zeros(3))) for t in c.texto[b]])


def memoria_difusa(c, a, ya, b):
    """Para cada linha de `b`: resumo dos rotulos dos vizinhos mais proximos em `a`, em cada
    espaco de vizinhanca (TF-IDF de palavras e embeddings de sentenca)."""
    return np.hstack([viz.resumo(*viz.topk(M[b], M[a], max(viz.KS), False), ya)
                      for M in [c.tfidf_pal] + [c.emb[n] for n in c.emb]])


# ----------------------------------------------------------------------------- stacking
def dobras(y):
    return list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(y, y))


def empilhar(c, a, b, cache=None, verbose=True):
    """Blocos de entrada do meta-classificador para as linhas de treino `a` (sem vazamento) e
    para as linhas-alvo `b` (modelos ajustados em todo `a`). Devolve (blocos_a, blocos_b, dobra_de_a)."""
    arq = RES / f"v2_blocos_{cache}.npz" if cache else None
    ya = c.y[a]
    fold = np.zeros(len(a), dtype=int)
    for k, (_, v) in enumerate(dobras(ya)):
        fold[v] = k
    Za, Zb = {}, {}
    if arq is not None and arq.exists():
        z = np.load(arq)
        Za = {k[2:]: z[k] for k in z.files if k.startswith("a_")}
        Zb = {k[2:]: z[k] for k in z.files if k.startswith("b_")}
    for nome, f in bases_disponiveis(c).items():
        if nome in Za:
            continue
        t0 = time.time()
        partes = [f(c, a[fold != k], ya[fold != k], a[fold == k]) for k in range(N_FOLDS)]
        oof = np.zeros((len(a), partes[0].shape[1]))
        for k in range(N_FOLDS):
            oof[fold == k] = partes[k]
        Za[nome], Zb[nome] = oof, f(c, a, ya, b)
        if verbose:
            ac = f"acuracia fora da dobra {(oof.argmax(1) == ya).mean():.4f}" if oof.shape[1] == 3 else "pontuacao unica"
            print(f"  {nome}: {ac} ({time.time()-t0:.0f}s)", flush=True)
        if arq is not None:
            salvar(arq, Za, Zb)
    # memorias: calculadas como os modelos-base, "fora da dobra" - os rotulos consultados nunca
    # incluem a propria linha nem linhas da mesma dobra (leave-one-out vaza o rotulo em textos repetidos)
    for nome, f in (("mem", memoria_exata), ("viz", memoria_difusa)):
        chave = "memf" if nome == "mem" else f"vizf_{'_'.join(['tfidf'] + list(c.emb))}"
        antigas = [k for k in Za if k.startswith("vizf_")]
        if nome == "viz" and chave not in Za and antigas:      # reaproveita a vizinhanca ja calculada
            chave = antigas[-1]
        if chave not in Za:
            t0 = time.time()
            partes = [f(c, a[fold != k], ya[fold != k], a[fold == k]) for k in range(N_FOLDS)]
            oof = np.zeros((len(a), partes[0].shape[1]))
            for k in range(N_FOLDS):
                oof[fold == k] = partes[k]
            Za[chave], Zb[chave] = oof, f(c, a, ya, b)
            if verbose:
                print(f"  {chave}: {oof.shape[1]} atributos ({time.time()-t0:.0f}s)", flush=True)
            if arq is not None:
                salvar(arq, Za, Zb)
        Za[nome], Zb[nome] = Za[chave], Zb[chave]
    return Za, Zb, fold


def salvar(arq, Za, Zb):
    np.savez(arq, **{f"a_{k}": v for k, v in Za.items() if k not in ("viz", "mem")},
             **{f"b_{k}": v for k, v in Zb.items() if k not in ("viz", "mem")})


def meta(C=0.1):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def juntar(Z, nomes):
    return np.hstack([Z[n] for n in nomes])


def cv_meta(Za, nomes, y, fold, C=0.1):
    X = juntar(Za, nomes)
    pred = np.zeros(len(y), dtype=int)
    for k in range(N_FOLDS):
        pred[fold == k] = meta(C).fit(X[fold != k], y[fold != k]).predict(X[fold == k])
    accs = [float((pred[fold == k] == y[fold == k]).mean()) for k in range(N_FOLDS)]
    return accs, pred


def baseline_oficial(c, a, b):
    """Codigo do baseline oficial: TF-IDF padrao + regressao logistica balanceada, max_iter=100."""
    v = TfidfVectorizer()
    Xa, Xb = v.fit_transform(c.bruto[a]), v.transform(c.bruto[b])
    m = LogisticRegression(class_weight="balanced", max_iter=100, random_state=SEED).fit(Xa, c.y[a])
    return m.predict(Xb)
