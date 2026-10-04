"""Reprocessa o baseline oficial (TF-IDF + regressao logistica, parametros padrao)
sob varios protocolos de avaliacao, para explicar por que o numero muda de grupo para grupo.
Uso:  python codigo/00_baseline.py
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.pipeline import make_pipeline
from common import load_data, ROOT, LABELS, SEED


def modelo():
    return make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))


def cv_acc(X, y, splits):
    accs = []
    for a, b in splits:
        accs.append(float((modelo().fit(X[a], y[a]).predict(X[b]) == y[b]).mean()))
    return np.mean(accs), np.std(accs), accs


def main():
    tr, _ = load_data()
    bruto = pd.read_excel(ROOT / "train.xlsx")["resp_text"].astype(str).values
    X, y, fold = tr["text"].values, tr["y"].values, tr["fold"].values
    linhas = []

    def reg(nome, m, s, extra=""):
        linhas.append(dict(protocolo=nome, acc=m, desvio=s, obs=extra))
        print(f"{nome:62s} {m:.4f} +- {s:.4f} {extra}", flush=True)

    # 1) protocolo do projeto: textos identicos sempre na mesma dobra
    grp = [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(5)]
    reg("5 dobras AGRUPADAS por texto (dados/folds.csv)", *cv_acc(X, y, grp)[:2])
    # 2) dobras estratificadas comuns: copias de um texto caem em treino e validacao
    for seed in (0, 1, 42):
        sk = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y))
        m, s, _ = cv_acc(X, y, sk)
        reg(f"5 dobras estratificadas comuns (seed={seed})", m, s)
    reg("5 dobras comuns, texto bruto (sem limpar espacos)",
        *cv_acc(bruto, y, list(StratifiedKFold(5, shuffle=True, random_state=42).split(bruto, y)))[:2])
    reg("10 dobras estratificadas comuns (seed=42)",
        *cv_acc(X, y, list(StratifiedKFold(10, shuffle=True, random_state=42).split(X, y)))[:2])
    reg("5 dobras SEM embaralhar (KFold padrao do cross_val_score)",
        *cv_acc(X, y, list(StratifiedKFold(5).split(X, y)))[:2])
    # 3) holdout unico, como em train_test_split
    for ts in (0.2, 0.1):
        accs = []
        for seed in range(10):
            a, b = train_test_split(np.arange(len(y)), test_size=ts, random_state=seed, stratify=y)
            accs.append(float((modelo().fit(X[a], y[a]).predict(X[b]) == y[b]).mean()))
        reg(f"holdout {int(ts*100)}% (10 sementes)", np.mean(accs), np.std(accs),
            f"min {min(accs):.4f} max {max(accs):.4f}")

    # 4) onde esta a diferenca: linhas com copia no treino x linhas unicas (dobras comuns, seed 42)
    sk = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    pred = np.zeros(len(y), dtype=int)
    vazou = np.zeros(len(y), dtype=bool)
    for a, b in sk:
        pred[b] = modelo().fit(X[a], y[a]).predict(X[b])
        vazou[b] = np.isin(X[b], X[a])
    pg = np.zeros(len(y), dtype=int)
    for a, b in grp:
        pg[b] = modelo().fit(X[a], y[a]).predict(X[b])
    dup = pd.Series(X).duplicated(keep=False).values
    print(f"\nlinhas com copia do proprio texto no treino da dobra (dobras comuns): {vazou.sum()} ({vazou.mean():.1%})")
    print(f"  dobras comuns   : com copia {(pred[vazou]==y[vazou]).mean():.4f} | sem copia {(pred[~vazou]==y[~vazou]).mean():.4f}")
    print(f"  dobras agrupadas: repetidas {(pg[dup]==y[dup]).mean():.4f} | unicas    {(pg[~dup]==y[~dup]).mean():.4f}")
    pd.DataFrame(linhas).to_csv(ROOT / "resultados" / "00_baseline_protocolos.csv", index=False)


if __name__ == "__main__":
    main()
