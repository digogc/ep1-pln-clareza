"""Abordagem 5 - embeddings de sentenca de modelos pequenos (CPU), extraidos uma unica vez.

Os vetores ficam em dados/emb_<nome>.npy (treino seguido do teste, na ordem original).
Uso:  python codigo/05_embeddings_sentenca.py [nome ...]
"""
import sys
import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from common import load_data, ROOT

MODELOS = {
    "e5small": ("intfloat/multilingual-e5-small", "query: ", 256),
    "minilm": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "", 256),
    "mpnet": ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "", 256),
    "distiluse": ("sentence-transformers/distiluse-base-multilingual-cased-v2", "", 128),
}


def main():
    nomes = sys.argv[1:] or ["e5small", "minilm"]
    tr, te = load_data()
    textos = list(tr["text"]) + list(te["text"])
    torch.set_num_threads(4)
    for nome in nomes:
        saida = ROOT / "dados" / f"emb_{nome}.npy"
        if saida.exists():
            print(nome, "ja existe", flush=True)
            continue
        hf, prefixo, max_len = MODELOS[nome]
        t0 = time.time()
        m = SentenceTransformer(hf, device="cpu")
        m.max_seq_length = max_len
        ent = [prefixo + t for t in textos]
        partes = []
        for i in range(0, len(ent), 2000):
            partes.append(m.encode(ent[i:i + 2000], batch_size=16, show_progress_bar=False,
                                   normalize_embeddings=True, convert_to_numpy=True))
            print(f"  {nome}: {min(i + 2000, len(ent))}/{len(ent)} em {time.time()-t0:.0f}s", flush=True)
        E = np.vstack(partes)
        np.save(saida, E.astype(np.float32))
        print(f"{nome}: {E.shape} em {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
