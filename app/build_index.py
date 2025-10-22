import os, json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

DATA = "data_clean/chunks.jsonl"
INDEX_DIR = "rag_index"
EMB_NAME = "BAAI/bge-m3"  # multilingual embedding model

os.makedirs(INDEX_DIR, exist_ok=True)
embedder = SentenceTransformer(EMB_NAME)

texts, metas = [], []
with open(DATA, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        texts.append(obj["text"])
        metas.append(obj["meta"])

emb = embedder.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
dim = emb.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(emb.astype(np.float32))

faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
with open(os.path.join(INDEX_DIR, "store.jsonl"), "w", encoding="utf-8") as w:
    for t, m in zip(texts, metas):
        w.write(json.dumps({"text": t, "meta": m}, ensure_ascii=False) + "\n")

print("✅ Built FAISS index with", index.ntotal, "chunks.")