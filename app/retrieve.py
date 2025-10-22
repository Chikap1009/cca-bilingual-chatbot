import faiss, json, os
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

INDEX_DIR = "rag_index"
EMB_NAME = "BAAI/bge-m3"  # multilingual embedding model

class HybridRetriever:
    def __init__(self, top_k_embed=8, top_k_bm25=8, rerank_k=8):
        # Load embedding model
        self.embedder = SentenceTransformer(EMB_NAME)
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
        # Load stored text + metadata
        with open(os.path.join(INDEX_DIR, "store.jsonl"), "r", encoding="utf-8") as f:
            self.store = [json.loads(line) for line in f]
        # Prepare BM25 corpus
        self.corpus_tokens = [d["text"].split() for d in self.store]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        # Hyperparameters
        self.top_k_embed = top_k_embed
        self.top_k_bm25 = top_k_bm25
        self.rerank_k = rerank_k

    def embed_search(self, query: str):
        qv = self.embedder.encode([query], normalize_embeddings=True)
        scores, idxs = self.index.search(qv.astype(np.float32), self.top_k_embed)
        return list(idxs[0])

    def bm25_search(self, query: str):
        scores = self.bm25.get_scores(query.split())
        idxs = np.argsort(scores)[::-1][:self.top_k_bm25]
        return list(idxs)

    def retrieve(self, query: str) -> List[Dict]:
        candidates = list(set(self.embed_search(query) + self.bm25_search(query)))
        qv = self.embedder.encode([query], normalize_embeddings=True)[0]
        cand_vecs = self.embedder.encode(
            [self.store[i]["text"] for i in candidates],
            normalize_embeddings=True
        )
        sims = cand_vecs @ qv
        order = np.argsort(sims)[::-1][:self.rerank_k]
        results = []
        for oi in order:
            i = candidates[oi]
            it = self.store[i]
            results.append({
                "text": it["text"],
                "meta": it["meta"],
                "score": float(sims[oi])
            })
        return results