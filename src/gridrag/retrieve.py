"""Hybrid retrieval: semantic search (embeddings) + keyword search (BM25),
combined with reciprocal rank fusion, with metadata filters.

Semantic search finds paraphrases ("stay connected during a voltage dip" ~
"fault-ride-through"); BM25 finds exact terms ("type D", "LFSM-O", "Article 16").
"""
import json
import re

import numpy as np
import ollama
from rank_bm25 import BM25Okapi

from .config import EMBED_MODEL, EMBEDDINGS_FILE, INDEX_CHUNKS_FILE, RRF_K, TOP_K

TOKEN_RE = re.compile(r"\w+(?:[-,.]\w+)*")  # \w keeps accented letters (défaut, Spannungsbereiche)
REFERENCE_RE = re.compile(r"\b(table|figure|article)\s+(\d+(?:\.\d+)?)", re.I)  # "Table 7.2", "Article 16"
COMPLIANCE_ARTICLES = range(42, 58)  # compliance tests and simulations (Title IV)
COMPLIANCE_RE = re.compile(r"test|simulat|complian|verif|demonstrat", re.I)
RECITALS_RE = re.compile(r"recital|preamble|\bwhy\b", re.I)


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class Retriever:
    def __init__(self):
        self.chunks = [json.loads(line) for line in open(INDEX_CHUNKS_FILE, encoding="utf-8")]
        self.vectors = np.load(EMBEDDINGS_FILE)
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    def _allowed(self, jurisdiction, in_force_on, articles):
        """Boolean mask from metadata filters: jurisdiction, version in force on a date, article numbers."""
        mask = np.ones(len(self.chunks), dtype=bool)
        for i, c in enumerate(self.chunks):
            if jurisdiction and c["jurisdiction"] not in jurisdiction:
                mask[i] = False
            if in_force_on and (c["valid_from"] > in_force_on or (c["valid_to"] and c["valid_to"] < in_force_on)):
                mask[i] = False
            if articles and c["article"] not in articles:
                mask[i] = False
        return mask

    def _on_topic(self, query):
        """Boolean mask without the compliance articles and the recitals, unless the query asks for them.
        They repeat the words of the requirements but not their values, so they push the requirements out."""
        mask = np.ones(len(self.chunks), dtype=bool)
        for i, c in enumerate(self.chunks):
            if c["article"] in COMPLIANCE_ARTICLES and not COMPLIANCE_RE.search(query):
                mask[i] = False
            if c["article"] is None and not RECITALS_RE.search(query):
                mask[i] = False
        return mask

    def semantic_ranking(self, query):
        q = np.array(ollama.embed(model=EMBED_MODEL, input=[query])["embeddings"][0], dtype=np.float32)
        scores = self.vectors @ (q / np.linalg.norm(q))  # cosine similarity
        return list(np.argsort(-scores)), scores

    def keyword_ranking(self, query):
        scores = self.bm25.get_scores(tokenize(query))
        return list(np.argsort(-scores)), scores

    def reference_ranking(self, query, ranking):
        """Chunks containing a table, figure or article named in the query ("Table 7.2"), in the
        order of another ranking. Embeddings do not know what "Table 7.2" refers to."""
        refs = REFERENCE_RE.findall(query)
        if not refs:
            return []
        patterns = [re.compile(rf"\b{kind}\s+{re.escape(num)}(?![\d.]\d)", re.I) for kind, num in refs]
        article_refs = {int(num) for kind, num in refs if kind.lower() == "article" and num.isdigit()}
        return [i for i in ranking
                if self.chunks[i]["article"] in article_refs or any(p.search(self.chunks[i]["text"]) for p in patterns)]

    def search(self, query, k=TOP_K, mode="hybrid", jurisdiction=None, in_force_on=None, articles=None):
        """Return the top-k chunks. mode: 'semantic', 'keyword' or 'hybrid'."""
        allowed = self._allowed(jurisdiction, in_force_on, articles)
        rankings = []
        if mode in ("semantic", "hybrid"):
            rankings.append(self.semantic_ranking(query)[0])
        if mode in ("keyword", "hybrid"):
            rankings.append(self.keyword_ranking(query)[0])
        if mode == "hybrid":
            rankings.append(self.reference_ranking(query, rankings[0]))
            allowed &= self._on_topic(query)

        # Reciprocal rank fusion: a chunk ranked high by either method rises to the top
        fused = {}
        for ranking in rankings:
            for rank, idx in enumerate(r for r in ranking if allowed[r]):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [dict(self.chunks[i], score=round(fused[i], 4)) for i in best]


def cite(chunk):
    """Human-readable citation of a chunk."""
    art = f"Art. {chunk['article']}" if chunk["article"] else "Preamble"
    pages = chunk["page_start"] if chunk["page_start"] == chunk["page_end"] else f"{chunk['page_start']}-{chunk['page_end']}"
    return f"{chunk['document']}, {art}, p. {pages}"
