from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log, sqrt
from pathlib import Path
import re
from typing import Dict, Iterable, List


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str
    tfidf_vector: Dict[str, float]


@dataclass
class SearchResult:
    chunk_id: str
    source: str
    text: str
    score: float


class SimpleRAGEngine:
    """
    Lightweight retrieval engine:
    1) Load docs
    2) Split into chunks
    3) Build TF-IDF vectors
    4) Retrieve by cosine similarity
    """

    def __init__(self, docs_dir: Path) -> None:
        self.docs_dir = docs_dir
        self.chunks: List[Chunk] = []
        self.idf: Dict[str, float] = {}
        self._doc_count = 0

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def doc_count(self) -> int:
        return self._doc_count

    def rebuild(self) -> None:
        raw_chunks = self._load_and_chunk_documents()
        self.idf = self._compute_idf(tokens for _, _, tokens in raw_chunks)
        self.chunks = []

        for idx, (source, text, tokens) in enumerate(raw_chunks, start=1):
            vector = self._to_tfidf_vector(tokens, self.idf)
            self.chunks.append(
                Chunk(
                    chunk_id=f"chunk-{idx}",
                    source=source,
                    text=text,
                    tfidf_vector=vector,
                )
            )

    def search(self, question: str, top_k: int = 3) -> List[SearchResult]:
        query_tokens = self._tokenize(question)
        if not query_tokens or not self.chunks:
            return []

        query_vector = self._to_tfidf_vector(query_tokens, self.idf)
        scored: List[SearchResult] = []

        for chunk in self.chunks:
            score = self._cosine_similarity(query_vector, chunk.tfidf_vector)
            if score <= 0:
                continue
            scored.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    text=chunk.text,
                    score=score,
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def answer_question(self, question: str, results: List[SearchResult]) -> str:
        if not results:
            return (
                "I could not find matching policy text in the local knowledge base. "
                "Please rephrase your question or add more documents."
            )

        top = results[0]
        if top.score < 0.12:
            return (
                "I found weakly related information only. "
                "Please verify with a human before taking action."
            )

        lines = ["Based on the available policy documents:"]
        for item in results[:3]:
            lines.append(f"- ({item.source}) {item.text}")
        return "\n".join(lines)

    def _load_and_chunk_documents(self) -> List[tuple[str, str, List[str]]]:
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Docs directory does not exist: {self.docs_dir}")

        docs = sorted(self.docs_dir.glob("*.md")) + sorted(self.docs_dir.glob("*.txt"))
        self._doc_count = len(docs)
        raw_chunks: List[tuple[str, str, List[str]]] = []

        for doc_path in docs:
            source = doc_path.name
            text = doc_path.read_text(encoding="utf-8")
            for chunk_text in self._chunk_text(text):
                tokens = self._tokenize(chunk_text)
                if not tokens:
                    continue
                raw_chunks.append((source, chunk_text, tokens))

        return raw_chunks

    def _chunk_text(self, text: str, max_chars: int = 420) -> Iterable[str]:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        buffer = ""
        for para in paragraphs:
            if len(buffer) + len(para) + 1 <= max_chars:
                buffer = f"{buffer} {para}".strip()
            else:
                if buffer:
                    yield buffer
                buffer = para
        if buffer:
            yield buffer

    def _tokenize(self, text: str) -> List[str]:
        return [m.group(0).lower() for m in TOKEN_PATTERN.finditer(text)]

    def _compute_idf(self, token_docs: Iterable[List[str]]) -> Dict[str, float]:
        df: Dict[str, int] = {}
        total_docs = 0

        for tokens in token_docs:
            total_docs += 1
            seen = set(tokens)
            for token in seen:
                df[token] = df.get(token, 0) + 1

        idf: Dict[str, float] = {}
        for token, freq in df.items():
            # +1 smoothing keeps values stable for tiny demo datasets
            idf[token] = log((1 + total_docs) / (1 + freq)) + 1
        return idf

    def _to_tfidf_vector(self, tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        vector: Dict[str, float] = {}
        for token, count in counts.items():
            if token not in idf:
                continue
            tf = count / total
            vector[token] = tf * idf[token]
        return vector

    def _cosine_similarity(self, v1: Dict[str, float], v2: Dict[str, float]) -> float:
        if not v1 or not v2:
            return 0.0

        common = set(v1).intersection(v2)
        numerator = sum(v1[t] * v2[t] for t in common)
        norm1 = sqrt(sum(x * x for x in v1.values()))
        norm2 = sqrt(sum(x * x for x in v2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return numerator / (norm1 * norm2)
