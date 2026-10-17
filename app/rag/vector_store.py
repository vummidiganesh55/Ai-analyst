from typing import Any

import numpy as np


class InMemoryVectorStore:
    """A simple in-memory cosine-similarity vector store."""

    def __init__(self):
        self._chunks: list[dict[str, Any]] = []
        self._embeddings: np.ndarray | None = None

    @property
    def size(self) -> int:
        return len(self._chunks)

    def add(
        self,
        chunks: list[dict[str, Any]],
        embeddings: np.ndarray,
    ) -> None:
        """Add chunks and their corresponding embedding vectors."""
        if not chunks:
            return

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array.")

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings."
            )

        if embeddings.shape[1] == 0:
            raise ValueError("Embedding dimension cannot be zero.")

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings must contain finite values.")

        norms = np.linalg.norm(embeddings, axis=1)

        if np.any(norms == 0):
            raise ValueError("Embedding vectors cannot be zero vectors.")

        normalized = embeddings / norms[:, None]

        if self._embeddings is not None:
            if normalized.shape[1] != self._embeddings.shape[1]:
                raise ValueError("Embedding dimensions do not match.")

            self._embeddings = np.vstack(
                [self._embeddings, normalized]
            )
        else:
            self._embeddings = normalized.copy()

        self._chunks.extend(dict(chunk) for chunk in chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Return the most similar chunks."""
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k < 1
        ):
            raise ValueError("top_k must be a positive integer.")

        if self._embeddings is None or not self._chunks:
            return []

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_embedding.ndim != 1:
            raise ValueError("query_embedding must be a 1D array.")

        if query_embedding.shape[0] != self._embeddings.shape[1]:
            raise ValueError("Query embedding dimension does not match.")

        if not np.isfinite(query_embedding).all():
            raise ValueError("Query embedding must contain finite values.")

        norm = np.linalg.norm(query_embedding)

        if norm == 0:
            raise ValueError("Query embedding cannot be a zero vector.")

        query_embedding = query_embedding / norm

        scores = self._embeddings @ query_embedding
        indices = np.argsort(-scores, kind="stable")[:top_k]

        results = []

        for index in indices:
            result = dict(self._chunks[int(index)])
            result["score"] = float(scores[index])
            results.append(result)

        return results