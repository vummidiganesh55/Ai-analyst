from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np


def _write_embedding_trace(message: str) -> None:
    """Write embedding diagnostics to the project trace file."""
    trace_file = (
        Path(__file__).resolve().parents[2]
        / "mcp_rag_trace.log"
    )

    try:
        with trace_file.open("a", encoding="utf-8") as file:
            file.write(f"{perf_counter():.6f} {message}\n")
            file.flush()
    except OSError:
        # Diagnostic logging must not interrupt embedding operations.
        pass


class EmbeddingService:
    """Generate semantic embeddings using Sentence Transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model=None,
    ):
        self.model_name = model_name
        self._model = model

    def _load_model(self):
        if self._model is None:
            _write_embedding_trace(
                f"EMBED 1: Loading model {self.model_name}"
            )

            started = perf_counter()

            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device="cpu",
            )

            _write_embedding_trace(
                "EMBED 2: Model loaded successfully. "
                f"elapsed={perf_counter() - started:.2f}s"
            )
        else:
            _write_embedding_trace(
                "EMBED 2: Reusing existing model instance"
            )

        return self._model

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """Embed a collection of documents."""
        _write_embedding_trace(
            f"EMBED 0: Received {len(texts)} document chunks"
        )

        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        if any(
            not isinstance(text, str) or not text.strip()
            for text in texts
        ):
            raise ValueError(
                "All document texts must be non-empty strings."
            )

        model = self._load_model()

        _write_embedding_trace(
            "EMBED 3: Starting model.encode"
        )

        started = perf_counter()

        vectors = model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        _write_embedding_trace(
            "EMBED 4: model.encode completed. "
            f"elapsed={perf_counter() - started:.2f}s"
        )

        result = np.asarray(vectors, dtype=np.float32)

        _write_embedding_trace(
            f"EMBED 5: Returning embeddings with shape {result.shape}"
        )

        return result

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single search query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "query must be a non-empty string."
            )

        return self.embed_documents([query])[0]