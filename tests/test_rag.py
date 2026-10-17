from pathlib import Path

import numpy as np
import pytest

from app.rag.chunker import chunk_document
from app.rag.document_loader import load_document, load_documents
from app.rag.rag_pipeline import RAGPipeline
from app.rag.vector_store import InMemoryVectorStore


class FakeEmbeddingService:
    """Deterministic embeddings for unit tests."""

    def embed_documents(self, texts):
        vectors = []

        for text in texts:
            lowered = text.lower()

            if "revenue" in lowered or "sales" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "profit" in lowered or "margin" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])

        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, query):
        return self.embed_documents([query])[0]


@pytest.fixture
def sample_document():
    return {
        "source": "sales.txt",
        "content": "Revenue increased. Profit margin requires review.",
        "metadata": {"file_name": "sales.txt"},
    }


@pytest.fixture
def pipeline():
    return RAGPipeline(
        embedding_service=FakeEmbeddingService(),
        vector_store=InMemoryVectorStore(),
        chunk_size=80,
        overlap=10,
    )


def test_load_text_document(tmp_path):
    path = tmp_path / "report.txt"
    path.write_text("Sales revenue increased.", encoding="utf-8")

    result = load_document(path)

    assert result["source"] == "report.txt"
    assert "revenue" in result["content"]
    assert result["metadata"]["file_type"] == ".txt"


def test_load_markdown_document(tmp_path):
    path = tmp_path / "report.md"
    path.write_text("# Sales\nRevenue increased.", encoding="utf-8")

    result = load_document(path)

    assert result["metadata"]["file_type"] == ".md"


def test_missing_document_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_document(tmp_path / "missing.txt")


def test_unsupported_document_type_raises_error(tmp_path):
    path = tmp_path / "report.pdf"
    path.write_text("not a real PDF", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported"):
        load_document(path)


def test_empty_document_raises_error(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("   ", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_document(path)


def test_load_documents_from_directory(tmp_path):
    (tmp_path / "one.txt").write_text("One", encoding="utf-8")
    (tmp_path / "two.md").write_text("Two", encoding="utf-8")
    (tmp_path / "ignore.csv").write_text("Three", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert len(documents) == 2
    assert [doc["source"] for doc in documents] == [
        "one.txt",
        "two.md",
    ]


def test_missing_directory_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "missing")


def test_chunk_document(sample_document):
    chunks = chunk_document(
        sample_document,
        chunk_size=20,
        overlap=5,
    )

    assert len(chunks) > 1
    assert all(chunk["text"] for chunk in chunks)


def test_chunk_preserves_source_metadata(sample_document):
    chunks = chunk_document(
        sample_document,
        chunk_size=20,
        overlap=5,
    )

    assert chunks[0]["source"] == "sales.txt"
    assert chunks[0]["metadata"]["file_name"] == "sales.txt"


def test_chunk_invalid_size(sample_document):
    with pytest.raises(ValueError):
        chunk_document(sample_document, chunk_size=0)


def test_chunk_invalid_overlap(sample_document):
    with pytest.raises(ValueError):
        chunk_document(
            sample_document,
            chunk_size=10,
            overlap=10,
        )


def test_chunk_empty_content():
    document = {"source": "empty.txt", "content": ""}

    assert chunk_document(document) == []


def test_chunk_wrong_document_type():
    with pytest.raises(TypeError):
        chunk_document("not a document")


def test_vector_store_add_and_search():
    store = InMemoryVectorStore()

    chunks = [
        {"chunk_id": "a", "source": "a.txt", "text": "Revenue"},
        {"chunk_id": "b", "source": "b.txt", "text": "Profit"},
    ]

    embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )

    store.add(chunks, embeddings)

    results = store.search(
        np.array([1.0, 0.0]),
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "a"
    assert results[0]["score"] == pytest.approx(1.0)


def test_vector_store_empty_search():
    store = InMemoryVectorStore()

    results = store.search(
        np.array([1.0, 0.0]),
        top_k=3,
    )

    assert results == []


def test_vector_store_rejects_mismatched_counts():
    store = InMemoryVectorStore()

    with pytest.raises(ValueError, match="match"):
        store.add(
            [{"text": "Revenue"}],
            np.array([[1.0, 0.0], [0.0, 1.0]]),
        )


def test_vector_store_rejects_dimension_mismatch():
    store = InMemoryVectorStore()

    store.add(
        [{"text": "Revenue"}],
        np.array([[1.0, 0.0]]),
    )

    with pytest.raises(ValueError, match="dimension"):
        store.search(np.array([1.0, 0.0, 0.0]))


def test_vector_store_rejects_invalid_top_k():
    store = InMemoryVectorStore()

    with pytest.raises(ValueError):
        store.search(np.array([1.0, 0.0]), top_k=0)


def test_vector_store_rejects_zero_vector():
    store = InMemoryVectorStore()

    with pytest.raises(ValueError, match="zero"):
        store.add(
            [{"text": "Revenue"}],
            np.array([[0.0, 0.0]]),
        )


def test_vector_store_rejects_non_finite_embeddings():
    store = InMemoryVectorStore()

    with pytest.raises(ValueError, match="finite"):
        store.add(
            [{"text": "Revenue"}],
            np.array([[float("nan"), 1.0]]),
        )


def test_pipeline_ingests_documents(tmp_path, pipeline):
    (tmp_path / "sales.txt").write_text(
        "Revenue increased. Sales improved.",
        encoding="utf-8",
    )

    result = pipeline.ingest_directory(str(tmp_path))

    assert result["documents"] == 1
    assert result["chunks"] >= 1
    assert pipeline.vector_store.size >= 1


def test_pipeline_retrieves_relevant_chunk(tmp_path, pipeline):
    (tmp_path / "sales.txt").write_text(
        "Sales revenue increased significantly.",
        encoding="utf-8",
    )

    pipeline.ingest_directory(str(tmp_path))

    results = pipeline.retrieve("What happened to revenue?")

    assert len(results) >= 1
    assert results[0]["source"] == "sales.txt"
    assert "score" in results[0]


def test_pipeline_requires_ingestion(pipeline):
    with pytest.raises(RuntimeError, match="(?i)ingest",):
        pipeline.retrieve("What happened to revenue?")


def test_pipeline_rejects_empty_query(tmp_path, pipeline):
    (tmp_path / "sales.txt").write_text(
        "Revenue increased.",
        encoding="utf-8",
    )
    pipeline.ingest_directory(str(tmp_path))

    with pytest.raises(ValueError):
        pipeline.retrieve("  ")


def test_answer_without_llm_returns_evidence(tmp_path, pipeline):
    (tmp_path / "sales.txt").write_text(
        "Revenue increased.",
        encoding="utf-8",
    )
    pipeline.ingest_directory(str(tmp_path))

    result = pipeline.answer("What happened to revenue?")

    assert result["answer_generated"] is False
    assert result["sources"] == ["sales.txt"]
    assert len(result["evidence"]) >= 1


def test_answer_with_llm_provider(tmp_path):
    class FakeLLM:
        def invoke(self, prompt):
            assert "Retrieved evidence" in prompt
            return "Revenue increased according to sales.txt."

    pipeline = RAGPipeline(
        embedding_service=FakeEmbeddingService(),
        llm_provider=FakeLLM(),
    )

    (tmp_path / "sales.txt").write_text(
        "Sales revenue increased.",
        encoding="utf-8",
    )

    pipeline.ingest_directory(str(tmp_path))

    result = pipeline.answer("What happened to sales?")

    assert result["answer_generated"] is True
    assert "sales.txt" in result["answer"]
    assert "sales.txt" in result["sources"]


def test_pipeline_empty_directory_returns_zero_chunks(
    tmp_path,
    pipeline,
):
    result = pipeline.ingest_directory(str(tmp_path))

    assert result == {"documents": 0, "chunks": 0}