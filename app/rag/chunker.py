from typing import Any


def chunk_document(
    document: dict[str, Any],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict[str, Any]]:
    """
    Split a document into overlapping character-based chunks.

    chunk_size and overlap are measured in characters, not tokens.
    """
    if not isinstance(document, dict):
        raise TypeError("document must be a dictionary.")

    if not isinstance(document.get("content"), str):
        raise ValueError("Document must contain string content.")

    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer.")

    if (
        isinstance(overlap, bool)
        or not isinstance(overlap, int)
        or overlap < 0
    ):
        raise ValueError("overlap must be a non-negative integer.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    content = document["content"].strip()

    if not content:
        return []

    step = chunk_size - overlap
    chunks = []

    for index, start in enumerate(range(0, len(content), step)):
        text = content[start : start + chunk_size]

        if not text:
            continue

        chunks.append(
            {
                "chunk_id": f"{document.get('source', 'document')}:{index}",
                "text": text,
                "source": document.get("source", "unknown"),
                "metadata": {
                    **document.get("metadata", {}),
                    "chunk_index": index,
                    "start_char": start,
                    "end_char": start + len(text),
                },
            }
        )

    return chunks