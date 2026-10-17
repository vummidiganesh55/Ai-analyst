from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = {".txt", ".md"}


def load_document(file_path: str | Path) -> dict[str, Any]:
    """
    Load a text or Markdown document.

    Returns:
        Dictionary containing source, content, and metadata.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension: {path.suffix}"
        )

    content = path.read_text(encoding="utf-8").strip()

    if not content:
        raise ValueError("Document is empty.")

    return {
        "source": path.name,
        "content": content,
        "metadata": {
            "file_name": path.name,
            "file_type": path.suffix.lower(),
            "file_path": str(path),
        },
    }


def load_documents(directory: str | Path) -> list[dict[str, Any]]:
    """Load supported documents from a directory."""
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Document directory not found: {directory}"
        )

    if not directory.is_dir():
        raise ValueError(
            f"Expected a directory: {directory}"
        )

    documents = []

    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append(load_document(path))

    return documents