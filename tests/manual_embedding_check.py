import sys
import time
import traceback
from pathlib import Path

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.embeddings import EmbeddingService


def main():
    print("Starting embedding diagnostic...", flush=True)
    print(f"Project root: {PROJECT_ROOT}", flush=True)
    print(f"Python executable: {sys.executable}", flush=True)

    service = EmbeddingService()

    try:
        # Test 1: Load the embedding model.
        started = time.perf_counter()

        print("\nStep 1: Loading embedding model...", flush=True)

        model = service._load_model()

        print(
            f"Step 1 passed: Model loaded in "
            f"{time.perf_counter() - started:.2f} seconds.",
            flush=True,
        )

        # Test 2: Generate embeddings.
        texts = [
            "Revenue increased in the western region.",
            "Gross profit margin declined after discounts.",
            "Sales reports must be validated.",
            "Managers should investigate revenue declines.",
        ]

        print("\nStep 2: Generating embeddings...", flush=True)

        started = time.perf_counter()

        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        print(
            f"Step 2 passed: Embeddings generated in "
            f"{time.perf_counter() - started:.2f} seconds.",
            flush=True,
        )

        print(f"Embedding shape: {vectors.shape}", flush=True)
        print("\nRESULT: Embedding diagnostic passed.", flush=True)

    except Exception:
        print("\nRESULT: Embedding diagnostic failed.", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()