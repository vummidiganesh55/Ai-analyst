from pathlib import Path


ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet"}


def validate_file(path: str, max_size_mb: int = 50) -> None:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {file_path.suffix}. "
            f"Allowed: {ALLOWED_EXTENSIONS}"
        )

    max_size_bytes = max_size_mb * 1024 * 1024

    if file_path.stat().st_size > max_size_bytes:
        raise ValueError(
            f"File exceeds maximum size of {max_size_mb} MB"
        )