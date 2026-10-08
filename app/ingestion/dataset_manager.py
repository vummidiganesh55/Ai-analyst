import hashlib
from pathlib import Path


def calculate_file_hash(path: str) -> str:
    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def create_dataset_id(path: str) -> str:
    file_path = Path(path)

    file_hash = calculate_file_hash(path)

    return f"{file_path.stem}_{file_hash[:12]}"