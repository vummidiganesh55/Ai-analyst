from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from app.api.main import app, database


client = TestClient(app)


def test_upload_rejects_unsupported_file_type():
    response = client.post(
        "/datasets/upload",
        files={
            "file": (
                "malicious.txt",
                b"not a supported dataset",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400


def test_upload_rejects_empty_filename():
    response = client.post(
        "/datasets/upload",
        files={
            "file": (
                "",
                b"sample data",
                "text/csv",
            )
        },
    )

    assert response.status_code in (400, 422)


def test_upload_does_not_replace_existing_table(monkeypatch):
    table_name = "upload_security_test"

    existing_data = pd.DataFrame({"value": [100]})
    database.register_dataframe(existing_data, table_name)

    response = client.post(
        "/datasets/upload",
        files={
            "file": (
                "upload_security_test.csv",
                b"value\n200\n",
                "text/csv",
            )
        },
    )

    assert response.status_code in (400, 409)

    result = database.execute_query(
        f'SELECT value FROM "{table_name}"'
    )

    assert result["value"].tolist() == [100]