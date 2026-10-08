from app.ingestion.validator import validate_file
from app.ingestion.loader import load_dataset
from app.ingestion.dataset_manager import create_dataset_id
from app.storage.duckdb_manager import DuckDBManager


def test_ingestion_pipeline():

    path = "data/raw/sales.csv"

    validate_file(path)

    dataframe = load_dataset(path)

    assert not dataframe.empty
    assert "revenue" in dataframe.columns

    dataset_id = create_dataset_id(path)

    assert dataset_id.startswith("sales_")

    db = DuckDBManager(
        "data/database/test.duckdb"
    )

    db.register_dataframe(
        dataframe,
        "sales",
    )

    tables = db.get_tables()

    assert "sales" in tables

    result = db.execute_query(
        "SELECT COUNT(*) AS total FROM sales"
    )

    assert result.iloc[0]["total"] == len(dataframe)