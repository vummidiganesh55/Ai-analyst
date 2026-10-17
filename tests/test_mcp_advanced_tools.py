import pandas as pd
import pytest

import app.integrations.mcp_server as mcp_module


@pytest.fixture
def dataset_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        mcp_module,
        "DATASET_DIRECTORY",
        tmp_path.resolve(),
    )
    return tmp_path


@pytest.fixture
def sales_dataset(dataset_directory):
    path = dataset_directory / "sales.csv"

    pd.DataFrame(
        {
            "period": ["January", "January", "February", "February"],
            "region": ["South", "North", "South", "North"],
            "revenue": [100, 200, 80, 220],
            "cost": [60, 100, 50, 110],
        }
    ).to_csv(path, index=False)

    dataset_id = mcp_module.create_dataset_id(str(path))
    return path, dataset_id


def test_root_cause_rejects_invalid_dataset():
    result = mcp_module.analyze_root_cause(
        dataset_id="../invalid.csv",
        metric="revenue",
        dimension="region",
        period_column="period",
        previous_period="January",
        current_period="February",
    )

    assert result["status"] == "error"


def test_root_cause_rejects_same_period(sales_dataset):
    _, dataset_id = sales_dataset

    result = mcp_module.analyze_root_cause(
        dataset_id=dataset_id,
        metric="revenue",
        dimension="region",
        period_column="period",
        previous_period="January",
        current_period="January",
    )

    assert result["status"] == "error"
    assert "must differ" in result["error"]


def test_root_cause_calls_existing_analyzer(monkeypatch):
    import app.integrations.mcp_server as mcp_server

    class FakeAnalyzer:
        def __init__(
            self,
            dataframe,
            dimension_columns,
            metric_column,
            period_column,
            previous_period,
            current_period,
        ):
            self.dataframe = dataframe
            self.dimension_columns = dimension_columns
            self.metric_column = metric_column
            self.period_column = period_column
            self.previous_period = previous_period
            self.current_period = current_period

        def analyze(self):
            return {
                "metric": self.metric_column,
                "overall": {
                    "previous_value": 100.0,
                    "current_value": 120.0,
                    "change": 20.0,
                    "change_pct": 20.0,
                },
                "dimensions": {},
                "interpretation": (
                    "Observed contributions do not establish causation."
                ),
            }

    monkeypatch.setattr(
        mcp_server,
        "RootCauseAnalyzer",
        FakeAnalyzer,
    )

    # Replace dataset access with deterministic test data.
    monkeypatch.setattr(
        mcp_server,
        "resolve_dataset_path",
        lambda dataset_id: __import__("pathlib").Path("fake.csv"),
    )

    import pandas as pd

    monkeypatch.setattr(
        mcp_server,
        "load_dataset",
        lambda path: pd.DataFrame(
            {
                "region": ["East", "East"],
                "revenue": [100.0, 120.0],
                "month": ["2026-01", "2026-02"],
            }
        ),
    )

    result = mcp_server.analyze_root_cause(
        dataset_id="sales_dataset_123456abcdef",
        metric="revenue",
        dimension="region",
        period_column="month",
        previous_period="2026-01",
        current_period="2026-02",
    )

    assert result["status"] == "success"
    assert result["result"]["metric"] == "revenue"

def test_recommendation_tool_returns_report():
    result = mcp_module.generate_recommendations(
        insights=[
            {
                "title": "Revenue decline",
                "observation": "Revenue declined.",
                "category": "revenue",
                "evidence": {"change": -10},
            }
        ]
    )

    assert result["status"] == "success"
    assert result["recommendation_count"] >= 1


def test_recommendation_tool_accepts_root_cause():
    result = mcp_module.generate_recommendations(
        root_cause={
            "metric": "revenue",
            "overall": {
                "previous_value": 100,
                "current_value": 80,
                "change": -20,
            },
        }
    )

    assert result["status"] == "success"
    assert result["recommendation_count"] >= 1


def test_recommendation_tool_rejects_invalid_limit():
    result = mcp_module.generate_recommendations(
        max_recommendations=0
    )

    assert result["status"] == "error"


def test_document_search_rejects_empty_query():
    result = mcp_module.search_business_documents("  ")

    assert result["status"] == "error"


def test_document_search_rejects_invalid_top_k():
    result = mcp_module.search_business_documents(
        query="sales policy",
        top_k=0,
    )

    assert result["status"] == "error"


def test_document_search_returns_error_when_directory_missing(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        mcp_module,
        "DOCUMENT_DIRECTORY",
        tmp_path / "missing",
    )

    result = mcp_module.search_business_documents(
        "sales policy"
    )

    assert result["status"] == "error"


def test_document_search_returns_empty_evidence_when_no_chunks(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        mcp_module,
        "DOCUMENT_DIRECTORY",
        tmp_path,
    )

    class FakeRAGPipeline:
        def ingest_directory(self, directory):
            return {"documents": 0, "chunks": 0}

    monkeypatch.setattr(
        mcp_module,
        "RAGPipeline",
        FakeRAGPipeline,
    )

    result = mcp_module.search_business_documents(
        "sales policy"
    )

    assert result["status"] == "success"
    assert result["evidence"] == []
    assert result["answer_generated"] is False


def test_document_search_returns_retrieved_evidence(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        mcp_module,
        "DOCUMENT_DIRECTORY",
        tmp_path,
    )

    class FakeRAGPipeline:
        def ingest_directory(self, directory):
            return {"documents": 1, "chunks": 1}

        def answer(self, query, top_k):
            return {
                "query": query,
                "answer": "The sales policy requires validation.",
                "sources": ["sales_policy.txt"],
                "evidence": [
                    {
                        "source": "sales_policy.txt",
                        "text": "Validate sales reports.",
                        "score": 0.9,
                    }
                ],
                "answer_generated": False,
            }

    monkeypatch.setattr(
        mcp_module,
        "RAGPipeline",
        FakeRAGPipeline,
    )

    result = mcp_module.search_business_documents(
        "sales policy",
        top_k=2,
    )

    assert result["status"] == "success"
    assert result["sources"] == ["sales_policy.txt"]
    assert len(result["evidence"]) == 1


def test_mcp_server_creation_still_succeeds():
    server = mcp_module.create_mcp_server()

    assert server is not None