"""
MCP server for the Autonomous AI Business Intelligence Platform.

Exposes approved dataset profiling, KPI calculation, root-cause analysis,
recommendations, and business-document retrieval through MCP.
"""

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from time import perf_counter
import logging
import re
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from app.analytics.kpi import KPIDetectionEngine
from app.agents.recommendation import RecommendationAgent
from app.agents.root_cause import RootCauseAnalyzer
from app.ingestion.dataset_manager import create_dataset_id
from app.ingestion.loader import load_dataset
from app.profiling.profiler import DataProfiler
from app.rag.rag_pipeline import RAGPipeline
from app.quality.quality_agent import DataQualityAgent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERVICE_NAME = "autonomous-bi-platform"

DATASET_DIRECTORY = Path("data/raw").resolve()
DOCUMENT_DIRECTORY = Path("data/documents").resolve()

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls",
    ".parquet",
}

DATASET_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9_. -]+_[a-f0-9]{12}$"
)


# MCP stdio transport uses stdout for protocol messages.
# All application logs must go to stderr.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)

logger = logging.getLogger(__name__)


def write_rag_trace(message: str) -> None:
    """Append temporary RAG diagnostics to a project-local log file.

    Trace-file failures are deliberately ignored so diagnostics never break
    MCP tool execution. The file is created at the project root.
    """
    trace_file = Path(__file__).resolve().parents[2] / "mcp_rag_trace.log"
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        with trace_file.open("a", encoding="utf-8") as file:
            file.write(f"{timestamp} {message}\n")
            file.flush()
    except OSError:
        logger.exception("Unable to write RAG diagnostic trace.")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def health_check() -> dict[str, Any]:
    """Return the health status of the MCP service."""
    return {
        "status": "healthy",
        "service": SERVICE_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
    }


# ---------------------------------------------------------------------------
# Dataset security and resolution
# ---------------------------------------------------------------------------

def resolve_dataset_path(dataset_id: str) -> Path:
    """
    Resolve a dataset ID inside the approved dataset directory.

    Rejects malformed IDs, unsupported extensions, symlinks, and files
    outside the approved directory.
    """
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError(
            "dataset_id must be a non-empty string."
        )

    if not DATASET_ID_PATTERN.fullmatch(dataset_id):
        raise ValueError(
            "Invalid dataset ID format."
        )

    if not DATASET_DIRECTORY.exists():
        raise FileNotFoundError(
            "Dataset directory does not exist."
        )

    if not DATASET_DIRECTORY.is_dir():
        raise ValueError(
            "Configured dataset location is not a directory."
        )

    for candidate in DATASET_DIRECTORY.iterdir():

        # Do not follow symlinks.
        if candidate.is_symlink() or not candidate.is_file():
            continue

        if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        resolved = candidate.resolve()

        # Require the resolved file to be directly inside the
        # approved directory.
        if resolved.parent != DATASET_DIRECTORY:
            continue

        if create_dataset_id(str(resolved)) == dataset_id:
            return resolved

    raise FileNotFoundError(
        f"Dataset '{dataset_id}' was not found in the approved directory."
    )


# ---------------------------------------------------------------------------
# Dataset profiling
# ---------------------------------------------------------------------------

def dataset_profile(dataset_id: str) -> dict[str, Any]:
    """Return a structural profile for an approved dataset."""
    try:
        path = resolve_dataset_path(dataset_id)

        dataframe = load_dataset(str(path))

        profile = DataProfiler().profile(dataframe)

        return {
            "status": "success",
            "dataset_id": dataset_id,
            "dataset_name": path.name,
            "profile": profile,
        }

    except (ValueError, FileNotFoundError) as exc:
        return {
            "status": "error",
            "error": str(exc),
        }

    except Exception:
        logger.exception(
            "Dataset profiling failed."
        )

        return {
            "status": "error",
            "error": "Dataset profiling failed.",
        }


# ---------------------------------------------------------------------------
# KPI calculation
# ---------------------------------------------------------------------------

def calculate_kpis(dataset_id: str) -> dict[str, Any]:
    """Calculate deterministic KPIs for an approved dataset."""
    try:
        path = resolve_dataset_path(dataset_id)

        dataframe = load_dataset(str(path))

        kpis = KPIDetectionEngine(dataframe).to_dict()

        return {
            "status": "success",
            "dataset_id": dataset_id,
            "dataset_name": path.name,
            "kpi_count": len(kpis),
            "kpis": kpis,
        }

    except (ValueError, FileNotFoundError) as exc:
        return {
            "status": "error",
            "error": str(exc),
        }

    except Exception:
        logger.exception(
            "KPI calculation failed."
        )

        return {
            "status": "error",
            "error": "KPI calculation failed.",
        }


# ---------------------------------------------------------------------------
# Root-cause analysis
# ---------------------------------------------------------------------------

def analyze_root_cause(
    dataset_id: str,
    metric: str,
    dimension: str,
    period_column: str,
    previous_period: Any,
    current_period: Any,
) -> dict[str, Any]:
    """
    Analyze observed contributions to a metric's change between periods.

    The analysis identifies observed changes and contributions. It does
    not independently establish causal relationships.
    """
    try:
        dataset_path = resolve_dataset_path(dataset_id)

        dataframe = load_dataset(str(dataset_path))

        analyzer = RootCauseAnalyzer(
            dataframe=dataframe,
            dimension_columns=[dimension],
            metric_column=metric,
            period_column=period_column,
            previous_period=previous_period,
            current_period=current_period,
        )

        result = analyzer.analyze()

        return {
            "status": "success",
            "dataset_id": dataset_id,
            "result": result,
        }

    except (ValueError, FileNotFoundError, TypeError) as exc:
        return {
            "status": "error",
            "error": str(exc),
        }

    except Exception:
        logger.exception(
            "Root-cause analysis failed."
        )

        return {
            "status": "error",
            "error": "Root-cause analysis failed unexpectedly.",
        }


# ---------------------------------------------------------------------------
# Recommendation generation
# ---------------------------------------------------------------------------

def generate_recommendations(
    insights: list[dict[str, Any]] | None = None,
    root_cause: dict[str, Any] | None = None,
    max_recommendations: int = 10,
) -> dict[str, Any]:
    """
    Generate recommendations from supplied analytical evidence.

    This tool does not independently verify caller-supplied evidence.
    Use trusted outputs from the BI analytics components.
    """
    try:
        agent = RecommendationAgent(
            max_recommendations=max_recommendations
        )

        report = agent.generate_report(
            insights=insights,
            root_cause=root_cause,
        )

        return {
            "status": "success",
            **report,
        }

    except (ValueError, TypeError) as exc:
        return {
            "status": "error",
            "error": str(exc),
        }

    except Exception:
        logger.exception(
            "Recommendation generation failed."
        )

        return {
            "status": "error",
            "error": "Recommendation generation failed.",
        }


# ---------------------------------------------------------------------------
# RAG pipeline initialization and caching
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def get_rag_pipeline(
    pipeline_class: Any,
    document_directory: str,
) -> tuple[Any, dict[str, Any]]:
    """
    Create and cache a RAG pipeline and its ingestion result.

    The pipeline class and directory are part of the cache key. This avoids
    reusing a pipeline when tests substitute a different implementation
    or use a different document directory.

    The cache exists only for the lifetime of this Python process.
    """
    directory = Path(document_directory)

    if not directory.exists():
        raise FileNotFoundError(
            "Business-document directory does not exist."
        )

    if not directory.is_dir():
        raise ValueError(
            "Configured business-document location is not a directory."
        )

    initialization_started = perf_counter()
    write_rag_trace(
        f"TRACE 2: Initializing pipeline; directory={directory}; "
        f"class={getattr(pipeline_class, '__name__', repr(pipeline_class))}"
    )
    logger.warning(
        "[RAG DEBUG 2] Initializing RAG pipeline. directory=%s class=%s",
        directory,
        getattr(pipeline_class, "__name__", repr(pipeline_class)),
    )

    pipeline = pipeline_class()
    write_rag_trace("TRACE 2A: Pipeline object created")

    logger.warning(
        "[RAG DEBUG 2A] RAG pipeline object created in %.2f seconds.",
        perf_counter() - initialization_started,
    )

    ingestion_started = perf_counter()
    write_rag_trace("TRACE 2B: Starting document ingestion")
    logger.warning("[RAG DEBUG 2B] Starting business-document ingestion.")

    ingestion_result = pipeline.ingest_directory(
        str(directory)
    )
    write_rag_trace(
        f"TRACE 2C: Document ingestion returned; result={ingestion_result}"
    )

    logger.warning(
        "[RAG DEBUG 2C] Document ingestion returned in %.2f seconds.",
        perf_counter() - ingestion_started,
    )

    if not isinstance(ingestion_result, dict):
        raise RuntimeError(
            "Document ingestion returned an invalid result."
        )

    logger.warning(
        "[RAG DEBUG 3] Business-document ingestion completed: %s",
        ingestion_result,
    )

    logger.warning(
        "[RAG DEBUG 3A] Total pipeline setup time: %.2f seconds.",
        perf_counter() - initialization_started,
    )

    return pipeline, ingestion_result


# ---------------------------------------------------------------------------
# Business-document search
# ---------------------------------------------------------------------------

def search_business_documents(
    query: str,
    top_k: int = 3,
) -> dict[str, Any]:
    """Retrieve relevant business-document evidence for a question."""
    request_started = perf_counter()
    write_rag_trace("TRACE 1: RAG function entered")
    logger.warning(
        "[RAG DEBUG 1] MCP RAG tool entered. top_k=%r query_length=%s",
        top_k,
        len(query) if isinstance(query, str) else "invalid",
    )

    try:
        # Validate query.
        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "query must be a non-empty string."
            )

        # Reject bool explicitly because bool is a subclass of int.
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= 10
        ):
            raise ValueError(
                "top_k must be between 1 and 10."
            )

        logger.warning("[RAG DEBUG 1A] Input validation passed.")

        # Obtain the cached pipeline and ingestion result.
        write_rag_trace("TRACE 2: Requesting cached RAG pipeline")
        logger.warning("[RAG DEBUG 1B] Requesting cached RAG pipeline.")
        pipeline, ingestion_result = get_rag_pipeline(
            RAGPipeline,
            str(DOCUMENT_DIRECTORY),
        )
        write_rag_trace(
            f"TRACE 3: Pipeline ready; ingestion={ingestion_result}"
        )
        logger.warning(
            "[RAG DEBUG 1C] Pipeline ready; ingestion_result=%s",
            ingestion_result,
        )

        # Handle an empty index without calling pipeline.answer().
        if ingestion_result.get("chunks", 0) == 0:
            logger.info(
                "No business-document chunks were indexed."
            )

            return {
                "status": "success",
                "query": query.strip(),
                "answer": (
                    "No business-document evidence was indexed."
                ),
                "sources": [],
                "evidence": [],
                "answer_generated": False,
            }

        answer_started = perf_counter()
        write_rag_trace("TRACE 4: Starting pipeline.answer")
        logger.warning(
            "[RAG DEBUG 4] Calling pipeline.answer(); query=%r top_k=%d",
            query.strip()[:160],
            top_k,
        )

        result = pipeline.answer(
            query.strip(),
            top_k=top_k,
        )
        write_rag_trace("TRACE 5: pipeline.answer completed")

        logger.warning(
            "[RAG DEBUG 5] pipeline.answer() returned after %.2f seconds.",
            perf_counter() - answer_started,
        )

        if not isinstance(result, dict):
            raise RuntimeError(
                "Document search returned an invalid result."
            )

        response = {
            "status": "success",
            **result,
        }
        logger.warning(
            "[RAG DEBUG 6] MCP RAG tool returning success after %.2f seconds.",
            perf_counter() - request_started,
        )
        write_rag_trace("TRACE 6: RAG function returning success")
        return response

    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        write_rag_trace(
            f"TRACE ERROR: Expected RAG error {type(exc).__name__}: {exc}"
        )
        logger.warning(
            "[RAG DEBUG ERROR] Business-document search failed after %.2f seconds: %s",
            perf_counter() - request_started,
            exc,
        )

        return {
            "status": "error",
            "error": str(exc),
        }

    except Exception as exc:
        write_rag_trace(
            f"TRACE ERROR: Unexpected RAG error {type(exc).__name__}: {exc}"
        )
        logger.exception(
            "[RAG DEBUG ERROR] Unexpected business-document search failure after %.2f seconds.",
            perf_counter() - request_started,
        )

        return {
            "status": "error",
            "error": (
                "Document search failed. Check the MCP server logs "
                "for the underlying exception."
            ),
        }

def search_business_documents_lexical(
    query: str,
    top_k: int = 3,
) -> dict[str, Any]:
    """
    Retrieve business-document evidence using lightweight keyword matching.

    This temporary MCP implementation does not load an embedding model
    or claim to generate an LLM answer.
    """
    if not isinstance(query, str) or not query.strip():
        return {
            "status": "error",
            "error": "query must be a non-empty string.",
        }

    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or not 1 <= top_k <= 10
    ):
        return {
            "status": "error",
            "error": "top_k must be between 1 and 10.",
        }

    if not DOCUMENT_DIRECTORY.is_dir():
        return {
            "status": "error",
            "error": "Business-document directory does not exist.",
        }

    # Read only supported text-based business documents.
    supported_extensions = {".txt", ".md"}
    query_terms = set(re.findall(r"\b[a-zA-Z0-9_]+\b", query.lower()))

    if not query_terms:
        return {
            "status": "error",
            "error": "Query contains no searchable terms.",
        }

    evidence = []

    for path in sorted(DOCUMENT_DIRECTORY.iterdir()):
        if (
            not path.is_file()
            or path.is_symlink()
            or path.suffix.lower() not in supported_extensions
        ):
            continue

        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            logger.exception("Could not read business document: %s", path.name)
            continue

        # Split documents into paragraph-sized passages.
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", content)
            if paragraph.strip()
        ]

        for index, paragraph in enumerate(paragraphs):
            passage_terms = set(
                re.findall(r"\b[a-zA-Z0-9_]+\b", paragraph.lower())
            )

            matched_terms = query_terms.intersection(passage_terms)

            if not matched_terms:
                continue

            evidence.append({
                "chunk_id": f"{path.name}:{index}",
                "source": path.name,
                "text": paragraph,
                "score": round(
                    len(matched_terms) / len(query_terms),
                    4,
                ),
            })

    evidence.sort(
        key=lambda item: (
            item["score"],
            len(item["text"]),
        ),
        reverse=True,
    )

    selected = evidence[:top_k]

    return {
        "status": "success",
        "query": query.strip(),
        "answer": (
            "Relevant business-document passages were retrieved "
            "using keyword matching. No LLM answer was generated."
            if selected
            else "No matching business-document passages were found."
        ),
        "sources": list(dict.fromkeys(
            item["source"] for item in selected
        )),
        "evidence": selected,
        "answer_generated": False,
        "retrieval_method": "lexical",
    }

# ---------------------------------------------------------------------------
# MCP server registration
# ---------------------------------------------------------------------------

def create_mcp_server() -> FastMCP:
    """Create the MCP server and register all six BI tools."""
    server = FastMCP(SERVICE_NAME)

    server.tool()(health_check)
    server.tool()(dataset_profile)
    server.tool()(calculate_kpis)
    server.tool()(analyze_root_cause)
    server.tool()(generate_recommendations)
    server.tool()(data_quality_check)
    server.tool(name="search_business_documents")(
    search_business_documents_lexical
    )

    return server

# ---------------------------------------------------------------------------
# Data quality analysis
# ---------------------------------------------------------------------------

def data_quality_check(dataset_id: str) -> dict[str, Any]:
    """Analyze data quality for an approved dataset."""

    try:
        # Resolve and validate the dataset path.
        path = resolve_dataset_path(dataset_id)

        # Load the approved dataset.
        dataframe = load_dataset(str(path))

        # Run data quality analysis.
        quality_report = DataQualityAgent().analyze(dataframe)

        return {
            "status": "success",
            "dataset_id": dataset_id,
            "dataset_name": path.name,
            "quality_report": quality_report,
        }

    except (ValueError, FileNotFoundError) as exc:
        return {
            "status": "error",
            "error": str(exc),
        }

    except Exception:
        logger.exception("Data quality analysis failed.")

        return {
            "status": "error",
            "error": "Data quality analysis failed.",
        }

# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

def run_server() -> None:
    """Start the MCP server using the SDK's default transport."""
    logger.info(
        "Starting %s MCP server.",
        SERVICE_NAME,
    )

    create_mcp_server().run()


if __name__ == "__main__":
    run_server()