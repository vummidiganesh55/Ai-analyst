from __future__ import annotations

import logging
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from app.config.settings import settings
from app.graph.workflow import BIWorkflow
from app.ingestion.loader import load_dataset
from app.ingestion.validator import validate_file
from app.profiling.profiler import DataProfiler
from app.quality.quality_agent import DataQualityAgent
from app.storage.duckdb_manager import DuckDBManager

load_dotenv()
logger = logging.getLogger(__name__)


app = FastAPI(
    title=settings.app_name,
    description="API for the Autonomous AI Business Intelligence Agent",
    version="1.0.0",
)


# ---------------------------------------------------------
# EXISTING SERVICES
# ---------------------------------------------------------

# Reuse the existing analytics architecture.
database = DuckDBManager(
    database_path=settings.database_path,
)

workflow = BIWorkflow()

# Existing profiling and data-quality services.
data_profiler = DataProfiler()
data_quality_agent = DataQualityAgent()

# Serialize dataset registration to avoid concurrent table replacement.
registration_lock = Lock()


# ---------------------------------------------------------
# REQUEST MODELS
# ---------------------------------------------------------

class QuestionRequest(BaseModel):
    """Request model for business questions."""

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Business question to analyze.",
    )

    selected_table: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Optional DuckDB table name to query.",
    )


# ---------------------------------------------------------
# JSON SERIALIZATION
# ---------------------------------------------------------

def json_safe(value: Any) -> Any:
    """
    Convert common pandas and NumPy values into JSON-compatible values.
    """

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (ValueError, TypeError):
            pass

    if value is None:
        return None

    if pd.isna(value):
        return None

    return value


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

@app.get("/health")
def health_check() -> dict[str, str]:
    """Check whether the API is running."""

    return {
        "status": "healthy",
        "service": settings.app_name,
    }


# ---------------------------------------------------------
# DATASET UPLOAD
# ---------------------------------------------------------

@app.post("/datasets/upload")
def upload_dataset(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """
    Validate a supported dataset and register it in DuckDB.
    """

    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()

    allowed_extensions = {
        ".csv",
        ".xlsx",
        ".xls",
        ".parquet",
    }

    if not filename or suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Supported file types: CSV, XLSX, XLS, and Parquet."
            ),
        )

    temp_path: str | None = None

    try:
        # Stream the upload to disk instead of loading the entire
        # uploaded file into memory.
        total_bytes = 0
        max_bytes = settings.max_file_size_mb * 1024 * 1024

        with NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temporary_file:

            temp_path = temporary_file.name

            while True:
                chunk = file.file.read(1024 * 1024)

                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File exceeds the "
                            f"{settings.max_file_size_mb} MB limit."
                        ),
                    )

                temporary_file.write(chunk)

        # Validate the uploaded file.
        validate_file(
            temp_path,
            max_size_mb=settings.max_file_size_mb,
        )

        # Load the dataset into a DataFrame.
        dataframe = load_dataset(temp_path)

        if dataframe.empty:
            raise HTTPException(
                status_code=400,
                detail="The uploaded dataset contains no rows.",
            )

        # Generate a safe table name from the uploaded filename.
        table_name = Path(filename).stem.lower()

        table_name = re.sub(
            r"[^a-z0-9_]+",
            "_",
            table_name,
        )

        table_name = table_name.strip("_")

        if not table_name:
            table_name = "uploaded_dataset"

        if table_name[0].isdigit():
            table_name = f"dataset_{table_name}"

        # Register the dataset in DuckDB.
        # Register the dataset without replacing an existing table.
        with registration_lock:
            existing_tables = database.get_tables()
            if table_name in existing_tables:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Dataset '{table_name}' already exists. "
                        "Choose a different filename."
                    ),
                )

            database.register_dataframe(
                dataframe=dataframe,
                table_name=table_name,
            )

        return {
            "status": "success",
            "filename": filename,
            "table_name": table_name,
            "row_count": int(len(dataframe)),
            "column_count": int(len(dataframe.columns)),
            "columns": [
                {
                    "name": str(column),
                    "dtype": str(dataframe[column].dtype),
                }
                for column in dataframe.columns
            ],
        }
    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Dataset upload failed")

        raise HTTPException(
            status_code=400,
            detail=(
                "Dataset could not be processed: "
                f"{type(exc).__name__}."
            ),
        ) from exc

    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)

        file.file.close()


# ---------------------------------------------------------
# LIST REGISTERED DATASETS
# ---------------------------------------------------------

@app.get("/datasets")
def list_datasets() -> dict[str, Any]:
    """
    List tables currently registered in DuckDB.
    """

    try:
        return {
            "tables": database.get_tables(),
        }

    except Exception as exc:
        logger.exception("Failed to list datasets")

        raise HTTPException(
            status_code=500,
            detail="Unable to list available datasets.",
        ) from exc


# ---------------------------------------------------------
# DATASET PROFILING AND QUALITY ANALYSIS
# ---------------------------------------------------------

@app.get("/datasets/{table_name}/quality")
def get_dataset_quality(
    table_name: str,
) -> dict[str, Any]:
    """Generate profiling and data-quality reports for a registered table."""

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
        raise HTTPException(
            status_code=400,
            detail="Invalid dataset table name.",
        )

    try:
        available_tables = database.get_tables()

        if table_name not in available_tables:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{table_name}' was not found.",
            )

        dataframe = database.execute_query(
            f'SELECT * FROM "{table_name}"'
        )

        if dataframe is None:
            raise HTTPException(
                status_code=500,
                detail="Unable to retrieve dataset for quality analysis.",
            )

        profile_report = data_profiler.profile(dataframe)
        quality_report = data_quality_agent.analyze(dataframe)

        return json_safe({
            "status": "success",
            "table_name": table_name,
            "profile": profile_report,
            "quality": quality_report,
        })

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Dataset quality analysis failed for table '%s'",
            table_name,
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to generate the dataset quality report.",
        ) from exc
# ---------------------------------------------------------
# BUSINESS QUESTION / SQL ANALYSIS
# ---------------------------------------------------------

@app.post("/query")
def ask_business_question(
    request: QuestionRequest,
) -> dict[str, Any]:
    """
    Execute a business question using the existing BI workflow.
    """

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    # Normalize and validate the optional table selection.
    selected_table = request.selected_table

    if selected_table is not None:
        selected_table = selected_table.strip()

        if not selected_table:
            raise HTTPException(
                status_code=422,
                detail="Selected table cannot be empty.",
            )

    try:
        # Pass the selected table to the existing BI workflow.
        # When omitted, the workflow retains its default selection logic.
        state = workflow.invoke(
            question,
            selected_table=selected_table,
        )

        return {
            "question": question,
            "answer": state.get("final_answer", ""),
            "selected_table": state.get("selected_table", ""),
            "execution_results": json_safe(
                state.get("execution_results", [])
            ),
            "evidence": json_safe(
                state.get("evidence", [])
            ),
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Business question failed")

        raise HTTPException(
            status_code=500,
            detail=(
                "The business question could not be processed."
            ),
        ) from exc