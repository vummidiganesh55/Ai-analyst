from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

API_BASE_URL = os.getenv(
    "NEXORA_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

APP_NAME = "NEXORA AI"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# API CLIENT
# ---------------------------------------------------------

class APIError(Exception):
    """Raised when a backend API request fails."""


def api_request(
    method: str,
    endpoint: str,
    *,
    timeout: int = 60,
    **kwargs: Any,
) -> Any:
    """Send a request to the existing FastAPI backend."""

    url = f"{API_BASE_URL}{endpoint}"

    try:
        response = requests.request(
            method=method,
            url=url,
            timeout=timeout,
            **kwargs,
        )

        try:
            payload = response.json()
        except ValueError:
            payload = response.text

        if not response.ok:
            if isinstance(payload, dict):
                detail = payload.get("detail", payload)
            else:
                detail = payload

            raise APIError(
                f"API returned HTTP {response.status_code}: {detail}"
            )

        return payload

    except requests.ConnectionError as exc:
        raise APIError(
            f"Cannot connect to the backend at {API_BASE_URL}. "
            "Start the FastAPI server and try again."
        ) from exc

    except requests.Timeout as exc:
        raise APIError(
            "The backend request timed out. Try again or use a smaller dataset."
        ) from exc


def get_health() -> dict[str, Any]:
    return api_request("GET", "/health", timeout=10)


def get_datasets() -> list[str]:
    response = api_request("GET", "/datasets", timeout=20)

    if isinstance(response, dict):
        tables = response.get("tables", [])
        return tables if isinstance(tables, list) else []

    return []


def upload_dataset(uploaded_file: Any) -> dict[str, Any]:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }

    return api_request(
        "POST",
        "/datasets/upload",
        files=files,
        timeout=120,
    )


def get_quality(table_name: str) -> dict[str, Any]:
    return api_request(
        "GET",
        f"/datasets/{table_name}/quality",
        timeout=60,
    )


def ask_question(
    question: str,
    selected_table: str,
) -> dict[str, Any]:
    return api_request(
        "POST",
        "/query",
        json={
            "question": question,
            "selected_table": selected_table,
        },
        timeout=180,
    )


# ---------------------------------------------------------
# STYLING
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2rem;
        }

        .brand-title {
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: 0.03em;
        }

        .brand-subtitle {
            font-size: 1rem;
            opacity: 0.75;
            margin-bottom: 1.5rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.25);
            padding: 1rem;
            border-radius: 0.75rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:
    st.markdown("## 📊 NEXORA AI")
    st.caption("Autonomous Business Intelligence")

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Dataset Management",
            "Data Quality",
            "AI Data Analyst",
            "System Health",
        ],
        key="navigation",
    )

    st.divider()
    st.caption("Backend configuration")
    st.code(API_BASE_URL, language=None)

    if st.button("🔄 Refresh application", use_container_width=True):
        st.rerun()


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="brand-title">NEXORA AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="brand-subtitle">'
    "Turn your datasets into actionable business insights."
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------

if page == "Dashboard":
    st.header("Business Intelligence Dashboard")

    try:
        health = get_health()
        datasets = get_datasets()

        health_status = health.get("status", "unknown")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Backend Status",
            "Online" if health_status == "healthy" else health_status,
        )

        col2.metric(
            "Registered Datasets",
            len(datasets),
        )

        col3.metric(
            "Analytics Platform",
            "Ready" if health_status == "healthy" else "Check backend",
        )

        st.subheader("Available Datasets")

        if datasets:
            st.dataframe(
                pd.DataFrame({"Dataset / Table": datasets}),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No datasets are registered. Open Dataset Management "
                "to upload your first dataset."
            )

    except APIError as exc:
        st.error(str(exc))
        st.info(
            "Start the FastAPI backend before using the dashboard."
        )

    st.divider()

    st.subheader("Platform Capabilities")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📁 Dataset Management**")
        st.write("Upload and manage supported business datasets.")

        st.markdown("**🔎 Data Quality**")
        st.write("Inspect missing values, duplicates, and data types.")

    with col2:
        st.markdown("**🤖 AI Data Analyst**")
        st.write("Ask business questions using natural language.")

        st.markdown("**⚙️ System Health**")
        st.write("Check backend connectivity and service status.")


# ---------------------------------------------------------
# DATASET MANAGEMENT
# ---------------------------------------------------------

elif page == "Dataset Management":
    st.header("Dataset Management")

    st.write(
        "Upload CSV, Excel, or Parquet files to register them "
        "in the existing DuckDB backend."
    )

    uploaded_file = st.file_uploader(
        "Choose a dataset",
        type=["csv", "xlsx", "xls", "parquet"],
        help="The backend applies its own file validation and size limits.",
    )

    if uploaded_file is not None:
        st.write(f"**Filename:** {uploaded_file.name}")
        st.write(f"**File size:** {uploaded_file.size:,} bytes")

        if st.button("Upload dataset", type="primary"):
            with st.spinner("Validating and registering dataset..."):
                try:
                    result = upload_dataset(uploaded_file)

                    st.success("Dataset uploaded successfully.")

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Rows",
                        result.get("row_count", 0),
                    )

                    col2.metric(
                        "Columns",
                        result.get("column_count", 0),
                    )

                    col3.metric(
                        "Table",
                        result.get("table_name", "Unknown"),
                    )

                    columns = result.get("columns", [])

                    if columns:
                        st.subheader("Column Information")
                        st.dataframe(
                            pd.DataFrame(columns),
                            use_container_width=True,
                            hide_index=True,
                        )

                    st.session_state["last_uploaded_table"] = (
                        result.get("table_name", "")
                    )

                except APIError as exc:
                    st.error(str(exc))

    st.divider()

    st.subheader("Registered Datasets")

    if st.button("Load registered datasets"):
        st.session_state["refresh_datasets"] = True

    try:
        datasets = get_datasets()

        if datasets:
            st.dataframe(
                pd.DataFrame({"Dataset / Table": datasets}),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No datasets are currently registered.")

    except APIError as exc:
        st.error(str(exc))


# ---------------------------------------------------------
# DATA QUALITY
# ---------------------------------------------------------

elif page == "Data Quality":
    st.header("Data Quality Assessment")

    try:
        datasets = get_datasets()

        if not datasets:
            st.info("Upload a dataset before running quality analysis.")
        else:
            preferred_table = st.session_state.get(
                "last_uploaded_table",
                "",
            )

            default_index = (
                datasets.index(preferred_table)
                if preferred_table in datasets
                else 0
            )

            selected_table = st.selectbox(
                "Select dataset",
                datasets,
                index=default_index,
            )

            if st.button("Run quality assessment", type="primary"):
                with st.spinner("Profiling dataset and calculating quality..."):
                    try:
                        report = get_quality(selected_table)

                        profile = report.get("profile", {})
                        quality = report.get("quality", {})

                        st.session_state["quality_report"] = report
                        st.session_state["quality_table"] = selected_table

                    except APIError as exc:
                        st.error(str(exc))

            report = st.session_state.get("quality_report")

            if (
                report
                and st.session_state.get("quality_table") == selected_table
            ):
                profile = report.get("profile", {})
                quality = report.get("quality", {})

                score = quality.get("quality_score")

                st.subheader(f"Report: {selected_table}")

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Quality Score",
                    f"{score}/100" if score is not None else "N/A",
                )

                col2.metric(
                    "Rows",
                    profile.get("row_count", 0),
                )

                col3.metric(
                    "Columns",
                    profile.get("column_count", 0),
                )

                col4.metric(
                    "Duplicate Rows",
                    quality.get("duplicate_rows", 0),
                )

                missing_values = quality.get(
                    "missing_values",
                    0,
                )

                st.subheader("Quality Issues")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Total Missing Values",
                        missing_values,
                    )

                with col2:
                    st.metric(
                        "Duplicate Rows",
                        quality.get("duplicate_rows", 0),
                    )

                negative_values = quality.get("negative_values", {})

                st.subheader("Negative Values by Column")

                if negative_values:
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Column": column,
                                    "Negative Values": count,
                                }
                                for column, count in negative_values.items()
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.success(
                        "No negative numeric values were reported."
                    )

                st.subheader("Missing Values by Column")

                missing_by_column = profile.get("missing_values", {})

                if missing_by_column:
                    missing_df = pd.DataFrame(
                        [
                            {
                                "Column": column,
                                "Missing Values": count,
                            }
                            for column, count in missing_by_column.items()
                        ]
                    )

                    st.dataframe(
                        missing_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                st.subheader("Column Profile")

                st.write("**Numeric columns**")
                st.write(profile.get("numeric_columns", []))

                st.write("**Categorical columns**")
                st.write(profile.get("categorical_columns", []))

                st.write("**Detected date columns**")
                st.write(profile.get("date_columns", []))

                st.write("**Data types**")

                dtypes = profile.get("dtypes", {})

                if dtypes:
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {"Column": column, "Data Type": dtype}
                                for column, dtype in dtypes.items()
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                st.caption(
                    "The quality score uses the backend's configured "
                    "custom formula. It is not a universal industry standard."
                )

                st.download_button(
                    "Download quality report (JSON)",
                    data=__import__("json").dumps(
                        report,
                        indent=2,
                        ensure_ascii=False,
                    ),
                    file_name=f"{selected_table}_quality_report.json",
                    mime="application/json",
                )

    except APIError as exc:
        st.error(str(exc))


# ---------------------------------------------------------
# AI DATA ANALYST
# ---------------------------------------------------------

elif page == "AI Data Analyst":
    st.header("AI Data Analyst")

    st.write(
        "Ask a question about your business data in natural language."
    )

    try:
        datasets = get_datasets()

        if not datasets:
            st.info("Upload a dataset before asking analytical questions.")
        else:
            preferred_table = st.session_state.get(
                "last_uploaded_table",
                "",
            )

            default_index = (
                datasets.index(preferred_table)
                if preferred_table in datasets
                else 0
            )

            selected_table = st.selectbox(
                "Dataset to analyze",
                datasets,
                index=default_index,
            )

            st.markdown("**Example questions**")

            examples = [
                "What is the total revenue?",
                "What is the total profit?",
                "What is the average revenue?",
                "Which products perform best?",
                "Compare revenue by region.",
            ]

            example = st.selectbox(
                "Choose an example or enter your own question below",
                ["Custom question"] + examples,
            )

            question = st.text_area(
                "Your business question",
                value="" if example == "Custom question" else example,
                max_chars=2000,
                placeholder="For example: Which region has the highest revenue?",
            )

            if st.button("Analyze", type="primary"):
                if not question.strip():
                    st.warning("Enter a business question first.")
                else:
                    with st.spinner("Analyzing your question..."):
                        try:
                            result = ask_question(
                                question.strip(),
                                selected_table,
                            )

                            st.session_state["last_query_result"] = result

                        except APIError as exc:
                            st.error(str(exc))

            result = st.session_state.get("last_query_result")

            if result:
                st.divider()
                st.subheader("Analysis Result")

                st.markdown("**Question**")
                st.write(result.get("question", ""))

                st.markdown("**Answer**")
                st.write(
                    result.get("answer")
                    or "The workflow did not return a final answer."
                )

                st.caption(
                    "Selected table: "
                    f"{result.get('selected_table', 'Not reported')}"
                )

                execution_results = result.get("execution_results", [])

                if execution_results:
                    st.subheader("Execution Results")

                    if isinstance(execution_results, list):
                        for index, item in enumerate(execution_results, 1):
                            with st.expander(f"Result {index}"):
                                if isinstance(item, (dict, list)):
                                    st.json(item)
                                else:
                                    st.write(item)
                    else:
                        st.json(execution_results)

                evidence = result.get("evidence", [])

                if evidence:
                    with st.expander("View evidence"):
                        st.json(evidence)

                st.download_button(
                    "Download analysis result (JSON)",
                    data=__import__("json").dumps(
                        result,
                        indent=2,
                        ensure_ascii=False,
                    ),
                    file_name="nexora_analysis_result.json",
                    mime="application/json",
                )

    except APIError as exc:
        st.error(str(exc))


# ---------------------------------------------------------
# SYSTEM HEALTH
# ---------------------------------------------------------

elif page == "System Health":
    st.header("System Health")

    if st.button("Check backend health", type="primary"):
        try:
            health = get_health()

            st.session_state["health_result"] = health
            st.success("Backend is reachable.")

        except APIError as exc:
            st.session_state.pop("health_result", None)
            st.error(str(exc))

    health = st.session_state.get("health_result")

    if health:
        st.json(health)

    st.markdown("**Backend URL**")
    st.code(API_BASE_URL, language=None)

    st.caption(
        "This page checks API connectivity. It does not claim to measure "
        "the health of every internal service."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "NEXORA AI | Autonomous AI Business Intelligence & Data Analytics"
)