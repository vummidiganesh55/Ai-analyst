# 🚀 NEXORA AI — Autonomous AI Business Intelligence & Data Analytics Agent

<p align="center">
  <strong>Transform business datasets into actionable insights using AI-powered analytics.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/LangGraph-Agent%20Workflow-1C3C3C" alt="LangGraph">
  <img src="https://img.shields.io/badge/DuckDB-Analytics-FFF000" alt="DuckDB">
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker" alt="Docker">
  <img src="https://img.shields.io/badge/Tests-440%20Passed-brightgreen" alt="Tests">
</p>

---

## 📌 Description

**NEXORA AI** is an AI-powered Business Intelligence and Data Analytics platform designed to simplify dataset exploration, data quality assessment, SQL-based analysis, and business insight generation.

The platform combines a FastAPI backend, Streamlit frontend, LangGraph-based agent workflow, DuckDB analytical storage, and modular analytics components.

Users can upload business datasets, inspect data quality, ask business questions, calculate KPIs, explore statistical insights, detect anomalies, and evaluate analytical workflows through a unified application.

The project also includes Docker-based deployment, Model Context Protocol (MCP) integration, retrieval-augmented generation (RAG) components, and LangSmith tracing.

## 🎯 Problem Statement

Traditional business intelligence workflows can require significant manual effort and technical expertise.

Common challenges include:

- Writing SQL queries to answer business questions.
- Identifying missing values, duplicate records, and data inconsistencies.
- Calculating KPIs and analyzing business performance.
- Detecting anomalies and understanding potential root causes.
- Exploring datasets without building reports manually.
- Testing and evaluating AI-assisted analytical workflows.
- Maintaining separate tools for data processing, analytics, and visualization.

NEXORA AI aims to bring these analytical capabilities together in a modular, AI-assisted platform.

## 💡 Solution

NEXORA AI integrates dataset ingestion, profiling, data quality checks, SQL analytics, statistical analysis, and AI agent orchestration.

The application follows this general process:

1. Upload and register a dataset.
2. Inspect its structure and data quality.
3. Submit a business question through the interface.
4. Coordinate analytical operations through the agent workflow.
5. Execute validated SQL queries and analytical operations.
6. Return results and supporting evidence.
7. Evaluate selected workflows using a dedicated evaluation framework.

The platform is designed to make analytical workflows more accessible while retaining structured backend processing and testable components.

## 🚀 Features

### 📂 Dataset Management
- Upload CSV, Excel, and Parquet files.
- Validate uploaded datasets.
- Register datasets in DuckDB.
- List available datasets through the API and frontend.

### 🔍 Data Profiling and Quality
- Inspect dataset columns and data types.
- Analyze missing values.
- Identify duplicate records.
- Inspect numerical and categorical columns.
- Generate data-quality reports.

### 🤖 AI Agent Workflow
- Planning and analytical task coordination.
- Business insight generation.
- Recommendation generation.
- Root-cause analysis.
- Self-correction workflow components.

### 🧮 SQL Analytics
- SQL-assisted business analysis.
- SQL query validation.
- Controlled query execution.
- Aggregations and analytical calculations.
- Structured execution results.

### 📊 Business Intelligence
- KPI calculations.
- Statistical analysis.
- Anomaly detection.
- Forecasting.
- Data visualization.

### 📚 Retrieval-Augmented Generation
- Document loading.
- Text chunking.
- Embedding generation.
- Vector storage and retrieval components.

### 🔌 API and Integrations
- FastAPI REST endpoints.
- Interactive Swagger API documentation.
- Model Context Protocol integration.
- Redis integration.
- Ollama-based LLM integration.

### 📈 Evaluation and Observability
- Automated evaluation cases.
- Task success measurement.
- Execution success measurement.
- Error-rate measurement.
- Latency measurement.
- LangSmith workflow tracing.

### 🐳 Deployment
- Separate backend and frontend containers.
- Dockerfiles for application services.
- Docker Compose orchestration.
- Health-check endpoint.

## 🏗️ Architecture

```text
                         ┌──────────────────────────┐
                         │           USER           │
                         └─────────────┬────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │    Streamlit Frontend    │
                         │ Dashboard and Data UI    │
                         └─────────────┬────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │      FastAPI Backend     │
                         │       REST API           │
                         └─────────────┬────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │   LangGraph Workflow     │
                         │ Agent Orchestration      │
                         └─────────────┬────────────┘
                                       │
                 ┌─────────────────────┼─────────────────────┐
                 │                     │                     │
                 ▼                     ▼                     ▼
        ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
        │ SQL Analytics   │  │ BI Analytics    │  │ RAG Components  │
        │ SQL Validation  │  │ KPI / Statistics│  │ Embeddings      │
        │ Query Execution │  │ Forecasting      │  │ Vector Storage  │
        └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
                 │                    │                    │
                 └────────────────────┼────────────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │       DuckDB Storage     │
                         │  Analytical Data Layer   │
                         └──────────────────────────┘

        Supporting Components
        ├── Ollama / LLM Integration
        ├── MCP Integration
        ├── Redis Integration
        ├── Evaluation Framework
        ├── LangSmith Tracing
        └── Docker Deployment
```

## 🔄 Workflow

1. **Dataset ingestion:** The user uploads a supported dataset.
2. **Input validation:** The backend validates the uploaded file.
3. **Data registration:** The dataset is loaded into DuckDB.
4. **Profiling:** Dataset structure, column types, and quality information are inspected.
5. **Question submission:** The user asks a business question.
6. **Workflow orchestration:** The agent workflow coordinates the analytical steps.
7. **SQL execution:** Relevant queries are validated and executed.
8. **Analytics:** The application computes the requested results.
9. **Response generation:** Results and supporting evidence are returned.
10. **Evaluation:** Selected workflows are measured using evaluation cases.

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Agent Orchestration | LangGraph |
| LLM Integration | Ollama |
| Data Processing | Pandas, Polars |
| Analytical Database | DuckDB |
| SQL Validation | SQLGlot |
| Visualization | Plotly |
| Machine Learning | Scikit-learn |
| Forecasting | Prophet |
| Caching / Integration | Redis |
| Retrieval-Augmented Generation | Custom RAG components |
| Embeddings | Configured embedding provider |
| Vector Storage | Configured vector-store implementation |
| AI Observability | LangSmith |
| Tool Integration | Model Context Protocol (MCP) |
| Containerization | Docker, Docker Compose |
| Testing | Pytest |
| API Testing | HTTPX |

*The stack reflects project dependencies and implementation areas. Individual optional components may require separate configuration before use.*

## 📂 Project Structure

```text
agent/
│
├── app/
│   ├── agents/
│   │   ├── bi_insight.py
│   │   ├── planner.py
│   │   ├── recommendation.py
│   │   ├── root_cause.py
│   │   ├── self_correction.py
│   │   └── sql_analyst.py
│   │
│   ├── analytics/
│   │   ├── anomaly.py
│   │   ├── forecasting.py
│   │   ├── kpi.py
│   │   ├── statistics.py
│   │   └── visualization.py
│   │
│   ├── api/
│   │   └── main.py
│   ├── config/
│   │   └── settings.py
│   ├── context/
│   │   └── conversation.py
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   └── run_evaluation.py
│   ├── graph/
│   │   ├── state.py
│   │   └── workflow.py
│   ├── ingestion/
│   ├── integrations/
│   │   └── mcp_server.py
│   ├── llm/
│   │   └── ollama_provider.py
│   ├── profiling/
│   ├── rag/
│   │   ├── chunker.py
│   │   ├── document_loader.py
│   │   ├── embeddings.py
│   │   ├── rag_pipeline.py
│   │   └── vector_store.py
│   ├── sql/
│   └── storage/
│       ├── duckdb_manager.py
│       └── redis_manager.py
│
├── data/
│   └── raw/
│       └── sales.csv
├── reports/
│   └── evaluation_report.json
├── streamlit_app/
│   └── app.py
├── tests/
├── .dockerignore
├── .gitignore
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 📋 Prerequisites

Before running NEXORA AI, ensure the following are available:

- Python 3.12 or a compatible version.
- Git.
- Docker Desktop, for containerized deployment.
- Ollama, if using the local LLM integration.
- The model configured in your environment.
- Required dependencies from `requirements.txt`.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/vummidiganesh55/Ai-analyst.git
cd Ai-analyst
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root. Configure the variables required by the application.

Example local configuration:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
DATABASE_PATH=data/database/bi_agent.duckdb
```

Check `app/config/settings.py` and `docker-compose.yml` for the exact environment-variable names expected by your current configuration.

**Security:** Never commit API keys, passwords, access tokens, or other secrets to GitHub. Keep `.env` excluded from version control.

### 5. Configure Ollama

If your selected workflow uses Ollama, install Ollama and download the configured model:

```bash
ollama pull qwen2.5:3b
```

Ensure the Ollama service is running before testing LLM-dependent features.

## ▶️ Usage

### Start the backend

From the project root:

```bash
uvicorn app.api.main:app --reload
```

The backend should be available at:

- **API:** http://localhost:8000
- **Swagger documentation:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health

### Start the frontend

Open a second terminal, activate the virtual environment, and run:

```bash
streamlit run streamlit_app/app.py
```

Open the local URL printed by Streamlit.

### Run with Docker Compose

The project includes separate backend and frontend containers.

Build and start the services:

```bash
docker compose up --build -d
```

Check container status:

```bash
docker compose ps
```

View service logs:

```bash
docker compose logs -f
```

Stop the services:

```bash
docker compose down
```

With the current Docker port mappings:

| Service | URL |
|---|---|
| Frontend | http://localhost:8502 |
| Backend | http://localhost:8001 |
| Swagger API Docs | http://localhost:8001/docs |
| Health Check | http://localhost:8001/health |

The containerized backend is configured to reach Ollama on the host through `host.docker.internal`. Ensure that the required model is installed and the service is reachable.

## 💡 Example

### Business question

```text
What is the total revenue?
```

### Example SQL operation

```sql
SELECT SUM(revenue) AS total_revenue
FROM sales_data;
```

### Example result

```json
{
  "total_revenue": 1080000
}
```

The included sample evaluation report records this result for its total-revenue workflow case.

The actual result depends on the selected dataset and its contents.

## 🧪 Testing

The automated test suite was run successfully in the captured test output.

**Recorded result: 440 tests passed in 35.94 seconds.**

Run the test suite:

```bash
python -m pytest -q
```

Run an individual test module:

```bash
python -m pytest tests/test_ingestion.py -v
```

Run the evaluation framework:

```bash
python -m app.evaluation.run_evaluation
```

These commands allow you to rerun the tests and evaluation after code changes.

## 📊 Evaluation

The project includes an evaluation framework for measuring selected analytical workflows.

### Recorded evaluation results

The included `reports/evaluation_report.json` records five evaluation cases.

| Metric | Recorded Result |
|---|---:|
| Total Evaluation Cases | 5 |
| Passed Cases | 5 |
| Failed Cases | 0 |
| Task Success Rate | 100% |
| Execution Success Rate | 100% |
| Error Rate | 0% |
| Average Latency | 0.053087 seconds |

### Evaluation coverage

The included cases cover:

- Total revenue calculation.
- Average revenue calculation.
- Revenue aggregation by region.
- Business KPI calculation.
- Handling an unsupported question.

### Interpretation

These figures describe the included five-case evaluation run. They do not establish universal accuracy or production reliability.

The measured latency is specific to the recorded environment and evaluation workload. Repeat the evaluation under controlled conditions before using these figures to compare model or system performance.

## 📸 Screenshots / Demo

The following screenshots document the application's interface, API, analytics, deployment, and evaluation.

**Before these images can render, upload the corresponding image files to a `screenshots/` folder in this GitHub repository.**

### 1. Business Intelligence Dashboard

![NEXORA AI Dashboard](screenshots/dashboard.png)

The dashboard displays backend status, registered datasets, and platform capabilities.

### 2. Dataset Ingestion

![Dataset Ingestion](screenshots/dataset_ingestion.png)

The dataset management interface supports uploading CSV, Excel, and Parquet files.

### 3. Data Quality Report

![Data Quality Report](screenshots/data_quality_report.png)

The data-quality endpoint reports dataset structure, missing values, duplicates, and quality information.

### 4. Business Intelligence Query

![Business Intelligence Query](screenshots/business_intelligence_query.png)

Example natural-language question and structured SQL execution results.

### 5. API Documentation

![FastAPI Swagger Documentation](screenshots/api_documentation.png)

Swagger UI exposes the available API endpoints and their request schemas.

### 6. API Health Check

![API Health Check](screenshots/api_health.png)

The health endpoint confirms that the backend service is responding.

### 7. Docker Deployment

![Docker Deployment](screenshots/docker_deployment.png)

Docker Compose shows the backend and frontend containers running.

### 8. Evaluation Results

![Evaluation Results](screenshots/evaluation_results.png)

The recorded evaluation report shows the results of five test cases.

### 9. Automated Test Results

![Pytest Results](screenshots/pytest_results.png)

The captured test output records 440 passing tests.

### 10. LangSmith Trace

![LangSmith Trace](screenshots/langsmith_trace.png)

The trace illustrates a LangGraph workflow execution and its associated inputs, outputs, and execution steps.

## ⚡ Performance

The project includes recorded measurements from its evaluation and test runs.

| Measurement | Recorded Result | Context |
|---|---:|---|
| Evaluation cases | 5 | Included evaluation report |
| Evaluation success rate | 100% | Five recorded cases |
| Average evaluation latency | 0.053087 seconds | Included evaluation report |
| Automated tests passed | 440 | Captured Pytest run |
| Pytest execution time | 35.94 seconds | Captured test output |

These are observed results from specific runs, not guaranteed performance benchmarks.

End-to-end latency can vary with dataset size, hardware, model availability, workflow complexity, and external services.

## 🔐 Security

Security-related implementation areas include:

- Uploaded-file validation.
- SQL query validation.
- Controlled SQL execution.
- DuckDB query resource-limit and timeout testing.
- Environment-variable-based configuration.
- Upload-security tests.
- Excluding local environment files from version control.

### Security recommendations

- Never publish credentials or API keys.
- Do not upload private or personally identifiable customer data.
- Review datasets and reports before committing them.
- Validate generated SQL before execution.
- Apply appropriate access controls before exposing the application publicly.
- Review container configuration and resource limits before production deployment.

Passing security-related tests does not guarantee that the entire application is free from vulnerabilities.

## 🔮 Future Improvements

Potential improvements include:

- Expanding the evaluation dataset with more representative business questions.
- Measuring analytical accuracy across different datasets.
- Improving agent recovery and error-handling behavior.
- Enhancing observability and tracing.
- Improving the user experience for complex analytical workflows.
- Expanding automated integration tests.
- Evaluating performance on larger datasets.
- Strengthening deployment security and operational monitoring.

## ⚠️ Limitations

- Results depend on the quality and structure of the uploaded data.
- LLM-generated responses may contain errors.
- Forecasting quality depends on historical data and model assumptions.
- Evaluation results cover a limited set of predefined cases.
- Local LLM performance depends on available system resources.
- Optional integrations may require additional configuration.
- Production use requires additional validation of security, reliability, and operational behavior.

## 🤝 Contributing

Contributions, bug reports, and suggestions are welcome.

To contribute:

1. Fork the repository.
2. Create a feature branch.
3. Implement your changes.
4. Add or update relevant tests.
5. Run the test suite.
6. Submit a pull request describing your changes.

Please avoid committing secrets, private datasets, local databases, or generated files containing sensitive information.

## 📄 License

No license has been specified in this repository.

Until a license is added, users should not assume that the project grants permission to reuse, modify, or redistribute its code.

## 👨‍💻 Author

**Vummidi Ganesh**

GitHub: [@vummidiganesh55](https://github.com/vummidiganesh55)

Project Repository: [NEXORA AI — Ai-analyst](https://github.com/vummidiganesh55/Ai-analyst)

---

<p align="center">
  <strong>NEXORA AI — Turning Data into Actionable Business Insights</strong>
</p>
