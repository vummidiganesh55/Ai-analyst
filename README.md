# 🚀 NEXORA AI — Autonomous AI Business Intelligence & Data Analytics Agent

An AI-powered business intelligence platform that helps users analyze datasets, validate data quality, generate SQL-based insights, identify anomalies, calculate KPIs, and explore business questions through an interactive interface.

---

## 📌 Project Overview

Business teams often spend significant time cleaning datasets, writing SQL queries, calculating metrics, and preparing reports. Traditional analytics workflows require users to have technical knowledge of databases and data analysis tools.

**NEXORA AI** aims to simplify this process by combining data analytics, AI agents, SQL execution, data profiling, and interactive visualizations in one platform.

Users can upload datasets, inspect data quality, ask business questions, and obtain analytical results through an interactive application.

---

## 🎯 Problem Statement

Traditional business intelligence workflows face several challenges:

- Manual data analysis consumes time.
- Writing SQL queries requires technical expertise.
- Data quality issues can affect analytical results.
- Identifying trends and anomalies requires additional analysis.
- Business users need accessible interfaces for exploring datasets.
- Analytical workflows often involve disconnected tools.

NEXORA AI addresses these challenges by integrating data processing, analytics, AI-assisted workflows, and evaluation into one application.

---

## 💡 Solution

NEXORA AI combines a Python analytics backend with AI-assisted analytical workflows and an interactive Streamlit frontend.

The platform supports dataset ingestion, data profiling, quality validation, SQL-based analysis, statistical calculations, KPI generation, forecasting, anomaly detection, and visualization.

Its modular architecture separates analytics, agent workflows, data storage, API endpoints, and evaluation components.

---

## ✨ Features

### 📂 Dataset Ingestion
- Upload supported CSV, Excel, and Parquet datasets.
- Validate uploaded files.
- Load datasets into DuckDB for analytical queries.

### 🔍 Data Profiling
- Inspect dataset structure and columns.
- Generate data profiling information.
- Identify potential data quality issues.

### 🛡️ Data Quality
- Validate uploaded datasets.
- Identify common data quality problems.
- Provide information to support data-quality assessment.

### 🤖 AI Agent Workflows
- Planning and analytical task coordination.
- Business insight generation.
- Recommendation generation.
- Root-cause analysis.
- Self-correction capabilities.

### 🧮 SQL Analytics
- SQL-assisted business analysis.
- SQL validation.
- Controlled query execution.
- Analytical result generation.

### 📊 Business Intelligence
- KPI calculations.
- Statistical analysis.
- Anomaly detection.
- Forecasting.
- Data visualization.

### 📚 Retrieval-Augmented Generation (RAG)
- Document loading.
- Text chunking.
- Embedding generation.
- Vector storage and retrieval.

### 🔌 API and Integrations
- FastAPI backend.
- REST API endpoints.
- Model Context Protocol (MCP) integration.
- Redis integration.

### 📈 Evaluation
- Evaluation cases for analytical workflows.
- Task success tracking.
- Execution success tracking.
- Error-rate measurement.
- Latency measurement.

### 🐳 Deployment
- Docker-based containerization.
- Docker Compose configuration.
- Separate backend and frontend services.

---

## 🏗️ Architecture

```text
                         ┌────────────────────────┐
                         │         USER           │
                         └────────────┬───────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │    Streamlit Frontend  │
                         │  Interactive Dashboard │
                         └────────────┬───────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │     FastAPI Backend    │
                         │ REST API and Endpoints │
                         └────────────┬───────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │   LangGraph Workflow   │
                         │  Agent Orchestration   │
                         └────────────┬───────────┘
                                      │
                 ┌────────────────────┼────────────────────┐
                 │                    │                    │
                 ▼                    ▼                    ▼
        ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
        │  SQL Analytics │  │ BI Analytics   │  │   RAG Pipeline │
        │ SQL Validation │  │ KPIs/Statistics│  │ Embeddings     │
        │ Query Execution│  │ Forecasting     │  │ Vector Storage │
        └───────┬────────┘  └───────┬────────┘  └───────┬────────┘
                │                   │                   │
                └───────────────────┼───────────────────┘
                                    │
                                    ▼
                         ┌────────────────────────┐
                         │      DuckDB Storage    │
                         │ Analytical Data Layer  │
                         └────────────────────────┘

                 Supporting Components:
                 • Ollama / LLM Integration
                 • MCP Integration
                 • Redis Integration
                 • Evaluation Framework
                 • Docker Deployment
```

---

## 🔄 Workflow

1. The user uploads a dataset through the frontend.
2. The backend validates the uploaded file.
3. The dataset is loaded into DuckDB.
4. Profiling and data-quality components inspect the dataset.
5. The user submits a business question.
6. The analytical workflow coordinates the relevant processing steps.
7. SQL and analytics components execute the required operations.
8. The application returns analytical results and supporting information.
9. Evaluation components measure selected workflow outcomes.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Backend | FastAPI |
| Frontend | Streamlit |
| Agent Orchestration | LangGraph |
| LLM Integration | Ollama |
| Data Processing | Pandas, Polars |
| Analytical Database | DuckDB |
| SQL Validation | SQLGlot |
| Data Visualization | Plotly |
| Machine Learning | Scikit-learn |
| Forecasting | Prophet |
| Vector Search | FAISS/Chroma or another configured vector-store implementation |
| Cache / Data Integration | Redis |
| AI Observability | LangSmith |
| Protocol Integration | MCP |
| Containerization | Docker, Docker Compose |
| Testing | Pytest |
| API Testing | HTTPX |

**Note:** The actual vector-store implementation and optional integrations depend on the project configuration. The table describes the technology stack and intended integration areas, not a claim that every component is active in every workflow.

---

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
│   │
│   ├── config/
│   │   └── settings.py
│   │
│   ├── context/
│   │   └── conversation.py
│   │
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   └── run_evaluation.py
│   │
│   ├── graph/
│   │   ├── state.py
│   │   └── workflow.py
│   │
│   ├── ingestion/
│   ├── integrations/
│   │   └── mcp_server.py
│   ├── llm/
│   │   └── ollama_provider.py
│   ├── profiling/
│   ├── rag/
│   ├── sql/
│   └── storage/
│       ├── duckdb_manager.py
│       └── redis_manager.py
│
├── data/
│   └── raw/
│       └── sales.csv
│
├── reports/
│   └── evaluation_report.json
│
├── streamlit_app/
│   └── app.py
│
├── tests/
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── package.json
├── package-lock.json
├── requirements.txt
└── README.md
```

---

## ⚙️ Prerequisites

Install or configure the following:

- Python 3.12 or a compatible version.
- Git.
- Docker Desktop, if using Docker deployment.
- Ollama, if using the configured local LLM.
- The LLM model configured in your environment.
- Required Python packages.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/vummidiganesh55/Ai-analyst.git
cd Ai-analyst
```

### 2. Create a virtual environment

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

Create a local `.env` file in the project root.

Add the configuration values required by `app/config/settings.py` and the integrations you intend to use.

For example:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
DATABASE_PATH=data/database/bi_agent.duckdb
```

Add other required settings according to your configuration.

**Security:** Never commit API keys, passwords, access tokens, or other secrets to GitHub. Keep `.env` out of version control.

### 5. Configure the local LLM

If using Ollama, install it and pull the configured model:

```bash
ollama pull qwen2.5:3b
```

Make sure the Ollama service is running before using features that depend on it.

### 6. Start the backend

```bash
uvicorn app.api.main:app --reload
```

Backend API:

http://localhost:8000

Interactive API documentation:

http://localhost:8000/docs

Health endpoint:

http://localhost:8000/health

### 7. Start the frontend

Open a second terminal, activate the virtual environment, and run:

```bash
streamlit run streamlit_app/app.py
```

Open the local URL displayed by Streamlit.

---

## 🐳 Docker Deployment

Docker Compose is included for running the backend and frontend as separate services.

### Start the application

From the project root:

```bash
docker compose up --build -d
```

### Check service status

```bash
docker compose ps
```

### View logs

```bash
docker compose logs -f
```

### Stop the application

```bash
docker compose down
```

With the current local port configuration:

| Component | URL |
|---|---|
| Streamlit Frontend | http://localhost:8502 |
| FastAPI Backend | http://localhost:8001 |
| API Documentation | http://localhost:8001/docs |
| Health Check | http://localhost:8001/health |

The Docker backend uses the configured Ollama host address. Ensure Ollama is available and the required model has been downloaded.

---

## 🧪 Testing

Run the automated test suite from the project root:

```bash
python -m pytest -q
```

To run an individual test module:

```bash
python -m pytest tests/test_ingestion.py -v
```

Run the project evaluation framework:

```bash
python -m app.evaluation.run_evaluation
```

Review the generated evaluation results in:

```text
reports/evaluation_report.json
```

Test results can vary with code changes, environment configuration, datasets, and model availability.

---

## 📊 Evaluation Metrics

The evaluation framework reports metrics such as:

| Metric | Description |
|---|---|
| Task Success Rate | Percentage of evaluation tasks that pass |
| Execution Success Rate | Percentage of evaluation cases that execute successfully |
| Error Rate | Percentage of evaluation cases that encounter errors |
| Average Latency | Average time required to execute evaluated cases |

### Sample Evaluation Results

The included evaluation report records the following results for five evaluation cases:

| Metric | Recorded Result |
|---|---:|
| Total Cases | 5 |
| Passed Cases | 5 |
| Failed Cases | 0 |
| Task Success Rate | 100% |
| Execution Success Rate | 100% |
| Error Rate | 0% |
| Average Latency | Approximately 0.053 seconds |

These figures describe the included evaluation run only. They do not establish universal accuracy, production reliability, or performance across larger datasets and different workloads.

---

## 🔐 Security Considerations

Security considerations for this project include:

- Input validation for uploaded datasets.
- SQL query validation before execution.
- Controlled handling of analytical queries.
- Environment-based configuration for sensitive settings.
- Excluding local environment files from version control.
- Testing upload and SQL execution behavior.

Before production deployment, additional validation of authentication, authorization, resource limits, logging, dependency security, and deployment configuration may be required.

---

## ⚠️ Limitations

- Analytical quality depends on dataset quality and schema consistency.
- LLM-generated outputs may contain errors and should be validated.
- Forecasting quality depends on the dataset, selected method, and available historical information.
- Evaluation results depend on the test cases and evaluation methodology.
- Local LLM performance depends on available hardware and model configuration.
- Production deployment requires appropriate security, monitoring, and resource management.

---

## 🔮 Future Improvements

Potential future improvements include:

- Expanding the evaluation dataset.
- Adding more representative business-analysis test cases.
- Improving error handling and recovery.
- Enhancing observability and tracing.
- Improving dashboard usability.
- Expanding automated integration testing.
- Evaluating analytical quality on diverse datasets.

---

## 👨‍💻 Author

**Sri Satya Thanuj Vummidi**

GitHub: [@SriSatyaThanujVummidi](https://github.com/SriSatyaThanujVummidi)

Project Repository: [NEXORA AI — Ai-analyst](https://github.com/vummidiganesh55/Ai-analyst)

---

## 📄 License

No license has been specified in this repository yet. Unless a license is added, the project should not be assumed to grant permission for reuse, modification, or redistribution.

---

## ⭐ Support

If you find this project useful, consider starring the repository on GitHub.
