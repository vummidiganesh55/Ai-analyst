from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Autonomous AI BI Agent"
    environment: str = "development"

    database_path: str = "data/database/bi_agent.duckdb"

    max_file_size_mb: int = 50
    max_dataset_rows: int = 500_000
    max_dataset_columns: int = 200
    max_dataset_memory_mb: int = 256
    max_result_rows: int = 10000
    query_timeout_seconds: int = 10

    model_name: str = "local"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    llm_temperature: float = 0.0
    duckdb_memory_limit: str = "512MB"
    duckdb_threads: int = 2
    duckdb_max_expression_depth: int = 500

settings = Settings()