from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Autonomous AI BI Agent"
    environment: str = "development"

    database_path: str = "data/database/bi_agent.duckdb"

    max_file_size_mb: int = 50
    max_result_rows: int = 10000
    query_timeout_seconds: int = 10

    model_name: str = "local"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()