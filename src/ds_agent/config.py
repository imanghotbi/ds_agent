from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr

class Nodes:
    SUPERVISOR = "supervisor"
    CLEANER = "cleaner"
    EDA = "eda"
    FEATURE_ENGINEER = "feature_engineer"
    TRAINER = "trainer"
    STORYTELLER = "storyteller"
    TOOLS = "tools"
    REPORTER = "reporter"
    FINISH = "FINISH"

class Settings(BaseSettings):
    """
    Application settings and environment variables.
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore",case_sensitive=False,)

    model_api_key: SecretStr
    e2b_api_key: SecretStr
    
    # Optional settings with defaults
    model_name: str = "qwen/qwen3-coder-480b-a35b-instruct"
    supervisor_model_name: str = "qwen/qwen3-235b-a22b"
    cleaner_model_name: str = "qwen/qwen3-coder-480b-a35b-instruct"
    eda_model_name: str = "qwen/qwen3-coder-480b-a35b-instruct"
    feature_engineer_model_name: str = "qwen/qwen3-coder-480b-a35b-instruct"
    trainer_model_name: str = "qwen/qwen3-coder-480b-a35b-instruct"
    storyteller_model_name: str = "qwen/qwen3-235b-a22b"
    reporter_model_name: str = "qwen/qwen3-235b-a22b"
    temperature: float = 0.0
    log_level: str = "INFO"
    log_file_path: str = "./logs/app.log"
    log_max_bytes:int = 30 * 1024 * 1024 #30 MB
    log_backup_count:int = 5
    sandbox_timeout: int = 3600
    max_retries: int = 3
    node_recursion_limit: int = 50

# Create a singleton instance
settings = Settings()