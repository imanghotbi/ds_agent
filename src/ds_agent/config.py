from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr

class Settings(BaseSettings):
    """
    Application settings and environment variables.
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore",case_sensitive=False,)

    model_api_key: SecretStr
    e2b_api_key: SecretStr
    
    # Optional settings with defaults
    model_name: str = "deepseek-ai/deepseek-v3.2"
    temperature: float = 0.0
    log_level: str = "INFO"

# Create a singleton instance
settings = Settings()