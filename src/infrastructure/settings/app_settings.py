from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    app_env: str = "dev"
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    litellm_api_key: str = ""
    google_api_key: str = ""
    twikit_username: str = ""
    twikit_password: str = ""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
