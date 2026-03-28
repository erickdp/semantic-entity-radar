from pydantic import BaseModel


class ProviderConfig(BaseModel):
    name: str
    model: str
    api_key_env: str
    max_retries: int = 3
    timeout_sec: int = 30
