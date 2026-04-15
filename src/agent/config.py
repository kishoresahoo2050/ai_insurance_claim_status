from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    APP_NAME: str = "ai-insurance-claim-assistant"
    APP_ENV: str = "development"

    # REQUIRED (no defaults)
    GOOGLE_API_KEY: str

    # Optional
    GOOGLE_AI_MODEL: str

    class Config:
        env_file = Path(__file__).resolve().parent.parent / ".env"
        extra = "ignore"


settings = Settings()
