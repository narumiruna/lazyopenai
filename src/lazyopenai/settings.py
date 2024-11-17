from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class Settings(BaseSettings):
    api_key: str | None = Field(default=None, description="The OpenAI API key.")
    model: str = Field(default="gpt-4o-mini", description="The OpenAI model name.")
    temperature: float = Field(default=0.0, description="The OpenAI temperature setting.")
    embedding_model: str = Field(default="text-embedding-3-small", description="The OpenAI embedding model.")

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file_encoding="utf-8",
        env_file=".env",
        env_prefix="openai_",
        extra="ignore",
    )


settings = Settings()
