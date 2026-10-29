from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables / `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_provider: Literal["groq", "anthropic", "openai", "ollama"] = "groq"
    llm_model: str | None = None
    llm_max_tokens: int = 1024
    groq_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"

    # Embeddings
    embedding_provider: Literal["onnx", "huggingface", "openai", "ollama"] = "onnx"
    embedding_model: str | None = None

    # Chunking & retrieval
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_k: int = 5
    retrieval_fetch_k: int = 20
    use_mmr: bool = True
    max_history_turns: int = 6

    # Storage
    data_dir: Path = Path("data")
    collection_name: str = "pdf_chunks"
    max_upload_mb: int = 50

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "registry.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
