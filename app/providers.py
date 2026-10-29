"""Factories for the chat model and embedding model, chosen via settings.

Provider packages are imported lazily so only the one you use has to be configured.
"""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr

from app.config import Settings

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

DEFAULT_LLM_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.1",
}

DEFAULT_EMBEDDING_MODELS = {
    "onnx": "all-MiniLM-L6-v2",
    "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
    "openai": "text-embedding-3-small",
    "ollama": "nomic-embed-text",
}


def llm_model_name(settings: Settings) -> str:
    return settings.llm_model or DEFAULT_LLM_MODELS[settings.llm_provider]


def embedding_model_name(settings: Settings) -> str:
    return settings.embedding_model or DEFAULT_EMBEDDING_MODELS[settings.embedding_provider]


def _api_key(secret: SecretStr | None) -> dict:
    # When no key is configured, the SDK falls back to its standard env var.
    return {"api_key": secret} if secret else {}


def build_chat_model(settings: Settings) -> BaseChatModel:
    model = llm_model_name(settings)

    if settings.llm_provider == "groq":
        # Groq exposes an OpenAI-compatible API, so the OpenAI client works as-is.
        from langchain_openai import ChatOpenAI

        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys")
        return ChatOpenAI(
            model=model,
            base_url=GROQ_BASE_URL,
            api_key=settings.groq_api_key,
            max_tokens=settings.llm_max_tokens,
            temperature=0.1,
        )

    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            max_tokens=settings.llm_max_tokens,
            **_api_key(settings.anthropic_api_key),
        )

    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            max_tokens=settings.llm_max_tokens,
            temperature=0.1,
            **_api_key(settings.openai_api_key),
        )

    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            base_url=settings.ollama_base_url,
            num_predict=settings.llm_max_tokens,
            temperature=0.1,
        )

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


class OnnxMiniLMEmbeddings(Embeddings):
    """all-MiniLM-L6-v2 run through ONNX Runtime (bundled with ChromaDB), so no PyTorch is needed."""

    def __init__(self):
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self._embed = ONNXMiniLM_L6_V2()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embeddings(settings: Settings) -> Embeddings:
    model = embedding_model_name(settings)

    if settings.embedding_provider == "onnx":
        if model != DEFAULT_EMBEDDING_MODELS["onnx"]:
            raise ValueError("The onnx embedding provider only supports all-MiniLM-L6-v2.")
        return OnnxMiniLMEmbeddings()

    if settings.embedding_provider == "huggingface":
        # Optional: pip install langchain-huggingface sentence-transformers
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=model,
            encode_kwargs={"normalize_embeddings": True},
        )

    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model, **_api_key(settings.openai_api_key))

    if settings.embedding_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=model, base_url=settings.ollama_base_url)

    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")
