"""common/model_factory.py - creates the chat model AND embedding model.

WHY THIS FILE EXISTS
--------------------
Day 1 only needed chat models. Day 2 adds RAG, which needs an *embedding
model* (text -> vector) as well:

    Chat model      Ollama  -> langchain_ollama.ChatOllama
                      Gemini  -> langchain_google_genai.ChatGoogleGenerativeAI
    Embedding model Ollama  -> langchain_ollama.OllamaEmbeddings
                      Gemini  -> langchain_google_genai.GoogleGenerativeAIEmbeddings

Both pairs expose the standard LangChain interface, so RAG code never
needs to know which provider is running:

    from common import model_factory
    llm  = model_factory.get_llm()            # chat model from .env
    emb  = model_factory.get_embeddings()     # embedding model from .env

Provider switching stays a one-line change in .env (LLM_PROVIDER=...).
The embedding provider always follows the chat provider so beginners only
manage one switch.
"""

from __future__ import annotations

from typing import Any

from common import config


def get_llm(provider: str | None = None, temperature: float = 0.2) -> Any:
    """Create and return a LangChain chat model.

    Parameters
    ----------
    provider : str, optional
        "ollama" or "gemini". When None, config.LLM_PROVIDER (.env) is used.
    temperature : float
        Sampling temperature. Low values keep finance answers consistent.

    Returns
    -------
    A chat model instance with the standard LangChain runnable interface.

    Raises
    ------
    RuntimeError / ValueError
        Propagated from config.validate_provider() for bad configuration.
    ImportError
        If the provider's integration package is not installed.
    """
    selected = config.validate_provider(provider)

    # ------------------------------------------------------------------
    # LOCAL MODEL (default): Ollama
    # ------------------------------------------------------------------
    if selected == "ollama":
        # Imported lazily so a missing optional package only breaks the
        # provider that actually needs it.
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.OLLAMA_MODEL,        # from .env (never hardcoded)
            base_url=config.OLLAMA_BASE_URL,   # from .env
            temperature=temperature,
        )

    # ------------------------------------------------------------------
    # CLOUD MODEL: Gemini (key already loaded into os.environ by config)
    # ------------------------------------------------------------------
    if selected == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:  # pragma: no cover - environment issue
            raise ImportError(
                "Gemini support requires: pip install langchain-google-genai"
            ) from exc

        return ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,   # from .env (never hardcoded)
            temperature=temperature,
            max_retries=2,               # transient API errors are retried
        )

    raise ValueError(f"Unsupported provider: {selected}")


def get_embeddings(provider: str | None = None) -> Any:
    """Create and return a LangChain embeddings model for RAG.

    Parameters
    ----------
    provider : str, optional
        "ollama" or "gemini". When None, config.LLM_PROVIDER (.env) is used,
        so embeddings always follow the chat provider unless overridden.

    Returns
    -------
    An embeddings instance with .embed_documents() and .embed_query().

    Raises
    ------
    RuntimeError / ValueError
        Propagated from config.validate_embeddings().

    Notes
    -----
    * Ollama: run `ollama pull <OLLAMA_EMBEDDING_MODEL>` once first.
      A small model such as nomic-embed-text (~274 MB) is enough.
    * Gemini: embedding-001 is part of the free tier quota.
    * NEVER index with one embedding model and query with another -
      the vector dimensions will not match. The RAG module uses the
      model name inside the Chroma collection name to prevent this.
    """
    selected = config.validate_embeddings(provider)

    if selected == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=config.OLLAMA_EMBEDDING_MODEL,  # from .env
            base_url=config.OLLAMA_BASE_URL,
        )

    if selected == "gemini":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Gemini embeddings require: pip install langchain-google-genai"
            ) from exc

        return GoogleGenerativeAIEmbeddings(
            model=config.GEMINI_EMBEDDING_MODEL,  # from .env
        )

    raise ValueError(f"Unsupported embedding provider: {selected}")
