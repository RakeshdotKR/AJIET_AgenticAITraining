"""common/config.py - single source of truth for FinDesk Day 2 configuration.

WHY THIS FILE EXISTS
--------------------
Same role as Day 1's config module, extended for Day 2:

    D2P1.py (RAG + tools)   ─┐
    D2P2.py (memory)        ─┼──> common/config.py ───> Day2/.env
    D2P3.py (parallel flow) ─┘

Day 2 adds three configuration areas on top of Day 1:

1. EMBEDDING settings
   RAG needs an embedding model (vector representations of text). Gemini
   and Ollama each need their own embedding model name, kept in .env like
   the chat model names.

2. CHECKPOINTER settings
   Session 5 persists graph state. The choice between the in-memory saver
   and SQLite, plus the database path, comes from .env.

3. DATA_DIR
   One constant pointing at Day2/data so every module loads sample
   documents from the same place.

RULE: no model name, API key, path default that contains a secret is
hardcoded in Python - the .env file is the only place they are written.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------
# 1. Locate and load the .env file.
#    Path(__file__)      -> .../Day2/common/config.py
#    .parent             -> .../Day2/common
#    .parent.parent      -> .../Day2        <- .env and data/ live here
# --------------------------------------------------------------------------
ENV_FILE: Path = Path(__file__).resolve().parent.parent / ".env"

# Absolute path of the Day2 folder (used to find data/ and .env).
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Sample documents, JSON records and reference data used by all sessions.
DATA_DIR: Path = BASE_DIR / "data"

# Inject the values from .env into os.environ (shell values win if set).
load_dotenv(dotenv_path=ENV_FILE)

# --------------------------------------------------------------------------
# 2. Chat model provider selection - "ollama" (default) or "gemini".
#    Model names have NO code defaults: they must exist in .env.
# --------------------------------------------------------------------------
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

# Ollama (local) settings --------------------------------------------------
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "")

# Gemini (cloud) settings --------------------------------------------------
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "")

# --------------------------------------------------------------------------
# 3. EMBEDDING model settings (Day 2 - used by RAG in Session 4).
#    Never mix two embedding models in one ChromaDB collection: the vector
#    dimensions will not match. build_retriever() therefore also uses the
#    embedding model name as part of the collection name.
# --------------------------------------------------------------------------
OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "")
GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "")

# --------------------------------------------------------------------------
# 4. CHECKPOINTER settings (Day 2 - used by Session 5).
#    CHECKPOINTER = "memory" (default) -> InMemorySaver (lost on exit)
#    CHECKPOINTER = "sqlite"           -> SqliteSaver   (survives restarts)
# --------------------------------------------------------------------------
CHECKPOINTER: str = os.getenv("CHECKPOINTER", "memory").strip().lower()
SQLITE_PATH: str = os.getenv("SQLITE_PATH", str(BASE_DIR / "findesk.db"))

# Placeholder values from .env.example are NOT valid credentials.
_PLACEHOLDER_PREFIX = "your_"


def validate_provider(provider: str | None = None) -> str:
    """Check that the requested chat provider can actually be used.

    Parameters
    ----------
    provider : str, optional
        "ollama" or "gemini". When None, the value from .env is checked.

    Returns
    -------
    str
        The normalised provider name ("ollama" or "gemini").

    Raises
    ------
    ValueError
        If the provider name is unknown.
    RuntimeError
        If the selected provider is missing its configuration in .env.
    """
    selected = (provider or LLM_PROVIDER).strip().lower()

    if selected == "ollama":
        if not OLLAMA_MODEL:
            raise RuntimeError(
                f"OLLAMA_MODEL is not set. Add it to {ENV_FILE}, "
                "for example:  OLLAMA_MODEL=<model-you-pulled>"
            )
        return selected

    if selected == "gemini":
        if not GOOGLE_API_KEY or GOOGLE_API_KEY.startswith(_PLACEHOLDER_PREFIX):
            raise RuntimeError(
                "Gemini selected but GOOGLE_API_KEY is not set.\n"
                f"Edit {ENV_FILE} and replace the placeholder value with a real "
                "key from https://aistudio.google.com/apikey\n"
                "Or switch back to the local model with:  LLM_PROVIDER=ollama"
            )
        if not GEMINI_MODEL:
            raise RuntimeError(
                f"GEMINI_MODEL is not set. Add it to {ENV_FILE}, "
                "for example:  GEMINI_MODEL=<gemini-flash-model>"
            )
        return selected

    raise ValueError(
        f"Unknown LLM_PROVIDER '{selected}'. Use 'ollama' or 'gemini' in .env."
    )


def validate_embeddings(provider: str | None = None) -> str:
    """Check that the requested EMBEDDING provider is configured.

    Separate from validate_provider() because a student may run the chat
    model on Ollama while keeping Gemini embeddings (or vice versa). The
    embedding provider is always derived from the chat provider choice so
    beginners only manage one switch.

    Returns
    -------
    str - the normalised provider name.

    Raises
    ------
    RuntimeError
        If the embedding model name for the selected provider is missing.
    """
    selected = (provider or LLM_PROVIDER).strip().lower()

    if selected == "ollama":
        if not OLLAMA_EMBEDDING_MODEL:
            raise RuntimeError(
                f"OLLAMA_EMBEDDING_MODEL is not set. Add it to {ENV_FILE}, "
                "for example:  OLLAMA_EMBEDDING_MODEL=nomic-embed-text"
            )
        return selected

    if selected == "gemini":
        # Gemini embeddings reuse the same GOOGLE_API_KEY as the chat model.
        if not GOOGLE_API_KEY or GOOGLE_API_KEY.startswith(_PLACEHOLDER_PREFIX):
            raise RuntimeError(
                "Gemini embeddings selected but GOOGLE_API_KEY is not set.\n"
                f"Edit {ENV_FILE} and paste a real key."
            )
        if not GEMINI_EMBEDDING_MODEL:
            raise RuntimeError(
                f"GEMINI_EMBEDDING_MODEL is not set. Add it to {ENV_FILE}, "
                "for example:  GEMINI_EMBEDDING_MODEL=models/embedding-001"
            )
        return selected

    raise ValueError(
        f"Unknown embedding provider '{selected}'. Use 'ollama' or 'gemini'."
    )


def describe() -> str:
    """Return a one-line description of the active chat configuration."""
    provider = validate_provider()
    if provider == "ollama":
        return f"provider=ollama | model={OLLAMA_MODEL} | url={OLLAMA_BASE_URL}"
    # Never print the API key itself - only that one is configured.
    return f"provider=gemini | model={GEMINI_MODEL} | api_key=***configured***"


def describe_embeddings() -> str:
    """Return a one-line description of the active embedding configuration."""
    provider = validate_embeddings()
    if provider == "ollama":
        return f"embeddings=ollama | model={OLLAMA_EMBEDDING_MODEL}"
    return f"embeddings=gemini | model={GEMINI_EMBEDDING_MODEL}"


def connection_hint() -> str:
    """Actionable hint printed when a model call fails at runtime."""
    try:
        provider = validate_provider()
    except (RuntimeError, ValueError) as exc:
        return f"Hint: {exc}"

    if provider == "ollama":
        return (
            "Hint: is Ollama running (`ollama serve`)? Is the model pulled "
            f"(`ollama pull {OLLAMA_MODEL}`)? See {ENV_FILE} for settings."
        )
    return "Hint: check GOOGLE_API_KEY and GEMINI_MODEL in .env, then retry."
