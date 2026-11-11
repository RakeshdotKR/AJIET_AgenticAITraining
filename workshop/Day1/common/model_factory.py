"""common/model_factory.py - creates the chat model for the active provider.

WHY THIS FILE EXISTS
--------------------
The training supports two providers:

    Ollama  -> langchain_ollama.ChatOllama        (local, default)
    Gemini  -> langchain_google_genai.ChatGoogleGenerativeAI (cloud)

Both expose the same LangChain interface (`.invoke()`, `.stream()`,
`.with_structured_output()`), so the rest of the application never has to
know which one is running. Switching providers is a one-line change in .env:

    LLM_PROVIDER=ollama     # or  gemini

...and NO workflow code changes. This is the "common model-configuration
module" described in the training plan.

USAGE
-----
    from common import model_factory
    llm = model_factory.get_llm()            # provider from .env
    llm = model_factory.get_llm("gemini")    # explicit override (CLI menu)
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
        Sampling temperature. A low value keeps finance answers consistent;
        raise it slightly (e.g. 0.7) if you want more varied wording.

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
    # ChatOllama talks to the Ollama HTTP server (default localhost:11434).
    # The model name comes from .env (OLLAMA_MODEL) - never from code.
    # ------------------------------------------------------------------
    if selected == "ollama":
        from langchain_ollama import ChatOllama  # imported lazily so that a
        # missing optional package only breaks the provider that needs it.

        return ChatOllama(
            model=config.OLLAMA_MODEL,        # from .env (never hardcoded)
            base_url=config.OLLAMA_BASE_URL,   # from .env
            temperature=temperature,
        )

    # ------------------------------------------------------------------
    # CLOUD MODEL: Gemini
    # ChatGoogleGenerativeAI reads the GOOGLE_API_KEY environment variable,
    # which config.py already loaded from .env - so no key appears here.
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

    # validate_provider() already rejects unknown names; this is a safety net.
    raise ValueError(f"Unsupported provider: {selected}")
