"""common/config.py - single source of truth for FinDesk configuration.

WHY THIS FILE EXISTS
--------------------
Every script in the training needs the same settings: which LLM provider to
use, the Ollama server URL, the Ollama model name, the Gemini API key and the
Gemini model name. Instead of copying those values into each file (and risking
hardcoded secrets), ALL files import this module.

    D1P1.py ─┐
    D1P2.py ─┼──> common/config.py ───> Day1/.env
    D1P3.py ─┘

WHAT IT DOES
------------
1. Locates Day1/.env (one directory above this package).
2. Loads it with python-dotenv -> values become environment variables.
3. Publishes them as module-level constants.
4. Validates the provider selection (including that the required model
   name / API key actually exists in .env) before any model is created.

RULE: no model name or API key is hardcoded anywhere in the Python code -
the .env file is the only place they are written.

NOTE: load_dotenv() never overwrites a variable that is already set in the
shell, so a shell override still works for quick tests:
    OLLAMA_MODEL=<other-model> python D1P1.py
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------
# 1. Locate and load the .env file.
#    Path(__file__)          -> .../Day1/common/config.py
#    .parent                 -> .../Day1/common
#    .parent.parent          -> .../Day1          <- .env lives here
# --------------------------------------------------------------------------
ENV_FILE: Path = Path(__file__).resolve().parent.parent / ".env"

# Inject the values from .env into os.environ (does not overwrite shell vars).
load_dotenv(dotenv_path=ENV_FILE)

# --------------------------------------------------------------------------
# 2. Provider selection - "ollama" (default) or "gemini".
#    os.getenv returns None if the key is missing, so we supply a default
#    ONLY for the provider switch. Model names and keys have NO code
#    defaults: they must be present in .env and are validated on use.
# --------------------------------------------------------------------------
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

# Ollama (local) settings --------------------------------------------------
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "")

# Gemini (cloud) settings --------------------------------------------------
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "")

# Placeholder values from .env.example are NOT valid credentials.
_PLACEHOLDER_PREFIX = "your_"


def validate_provider(provider: str | None = None) -> str:
    """Check that the requested provider can actually be used.

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
        If the selected provider is missing its configuration in .env:
        OLLAMA_MODEL for Ollama, GOOGLE_API_KEY / GEMINI_MODEL for Gemini.
    """
    selected = (provider or LLM_PROVIDER).strip().lower()

    if selected == "ollama":
        # No API key required, but the local model name must be in .env.
        # Connection errors surface later, when the model is first invoked
        # (Ollama must be running: `ollama serve`).
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


def describe() -> str:
    """Return a one-line description of the active configuration.

    Used by the CLI banner so students always know which model is running.
    """
    provider = validate_provider()
    if provider == "ollama":
        return f"provider=ollama | model={OLLAMA_MODEL} | url={OLLAMA_BASE_URL}"
    # Do not print the API key itself - only that one is configured.
    return f"provider=gemini | model={GEMINI_MODEL} | api_key=***configured***"


def connection_hint() -> str:
    """Actionable hint printed when a model call fails at runtime.

    Built from the CURRENT configuration (never from a hardcoded model
    name), so it stays correct after the student edits .env.
    """
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
