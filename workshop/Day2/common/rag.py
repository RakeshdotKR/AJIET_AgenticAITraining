"""common/rag.py - Retrieval-Augmented Generation helpers for FinDesk.

WHY THIS FILE EXISTS
--------------------
Session 4 must answer questions from *company* documents (travel limits,
reimbursement rules...). The RAG pipeline is:

    .txt files  --load-->  Document objects
              --split-->  chunks (overlapping text pieces)
              --embed-->  vectors (numbers representing meaning)
              --store-->  ChromaDB (vector database)
    question    --embed-->  query vector
              --search-->  top-k similar chunks  ->  LLM context

This module owns steps 1-5 so D2P1.py stays focused on the graph. Every
function is provider-agnostic: the embedding model comes from
model_factory.get_embeddings(), which reads .env.

IMPORTANT RULE (from the training plan)
----------------------------------------
One Chroma collection may only ever hold vectors from ONE embedding
model. We therefore derive the collection name from the embedding model
name, so switching providers cannot silently mix incompatible vectors.
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common import config, model_factory

# --------------------------------------------------------------------------
# Text splitter settings.
# chunk_size  - max characters per chunk (small = precise, large = context).
# chunk_overlap - characters repeated between chunks so a fact split across
#                 a boundary is still found in one of the two chunks.
# RecursiveCharacterTextSplitter tries "\n\n", "\n", " ", "" in order, so
# paragraphs and sentences stay intact wherever possible.
# --------------------------------------------------------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# How many similar chunks are returned per question (retriever k).
RETRIEVER_K = 4


def load_policy_documents(folder: Path | None = None) -> list[Document]:
    """Load every .txt policy file into LangChain Document objects.

    Parameters
    ----------
    folder : Path, optional
        Directory to read. Defaults to Day2/data/policies.

    Returns
    -------
    list[Document]
        One Document per file. Each Document carries:
          page_content : the full text of the file
          metadata     : {"source": file name, "policy": readable title}

    WHY Document OBJECTS?
    ---------------------
    Document is LangChain's standard envelope for any text the LLM should
    see. Keeping metadata (here: which policy the text came from) means the
    answer can cite its source later.

    NOTE ON DOCUMENT LOADERS
    ------------------------
    In production you would use a loader such as
    langchain_community.document_loaders.TextLoader (or PDF/SharePoint
    loaders). We read files manually here to keep the dependency list
    small - the resulting Document objects are identical.
    """
    folder = folder or (config.DATA_DIR / "policies")
    documents: list[Document] = []

    # sorted() gives a stable, alphabetical load order every run.
    for path in sorted(folder.glob("*.txt")):
        # encoding="utf-8" handles the ₹ character used in the policies.
        text = path.read_text(encoding="utf-8")

        # Turn "corporate_travel_policy.txt" into "corporate travel policy"
        # for a human-readable metadata title.
        title = path.stem.replace("_", " ")

        documents.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "policy": title},
            )
        )
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split long policy texts into overlapping chunks suitable for search.

    Parameters
    ----------
    documents : list[Document]
        Full documents as returned by load_policy_documents().

    Returns
    -------
    list[Document]
        Smaller Documents. Metadata (source, policy) is copied to every
        chunk automatically, so results still cite the right policy.

    WHY SPLIT?
    ----------
    Embedding models have a maximum input length, and one giant vector for
    a whole policy would blur together unrelated rules. Small chunks make
    similarity search precise: the query "hotel limit" should surface the
    hotel section, not the whole travel policy.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Never cut in the middle of a word or a markdown heading.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def _collection_name(embedding_model: str) -> str:
    """Build a safe Chroma collection name from the embedding model name.

    Chroma collection names may only contain letters, digits, underscores,
    hyphens and dots (max 63 chars). Gemini model names contain '/' which
    is illegal, so every non-alphanumeric character becomes '_'.
    Example: "models/embedding-001" -> "models_embedding_001"
    """
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", embedding_model)
    # Prefix keeps names readable and within Chroma's 63-char limit.
    return f"findesk_{safe}"[:63]


def build_vectorstore(
    provider: str | None = None,
    documents: list[Document] | None = None,
) -> Chroma:
    """Load, split, embed and store the policy documents in Chroma.

    Parameters
    ----------
    provider : str, optional
        "ollama" or "gemini" - decides which embedding model is used.
        None means "follow LLM_PROVIDER from .env".
    documents : list[Document], optional
        Pre-loaded documents (mainly for tests). When None the policy
        folder is loaded automatically.

    Returns
    -------
    Chroma
        An in-memory vector store holding the indexed chunks.

    NOTES
    -----
    * In-memory storage keeps the classroom simple: the index is rebuilt
      on every start (only a few seconds for six small policies) and can
      never go stale or mix sessions. For production add
      persist_directory=... to Chroma to keep the index on disk.
    * from_documents() performs the embedding calls - with Gemini this
      counts against the free-tier quota, with Ollama it uses local CPU.
    """
    if documents is None:
        documents = load_policy_documents()
    if not documents:
        raise RuntimeError(
            f"No .txt policies found in {config.DATA_DIR / 'policies'}. "
            "Check the data folder."
        )

    # Embeddings always follow the chat provider unless overridden, so a
    # single LLM_PROVIDER switch moves chat AND retrieval together.
    embeddings = model_factory.get_embeddings(provider)

    # The collection name embeds the model name: switching providers
    # creates a NEW collection instead of mixing vector dimensions.
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=_collection_name(
            config.OLLAMA_EMBEDDING_MODEL
            if config.validate_embeddings(provider) == "ollama"
            else config.GEMINI_EMBEDDING_MODEL
        ),
    )


def build_retriever(provider: str | None = None) -> object:
    """Build the full RAG pipeline: docs -> chunks -> vectors -> retriever.

    Parameters
    ----------
    provider : str, optional
        Chat/embedding provider ("ollama" | "gemini" | None = .env).

    Returns
    -------
    A retriever - call .invoke(query) to get list[Document] ranked by
    similarity. search_kwargs={"k": RETRIEVER_K} limits each call to the
    four best-matching chunks (enough context, few tokens).

    USAGE IN D2P1.py
    ----------------
        chunks = retriever.invoke(question)      # no LLM involved
        context = format_context(chunks)         # -> one string
    """
    vectorstore = build_vectorstore(provider)
    return vectorstore.as_retriever(
        search_kwargs={"k": RETRIEVER_K},
    )


def format_context(chunks: list[Document], max_chars: int = 3000) -> str:
    """Turn retrieved chunks into one context string for the prompt.

    Parameters
    ----------
    chunks : list[Document]
        Result of retriever.invoke(question).
    max_chars : int
        Safety cap so a burst of chunks cannot blow up the prompt (and the
        free-tier token quota). Chunks are added whole until the cap.

    Returns
    -------
    str
        Context blocks separated by a delimiter, each labelled with its
        source policy so the model can cite the document name.

    Returns prompts.NO_CONTEXT when nothing relevant was found - this is
    the provider-compatible "I don't have company data" signal.
    """
    if not chunks:
        # Imported here to avoid a circular import (prompts does not import
        # rag, but keeping the constant in prompts keeps wording together).
        from common import prompts

        return prompts.NO_CONTEXT

    blocks: list[str] = []
    used = 0
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        block = f"[Chunk {i} | source: {source}]\n{chunk.page_content}"
        if used + len(block) > max_chars and blocks:
            break  # stop before exceeding the token safety cap
        blocks.append(block)
        used += len(block)

    return "\n\n---\n\n".join(blocks)
