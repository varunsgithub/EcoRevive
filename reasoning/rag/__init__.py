"""
EcoRevive RAG Module
====================
Retrieval-Augmented Generation for ecology and legal knowledge.

Uses Gemini's text-embedding-004 for document vectorization and
semantic search to augment Gemini prompts with domain-specific knowledge.

Modules:
    embeddings: Gemini embedding client and document utilities
    vector_store: Simple file-based vector storage
    ecology_rag: RAG for species and ecoregion knowledge

Example:
    >>> from reasoning.rag import EcologyRAG, LegalRAG, CombinedRAG
    >>> 
    >>> # Get ecological context for a location
    >>> rag = EcologyRAG()
    >>> context = rag.get_restoration_context("Dixie Fire, Plumas County")
    >>> 
    >>> # Use context in Gemini prompt
    >>> prompt = f"{context}\\n\\nQuestion: What species should we plant?"
"""

from .embeddings import (
    GeminiEmbeddings,
    Document,
    chunk_text,
    load_json_as_documents
)

from .vector_store import (
    SimpleVectorStore,
    create_and_populate_store
)

from .ecology_rag import (
    EcologyRAG,
    LegalRAG,
    CombinedRAG
)

__all__ = [
    # Embeddings
    "GeminiEmbeddings",
    "Document",
    "chunk_text",
    "load_json_as_documents",
    
    # Vector Store
    "SimpleVectorStore",
    "create_and_populate_store",
    
    # RAG Systems
    "EcologyRAG",
    "LegalRAG",
    "CombinedRAG",
]
