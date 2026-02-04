"""
EcoRevive Vector Store
=======================
Simple file-based vector store for storing and searching document embeddings.

No external database required - uses NumPy for similarity search.
Optimized for the ~100-500 document scale of our knowledge bases.
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .embeddings import Document, GeminiEmbeddings


class SimpleVectorStore:
    """
    Lightweight vector store for document embeddings.
    
    Features:
    - In-memory storage with file persistence
    - Cosine similarity search
    - Metadata filtering
    - No external dependencies beyond NumPy
    
    Suitable for knowledge bases up to ~10,000 documents.
    """
    
    def __init__(self, embedding_dimension: int = 768):
        """
        Initialize the vector store.
        
        Args:
            embedding_dimension: Dimension of embedding vectors (768 for text-embedding-004)
        """
        self.embedding_dimension = embedding_dimension
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None
        self._index_built = False
    
    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None
    ):
        """
        Add documents to the store.
        
        Args:
            documents: List of Document objects
            embeddings: Optional pre-computed embeddings (same order as documents)
        """
        if embeddings is not None:
            if len(embeddings) != len(documents):
                raise ValueError("Number of embeddings must match number of documents")
            
            for doc, emb in zip(documents, embeddings):
                doc.embedding = emb
        
        # Verify all documents have embeddings
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"Document {doc.doc_id} has no embedding")
        
        self.documents.extend(documents)
        self._index_built = False
        
        print(f"   Added {len(documents)} documents (total: {len(self.documents)})")
    
    def build_index(self):
        """
        Build the search index (convert embeddings to numpy array).
        
        Call this after adding all documents for efficient search.
        """
        if not self.documents:
            raise ValueError("No documents in store")
        
        # Stack embeddings into numpy array
        embedding_list = [doc.embedding for doc in self.documents]
        self.embeddings = np.array(embedding_list, dtype=np.float32)
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / np.maximum(norms, 1e-10)
        
        self._index_built = True
        print(f"   [OK] Index built: {len(self.documents)} documents, {self.embedding_dimension} dimensions")
    
    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: The query vector
            k: Number of results to return
            filter_metadata: Optional metadata filters (exact match)
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of (Document, score) tuples, sorted by score descending
        """
        if not self._index_built:
            self.build_index()
        
        # Normalize query
        query = np.array(query_embedding, dtype=np.float32)
        query = query / np.maximum(np.linalg.norm(query), 1e-10)
        
        # Compute cosine similarity (dot product of normalized vectors)
        scores = np.dot(self.embeddings, query)
        
        # Apply metadata filter if provided
        if filter_metadata:
            mask = np.ones(len(self.documents), dtype=bool)
            for key, value in filter_metadata.items():
                for i, doc in enumerate(self.documents):
                    if doc.metadata.get(key) != value:
                        mask[i] = False
            scores = np.where(mask, scores, -1)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k * 2]  # Get extra in case of filtering
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= score_threshold and len(results) < k:
                results.append((self.documents[idx], score))
        
        return results
    
    def search(
        self,
        query: str,
        embedder: GeminiEmbeddings,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search using a text query.
        
        Convenience method that handles embedding the query.
        
        Args:
            query: The search query text
            embedder: GeminiEmbeddings instance
            k: Number of results
            filter_metadata: Optional metadata filters
            
        Returns:
            List of (Document, score) tuples
        """
        query_embedding = embedder.embed_query(query)
        return self.similarity_search(query_embedding, k, filter_metadata)
    
    def save(self, path: str):
        """
        Save the vector store to disk.
        
        Args:
            path: Path to save file (JSON format)
        """
        data = {
            "embedding_dimension": self.embedding_dimension,
            "documents": [doc.to_dict() for doc in self.documents]
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"   [OK] Saved vector store to {path}")
    
    def load(self, path: str):
        """
        Load the vector store from disk.
        
        Args:
            path: Path to saved file
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.embedding_dimension = data.get("embedding_dimension", 768)
        self.documents = [Document.from_dict(d) for d in data["documents"]]
        self._index_built = False
        
        print(f"   Loaded {len(self.documents)} documents from {path}")
    
    def __len__(self):
        return len(self.documents)
    
    def __repr__(self):
        return f"SimpleVectorStore(documents={len(self.documents)}, dim={self.embedding_dimension})"


def create_and_populate_store(
    json_paths: List[str],
    embedder: GeminiEmbeddings,
    save_path: Optional[str] = None
) -> SimpleVectorStore:
    """
    Create a vector store from knowledge base JSON files.
    
    This is the main entry point for building the RAG index.
    
    Args:
        json_paths: List of paths to JSON knowledge base files
        embedder: GeminiEmbeddings instance
        save_path: Optional path to save the populated store
        
    Returns:
        Populated SimpleVectorStore
    """
    from .embeddings import load_json_as_documents
    
    store = SimpleVectorStore()
    
    print("Loading and embedding knowledge base...")
    
    for path in json_paths:
        print(f"\n   Processing: {Path(path).name}")
        
        # Load documents from JSON
        documents = load_json_as_documents(path)
        print(f"   Loaded {len(documents)} documents")
        
        if documents:
            # Embed documents
            texts = [doc.content for doc in documents]
            embeddings = embedder.embed_documents(texts)
            
            # Add to store
            store.add_documents(documents, embeddings)
    
    # Build index
    store.build_index()
    
    # Save if path provided
    if save_path:
        store.save(save_path)
    
    return store


if __name__ == "__main__":
    print("=" * 60)
    print("Vector Store Test")
    print("=" * 60)
    
    # Test basic functionality without API
    store = SimpleVectorStore()
    
    # Create test documents with fake embeddings
    test_docs = [
        Document("Ponderosa Pine grows at 1000-2500m elevation", metadata={"type": "species"}),
        Document("Black Oak resprouts vigorously after fire", metadata={"type": "species"}),
        Document("Plumas National Forest covers 1.1 million acres", metadata={"type": "land"}),
    ]
    
    # Fake embeddings for testing
    fake_embeddings = [
        np.random.randn(768).tolist(),
        np.random.randn(768).tolist(),
        np.random.randn(768).tolist(),
    ]
    
    store.add_documents(test_docs, fake_embeddings)
    store.build_index()
    
    print(f"\n[OK] Store created: {store}")
    
    # Test search with fake query embedding
    fake_query = np.random.randn(768).tolist()
    results = store.similarity_search(fake_query, k=2)
    
    print(f"\nSearch results (with random embeddings):")
    for doc, score in results:
        print(f"   [{score:.3f}] {doc.content[:50]}...")
