"""
EcoRevive Ecology RAG
======================
Retrieval-Augmented Generation for ecological knowledge.

Uses Gemini embeddings to find relevant species, ecoregion, and
restoration information, then augments prompts with that context.
"""

import os
import json
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from .embeddings import GeminiEmbeddings, Document, load_json_as_documents
from .vector_store import SimpleVectorStore, create_and_populate_store


# Path to knowledge base files
KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent / "knowledge_base"
ECOLOGY_DIR = KNOWLEDGE_BASE_DIR / "ecology"
LEGAL_DIR = KNOWLEDGE_BASE_DIR / "legal"

# Cache directory for vector stores
CACHE_DIR = Path(__file__).parent / ".cache"


class EcologyRAG:
    """
    RAG system for ecological knowledge retrieval.
    
    Provides context-aware species recommendations and
    ecoregion information for restoration planning.
    """
    
    def __init__(
        self,
        embedder: Optional[GeminiEmbeddings] = None,
        rebuild_index: bool = False
    ):
        """
        Initialize the Ecology RAG system.
        
        Args:
            embedder: Optional pre-configured embedder
            rebuild_index: If True, rebuild index even if cached version exists
        """
        self.embedder = embedder
        self.store: Optional[SimpleVectorStore] = None
        self._initialized = False
        self.rebuild_index = rebuild_index
    
    def initialize(self):
        """
        Initialize the RAG system (load or build index).
        
        Call this before making queries. Separated from __init__
        to allow lazy initialization.
        """
        if self._initialized:
            return
        
        if self.embedder is None:
            self.embedder = GeminiEmbeddings()
        
        cache_path = CACHE_DIR / "ecology_vectors.json"
        
        if cache_path.exists() and not self.rebuild_index:
            print("Loading cached ecology vector store...")
            self.store = SimpleVectorStore()
            self.store.load(str(cache_path))
        else:
            print("Building ecology vector store...")
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            
            # Find all ecology knowledge base files
            ecology_files = list(ECOLOGY_DIR.glob("*.json"))
            
            if not ecology_files:
                raise FileNotFoundError(f"No JSON files found in {ECOLOGY_DIR}")
            
            self.store = create_and_populate_store(
                json_paths=[str(f) for f in ecology_files],
                embedder=self.embedder,
                save_path=str(cache_path)
            )
        
        self._initialized = True
        print(f"[OK] Ecology RAG initialized with {len(self.store)} documents")
    
    def get_species_recommendations(
        self,
        query: str,
        k: int = 5,
        filter_by_fire: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get species recommendations for a restoration query.
        
        Args:
            query: Natural language query (e.g., "trees for high elevation Sierra Nevada")
            k: Number of species to return
            filter_by_fire: Optional fire name to filter by (e.g., "dixie_2021")
            
        Returns:
            List of species recommendations with relevance scores
        """
        self.initialize()
        
        # Build enhanced query
        enhanced_query = f"Native plant species recommendation: {query}"
        
        # Filter for species documents
        filter_metadata = {"type": "species"}
        
        # Search
        results = self.store.search(
            query=enhanced_query,
            embedder=self.embedder,
            k=k * 2,  # Get extra to filter
            filter_metadata=filter_metadata
        )
        
        # Filter by fire if specified
        if filter_by_fire:
            results = [
                (doc, score) for doc, score in results
                if filter_by_fire in doc.content.lower()
            ]
        
        # Format results
        recommendations = []
        for doc, score in results[:k]:
            recommendations.append({
                "content": doc.content,
                "score": score,
                "species_id": doc.metadata.get("species_id"),
                "scientific_name": doc.metadata.get("scientific_name"),
                "source": doc.metadata.get("source")
            })
        
        return recommendations
    
    def get_ecoregion_info(
        self,
        query: str,
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get ecoregion information for a location query.
        
        Args:
            query: Query about location or ecoregion
            k: Number of results
            
        Returns:
            List of relevant ecoregion documents
        """
        self.initialize()
        
        enhanced_query = f"California ecoregion ecosystem information: {query}"
        
        results = self.store.search(
            query=enhanced_query,
            embedder=self.embedder,
            k=k,
            filter_metadata={"type": "ecoregion"}
        )
        
        return [
            {
                "content": doc.content,
                "score": score,
                "ecoregion_id": doc.metadata.get("ecoregion_id"),
                "source": doc.metadata.get("source")
            }
            for doc, score in results
        ]
    
    def get_invasive_species_warnings(
        self,
        region: str
    ) -> List[Dict[str, Any]]:
        """
        Get invasive species warnings for a region.
        
        Args:
            region: Region name or description
            
        Returns:
            List of invasive species warnings
        """
        self.initialize()
        
        query = f"Invasive species threat warning for {region} California post-fire"
        
        results = self.store.search(
            query=query,
            embedder=self.embedder,
            k=5,
            filter_metadata={"type": "invasive_warning"}
        )
        
        return [
            {
                "content": doc.content,
                "score": score
            }
            for doc, score in results
        ]
    
    def get_restoration_context(
        self,
        location_description: str,
        severity_level: str = "high",
        k: int = 5
    ) -> str:
        """
        Get comprehensive restoration context for a Gemini prompt.
        
        This is the main method for augmenting Gemini prompts with
        relevant ecological knowledge.
        
        Args:
            location_description: Description of the restoration site
            severity_level: "low", "moderate", or "high"
            k: Number of documents to include
            
        Returns:
            Formatted context string for prompt augmentation
        """
        self.initialize()
        
        # Build comprehensive query
        query = f"""
        Post-fire restoration site in California.
        Location: {location_description}
        Burn severity: {severity_level}
        Need: ecoregion classification, native species recommendations, 
        restoration approach, invasive species warnings
        """
        
        # Search all document types
        results = self.store.search(
            query=query,
            embedder=self.embedder,
            k=k
        )
        
        # Format as context block
        context_parts = ["RETRIEVED ECOLOGICAL KNOWLEDGE:"]
        context_parts.append("-" * 40)
        
        for i, (doc, score) in enumerate(results, 1):
            doc_type = doc.metadata.get("type", "unknown").upper()
            context_parts.append(f"[{i}] {doc_type} (relevance: {score:.2f})")
            context_parts.append(doc.content)
            context_parts.append("")
        
        context_parts.append("-" * 40)
        
        return "\n".join(context_parts)


class LegalRAG:
    """
    RAG system for legal and land ownership information.
    
    Provides land ownership, permit requirements, and contact
    information for restoration planning.
    """
    
    def __init__(
        self,
        embedder: Optional[GeminiEmbeddings] = None,
        rebuild_index: bool = False
    ):
        """
        Initialize the Legal RAG system.
        
        Args:
            embedder: Optional pre-configured embedder
            rebuild_index: If True, rebuild index even if cached version exists
        """
        self.embedder = embedder
        self.store: Optional[SimpleVectorStore] = None
        self._initialized = False
        self.rebuild_index = rebuild_index
    
    def initialize(self):
        """Initialize the RAG system."""
        if self._initialized:
            return
        
        if self.embedder is None:
            self.embedder = GeminiEmbeddings()
        
        cache_path = CACHE_DIR / "legal_vectors.json"
        
        if cache_path.exists() and not self.rebuild_index:
            print("Loading cached legal vector store...")
            self.store = SimpleVectorStore()
            self.store.load(str(cache_path))
        else:
            print("Building legal vector store...")
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            
            legal_files = list(LEGAL_DIR.glob("*.json"))
            
            if not legal_files:
                raise FileNotFoundError(f"No JSON files found in {LEGAL_DIR}")
            
            self.store = create_and_populate_store(
                json_paths=[str(f) for f in legal_files],
                embedder=self.embedder,
                save_path=str(cache_path)
            )
        
        self._initialized = True
        print(f"[OK] Legal RAG initialized with {len(self.store)} documents")
    
    def get_land_ownership(
        self,
        location_description: str,
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get land ownership information for a location.
        
        Args:
            location_description: Description of the location
            k: Number of results
            
        Returns:
            List of relevant land ownership documents
        """
        self.initialize()
        
        query = f"Land ownership management agency for: {location_description}"
        
        results = self.store.search(
            query=query,
            embedder=self.embedder,
            k=k,
            filter_metadata={"type": "national_forest"}
        )
        
        return [
            {
                "content": doc.content,
                "score": score,
                "land_id": doc.metadata.get("land_id"),
                "source": doc.metadata.get("source")
            }
            for doc, score in results
        ]
    
    def get_permit_requirements(
        self,
        activity_type: str,
        land_type: str = "national_forest"
    ) -> List[Dict[str, Any]]:
        """
        Get permit requirements for restoration activities.
        
        Args:
            activity_type: Type of activity (e.g., "tree planting", "volunteer event")
            land_type: Type of land (e.g., "national_forest", "blm", "state_park")
            
        Returns:
            List of relevant permit information
        """
        self.initialize()
        
        query = f"Permit requirements for {activity_type} on {land_type} land"
        
        results = self.store.search(
            query=query,
            embedder=self.embedder,
            k=3,
            filter_metadata={"type": "permit"}
        )
        
        return [
            {
                "content": doc.content,
                "score": score,
                "permit_id": doc.metadata.get("permit_id")
            }
            for doc, score in results
        ]
    
    def get_legal_context(
        self,
        location_description: str,
        activity_type: str = "restoration"
    ) -> str:
        """
        Get comprehensive legal context for a Gemini prompt.
        
        Args:
            location_description: Description of the restoration site
            activity_type: Type of planned activity
            
        Returns:
            Formatted context string for prompt augmentation
        """
        self.initialize()
        
        query = f"""
        Legal and permit information for {activity_type} activities.
        Location: {location_description}
        Need: land ownership, managing agency, contact information, required permits
        """
        
        results = self.store.search(
            query=query,
            embedder=self.embedder,
            k=5
        )
        
        context_parts = ["RETRIEVED LEGAL/OWNERSHIP INFORMATION:"]
        context_parts.append("-" * 40)
        
        for i, (doc, score) in enumerate(results, 1):
            doc_type = doc.metadata.get("type", "unknown").upper()
            context_parts.append(f"[{i}] {doc_type} (relevance: {score:.2f})")
            context_parts.append(doc.content)
            context_parts.append("")
        
        context_parts.append("-" * 40)
        
        return "\n".join(context_parts)


class CombinedRAG:
    """
    Combined RAG that searches both ecology and legal knowledge bases.
    
    Use this for comprehensive restoration planning queries.
    """
    
    def __init__(
        self,
        embedder: Optional[GeminiEmbeddings] = None,
        rebuild_index: bool = False
    ):
        """Initialize combined RAG."""
        self.embedder = embedder
        self.ecology_rag = EcologyRAG(embedder, rebuild_index)
        self.legal_rag = LegalRAG(embedder, rebuild_index)
    
    def initialize(self):
        """Initialize both RAG systems."""
        if self.embedder is None:
            self.embedder = GeminiEmbeddings()
            self.ecology_rag.embedder = self.embedder
            self.legal_rag.embedder = self.embedder
        
        self.ecology_rag.initialize()
        self.legal_rag.initialize()
    
    def get_full_context(
        self,
        location_description: str,
        severity_level: str = "high",
        activity_type: str = "restoration"
    ) -> str:
        """
        Get complete context (ecology + legal) for prompt augmentation.
        
        This is the main method for RAG-augmented Gemini queries.
        
        Args:
            location_description: Description of the site
            severity_level: Burn severity level
            activity_type: Planned activity type
            
        Returns:
            Combined context string
        """
        self.initialize()
        
        ecology_context = self.ecology_rag.get_restoration_context(
            location_description,
            severity_level
        )
        
        legal_context = self.legal_rag.get_legal_context(
            location_description,
            activity_type
        )
        
        return f"{ecology_context}\n\n{legal_context}"


if __name__ == "__main__":
    print("=" * 60)
    print("EcoRevive RAG Test")
    print("=" * 60)
    
    try:
        # Test ecology RAG
        print("\n--- Testing Ecology RAG ---")
        ecology_rag = EcologyRAG(rebuild_index=True)
        ecology_rag.initialize()
        
        # Search for species
        species = ecology_rag.get_species_recommendations(
            "fire-resistant trees for Sierra Nevada high elevation"
        )
        
        print(f"\nSpecies recommendations:")
        for s in species[:3]:
            print(f"   [{s['score']:.2f}] {s.get('scientific_name', 'Unknown')}")
        
        # Get ecoregion info
        ecoregions = ecology_rag.get_ecoregion_info(
            "Dixie Fire area Plumas County"
        )
        
        print(f"\nEcoregion info:")
        for e in ecoregions[:2]:
            print(f"   [{e['score']:.2f}] {e.get('ecoregion_id', 'Unknown')}")
        
    except ValueError as e:
        print(f"\n[WARNING] {e}")
        print("Set GOOGLE_API_KEY to test the RAG system.")
    except FileNotFoundError as e:
        print(f"\n[WARNING] {e}")
        print("Create knowledge base files first.")
