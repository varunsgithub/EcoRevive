"""
EcoRevive Embedding Module
===========================
Uses Gemini's text-embedding-004 model to create vector embeddings
for the ecology and legal knowledge bases.

This module demonstrates Gemini's embedding API for RAG applications.
"""

import os
import json
import hashlib
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

import google.generativeai as genai


class GeminiEmbeddings:
    """
    Gemini embedding client for document vectorization.
    
    Uses text-embedding-004 to create high-quality embeddings
    for semantic search and retrieval.
    """
    
    MODEL_NAME = "models/text-embedding-004"
    EMBEDDING_DIMENSION = 768  # text-embedding-004 produces 768-dim vectors
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the embedding client.
        
        Args:
            api_key: Google API key. If not provided, reads from GOOGLE_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        genai.configure(api_key=self.api_key)
        
        print(f"✅ Gemini Embeddings initialized")
        print(f"   Model: {self.MODEL_NAME}")
        print(f"   Dimension: {self.EMBEDDING_DIMENSION}")
    
    def embed_text(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """
        Embed a single text string.
        
        Args:
            text: The text to embed
            task_type: The intended use of the embedding:
                - "retrieval_document": For indexing documents
                - "retrieval_query": For search queries
                - "semantic_similarity": For comparing texts
                - "classification": For text classification
                
        Returns:
            List of floats representing the embedding vector
        """
        result = genai.embed_content(
            model=self.MODEL_NAME,
            content=text,
            task_type=task_type
        )
        return result['embedding']
    
    def embed_documents(
        self,
        texts: List[str],
        task_type: str = "retrieval_document",
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Embed multiple documents.
        
        Args:
            texts: List of texts to embed
            task_type: The intended use (see embed_text)
            show_progress: Whether to print progress
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            if show_progress and (i + 1) % 10 == 0:
                print(f"   Embedding documents: {i + 1}/{total}")
            
            embedding = self.embed_text(text, task_type)
            embeddings.append(embedding)
        
        if show_progress:
            print(f"   ✅ Embedded {total} documents")
            
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a search query.
        
        Uses task_type="retrieval_query" for optimal search performance.
        
        Args:
            query: The search query
            
        Returns:
            Query embedding vector
        """
        return self.embed_text(query, task_type="retrieval_query")
    
    def embed_batch(
        self,
        texts: List[str],
        task_type: str = "retrieval_document",
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Embed texts in batches for efficiency.
        
        Note: Gemini's embed_content currently processes one at a time,
        but this method provides a consistent interface for future
        batch endpoint support.
        
        Args:
            texts: List of texts to embed
            task_type: The intended use
            batch_size: Number of texts per batch
            
        Returns:
            List of embedding vectors
        """
        # For now, delegate to embed_documents
        # Future: Use batch endpoint when available
        return self.embed_documents(texts, task_type)


class Document:
    """A document with content and metadata for RAG."""
    
    def __init__(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ):
        """
        Create a document.
        
        Args:
            content: The text content
            metadata: Optional metadata dict
            doc_id: Optional document ID (generated if not provided)
        """
        self.content = content
        self.metadata = metadata or {}
        self.doc_id = doc_id or self._generate_id(content)
        self.embedding: Optional[List[float]] = None
    
    def _generate_id(self, content: str) -> str:
        """Generate a unique ID from content hash."""
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create from dictionary."""
        doc = cls(
            content=data["content"],
            metadata=data.get("metadata", {}),
            doc_id=data.get("doc_id")
        )
        doc.embedding = data.get("embedding")
        return doc
    
    def __repr__(self):
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Document(id={self.doc_id}, content='{preview}')"


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: List[str] = None
) -> List[str]:
    """
    Split text into overlapping chunks for embedding.
    
    Args:
        text: The text to chunk
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        separators: List of separators to try, in order of priority
        
    Returns:
        List of text chunks
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]
    
    chunks = []
    current_chunk = ""
    
    # Simple chunking by sentences for now
    sentences = text.replace("\n\n", "\n").split(". ")
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Add period back if it was removed
        if not sentence.endswith("."):
            sentence += "."
        
        # Check if adding this sentence would exceed chunk size
        if len(current_chunk) + len(sentence) + 1 > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep overlap
            overlap_text = current_chunk[-chunk_overlap:] if chunk_overlap > 0 else ""
            current_chunk = overlap_text + " " + sentence
        else:
            current_chunk = (current_chunk + " " + sentence).strip()
    
    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def load_json_as_documents(
    json_path: Union[str, Path],
    content_fields: List[str] = None,
    metadata_fields: List[str] = None
) -> List[Document]:
    """
    Load a JSON file and convert to documents for embedding.
    
    This function handles the specific structure of our knowledge base files.
    
    Args:
        json_path: Path to JSON file
        content_fields: Fields to include in document content
        metadata_fields: Fields to include in metadata
        
    Returns:
        List of Document objects
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    documents = []
    
    # Handle our specific knowledge base formats
    
    # Ecoregions format
    if "ecoregions" in data:
        for eco_id, eco_data in data["ecoregions"].items():
            content = _format_ecoregion_document(eco_id, eco_data)
            doc = Document(
                content=content,
                metadata={
                    "source": "california_ecoregions",
                    "ecoregion_id": eco_id,
                    "type": "ecoregion"
                }
            )
            documents.append(doc)
    
    # Species catalog format
    if "species" in data:
        for species_data in data["species"]:
            content = _format_species_document(species_data)
            doc = Document(
                content=content,
                metadata={
                    "source": "native_species_catalog",
                    "species_id": species_data.get("id"),
                    "scientific_name": species_data.get("scientific_name"),
                    "type": "species"
                }
            )
            documents.append(doc)
    
    # Invasive species warnings
    if "invasive_species_warnings" in data:
        for invasive in data["invasive_species_warnings"]:
            content = _format_invasive_document(invasive)
            doc = Document(
                content=content,
                metadata={
                    "source": "invasive_species",
                    "type": "invasive_warning"
                }
            )
            documents.append(doc)
    
    # Federal lands format
    if "national_forests" in data:
        for nf_id, nf_data in data["national_forests"].items():
            content = _format_national_forest_document(nf_id, nf_data)
            doc = Document(
                content=content,
                metadata={
                    "source": "federal_state_lands",
                    "land_id": nf_id,
                    "type": "national_forest"
                }
            )
            documents.append(doc)
    
    if "permit_types" in data:
        for permit_id, permit_data in data["permit_types"].items():
            content = _format_permit_document(permit_id, permit_data)
            doc = Document(
                content=content,
                metadata={
                    "source": "permit_types",
                    "permit_id": permit_id,
                    "type": "permit"
                }
            )
            documents.append(doc)
    
    # Ecological principles format (new)
    if "core_principles" in data:
        for principle_id, principle_data in data["core_principles"].items():
            content = _format_principle_document(principle_id, principle_data)
            doc = Document(
                content=content,
                metadata={
                    "source": "ecological_principles",
                    "principle_id": principle_id,
                    "type": "ecological_principle"
                }
            )
            documents.append(doc)
    
    # Scientific literature format
    if "scientific_literature" in data:
        lit = data["scientific_literature"]
        
        # Essential references
        for ref in lit.get("essential_references", []):
            content = _format_reference_document(ref)
            doc = Document(
                content=content,
                metadata={
                    "source": "scientific_literature",
                    "type": "essential_reference"
                }
            )
            documents.append(doc)
        
        # Key papers
        for paper in lit.get("key_papers", []):
            content = _format_paper_document(paper)
            doc = Document(
                content=content,
                metadata={
                    "source": "scientific_literature",
                    "type": "key_paper"
                }
            )
            documents.append(doc)
    
    # Policy frameworks
    if "policy_frameworks" in data:
        for category, policies in data["policy_frameworks"].items():
            for policy in policies:
                content = _format_policy_document(policy, category)
                doc = Document(
                    content=content,
                    metadata={
                        "source": "policy_frameworks",
                        "category": category,
                        "type": "policy"
                    }
                )
                documents.append(doc)
    
    # Decision frameworks
    if "decision_frameworks" in data:
        for framework_id, framework_data in data["decision_frameworks"].items():
            content = _format_decision_framework_document(framework_id, framework_data)
            doc = Document(
                content=content,
                metadata={
                    "source": "decision_frameworks",
                    "framework_id": framework_id,
                    "type": "decision_framework"
                }
            )
            documents.append(doc)
    
    # Legal frameworks (international, federal, state)
    for framework_key in ["international_frameworks", "us_federal_laws", "california_state_laws"]:
        if framework_key in data:
            for law_id, law_data in data[framework_key].items():
                content = _format_law_document(law_id, law_data, framework_key)
                doc = Document(
                    content=content,
                    metadata={
                        "source": framework_key,
                        "law_id": law_id,
                        "type": "legal_framework"
                    }
                )
                documents.append(doc)
    
    # Indigenous rights
    if "indigenous_rights" in data:
        for right_id, right_data in data["indigenous_rights"].items():
            content = _format_indigenous_rights_document(right_id, right_data)
            doc = Document(
                content=content,
                metadata={
                    "source": "indigenous_rights",
                    "right_id": right_id,
                    "type": "indigenous_rights"
                }
            )
            documents.append(doc)
    
    # Ethical considerations
    if "ethical_considerations" in data:
        for issue_id, issue_data in data["ethical_considerations"].items():
            content = _format_ethical_document(issue_id, issue_data)
            doc = Document(
                content=content,
                metadata={
                    "source": "ethical_considerations",
                    "issue_id": issue_id,
                    "type": "ethical_consideration"
                }
            )
            documents.append(doc)
    
    # Site-specific legal templates
    if "site_specific_templates" in data:
        for site_type, template_data in data["site_specific_templates"].items():
            content = _format_site_legal_template(site_type, template_data)
            doc = Document(
                content=content,
                metadata={
                    "source": "legal_templates",
                    "site_type": site_type,
                    "type": "legal_template"
                }
            )
            documents.append(doc)
    
    return documents


def _format_ecoregion_document(eco_id: str, data: Dict) -> str:
    """Format ecoregion data as a document string."""
    lines = [
        f"Ecoregion: {data.get('name', eco_id)}",
        f"EPA Ecoregion: {data.get('epa_ecoregion', 'Unknown')}",
        f"Elevation range: {data.get('elevation_range_m', [0, 0])[0]}-{data.get('elevation_range_m', [0, 0])[1]} meters",
        f"Precipitation: {data.get('precipitation_mm_annual', [0, 0])[0]}-{data.get('precipitation_mm_annual', [0, 0])[1]} mm annually",
        f"Climate: {data.get('climate_type', 'Unknown')}",
    ]
    
    # Add dominant species
    if "dominant_species" in data:
        species_list = [f"{s['common']} ({s['scientific']})" for s in data["dominant_species"][:5]]
        lines.append(f"Dominant species: {', '.join(species_list)}")
    
    # Add fire regime
    if "fire_regime" in data:
        fr = data["fire_regime"]
        lines.append(f"Fire regime: {fr.get('fire_type', 'Unknown')}, interval {fr.get('historical_interval_years', [0, 0])} years")
    
    # Add recovery info
    if "post_fire_recovery" in data:
        pfr = data["post_fire_recovery"]
        lines.append(f"Recovery: Ground cover in {pfr.get('ground_cover_return_years', [0, 0])} years, canopy closure in {pfr.get('canopy_closure_years', [0, 0])} years")
        lines.append(f"Carbon sequestration rate: {pfr.get('carbon_sequestration_rate_tonnes_ha_year', 0)} tonnes CO2/ha/year")
    
    # Add associated fires
    if "fires_in_region" in data:
        lines.append(f"Fires in this region: {', '.join(data['fires_in_region'])}")
    
    return "\n".join(lines)


def _format_species_document(data: Dict) -> str:
    """Format species data as a document string."""
    lines = [
        f"Species: {data.get('common_name', 'Unknown')}",
        f"Scientific name: {data.get('scientific_name', 'Unknown')}",
        f"Family: {data.get('family', 'Unknown')}",
        f"Life form: {data.get('life_form', 'Unknown')}",
        f"Native range: {data.get('native_range', 'Unknown')}",
    ]
    
    # Habitat
    if "habitat" in data:
        h = data["habitat"]
        lines.append(f"Elevation: {h.get('elevation_range_m', [0, 0])[0]}-{h.get('elevation_range_m', [0, 0])[1]} meters")
        lines.append(f"Drought tolerance: {h.get('drought_tolerance', 'Unknown')}")
    
    # Fire ecology
    if "fire_ecology" in data:
        fe = data["fire_ecology"]
        lines.append(f"Fire adaptation: {fe.get('fire_adaptation', 'Unknown')}")
        lines.append(f"Post-fire response: {fe.get('post_fire_response', 'Unknown')}")
        if fe.get('sprouting_ability') and fe['sprouting_ability'] != 'none':
            lines.append(f"Sprouting ability: {fe['sprouting_ability']}")
    
    # Restoration
    if "restoration" in data:
        r = data["restoration"]
        lines.append(f"Priority for planting: {r.get('priority_for_planting', 'Unknown')}")
        lines.append(f"Planting density: {r.get('planting_density_stems_ha', 'Unknown')} stems/hectare")
        lines.append(f"Survival rate: {r.get('survival_rate_typical', 0):.0%}")
        lines.append(f"Nursery availability: {r.get('nursery_availability', 'Unknown')}")
    
    # Notes
    if "ecological_notes" in data:
        lines.append(f"Notes: {data['ecological_notes']}")
    
    return "\n".join(lines)


def _format_invasive_document(data: Dict) -> str:
    """Format invasive species warning as a document string."""
    return (
        f"INVASIVE SPECIES WARNING: {data.get('common_name', 'Unknown')} "
        f"({data.get('scientific_name', 'Unknown')}). "
        f"Threat level: {data.get('threat_level', 'Unknown')}. "
        f"Post-fire risk: {data.get('post_fire_risk', 'Unknown')}. "
        f"Regions affected: {', '.join(data.get('regions', []))}. "
        f"Management: {data.get('management', 'Unknown')}"
    )


def _format_national_forest_document(nf_id: str, data: Dict) -> str:
    """Format national forest data as a document string."""
    lines = [
        f"Land Unit: {data.get('name', nf_id)}",
        f"Agency: {data.get('agency', 'Unknown')}",
        f"Region: {data.get('region', 'Unknown')}",
        f"Total acres: {data.get('total_acres', 0):,}",
    ]
    
    if "fires_affected" in data:
        lines.append(f"Fires in this unit: {', '.join(data['fires_affected'])}")
    
    if "supervisor_office" in data:
        so = data["supervisor_office"]
        lines.append(f"Supervisor Office: {so.get('address', 'Unknown')}")
        lines.append(f"Phone: {so.get('phone', 'Unknown')}")
    
    if "ranger_districts" in data:
        for rd in data["ranger_districts"]:
            lines.append(f"Ranger District: {rd.get('name', 'Unknown')}, Phone: {rd.get('phone', 'Unknown')}")
    
    if "permits_required" in data:
        lines.append("Permits required: " + ", ".join(f"{k}: {v}" for k, v in data["permits_required"].items()))
    
    return "\n".join(lines)


def _format_permit_document(permit_id: str, data: Dict) -> str:
    """Format permit data as a document string."""
    return (
        f"Permit Type: {data.get('name', permit_id)}. "
        f"Agency: {data.get('agency', 'Unknown')}. "
        f"Purpose: {data.get('purpose', 'Unknown')}. "
        f"Requirements: {', '.join(data.get('requirements', []))}. "
        f"Processing time: {data.get('processing_time_days', [0, 0])} days. "
        f"Cost: {data.get('cost', 'Unknown')}."
    )


def _format_principle_document(principle_id: str, data: Dict) -> str:
    """Format ecological principle as a document string."""
    lines = [
        f"ECOLOGICAL PRINCIPLE: {data.get('name', principle_id)}",
        f"Description: {data.get('description', '')}",
    ]
    
    # Key concepts
    if "key_concepts" in data:
        lines.append("Key concepts:")
        for concept in data["key_concepts"]:
            lines.append(f"  - {concept}")
    
    # Decision rules
    if "decision_rules" in data:
        lines.append("Decision rules:")
        for rule in data["decision_rules"]:
            lines.append(f"  - {rule}")
    
    # California specific info
    if "california_specific" in data:
        lines.append("California-specific:")
        for item in data["california_specific"]:
            lines.append(f"  - {item}")
    
    # Recovery timeframes
    if "recovery_timeframes" in data:
        lines.append("Recovery timeframes:")
        for eco_type, info in data["recovery_timeframes"].items():
            if isinstance(info, dict):
                years = info.get("years", [0, 0])
                lines.append(f"  - {eco_type.replace('_', ' ').title()}: {years[0]}-{years[1]} years")
    
    return "\n".join(lines)


def _format_reference_document(data: Dict) -> str:
    """Format scientific reference as a document string."""
    return (
        f"REFERENCE: {data.get('title', 'Unknown')}. "
        f"Author: {data.get('author', 'Unknown')}. "
        f"Type: {data.get('type', 'Unknown')}. "
        f"Key concept: {data.get('key_concept', '')}."
    )


def _format_paper_document(data: Dict) -> str:
    """Format scientific paper as a document string."""
    lines = [
        f"RESEARCH PAPER: {data.get('title', 'Unknown')}",
        f"Authors: {data.get('authors', 'Unknown')}",
        f"Journal: {data.get('journal', 'Unknown')} ({data.get('year', 'Unknown')})",
        f"Key finding: {data.get('key_finding', '')}",
    ]
    if data.get("doi"):
        lines.append(f"DOI: {data['doi']}")
    return "\n".join(lines)


def _format_policy_document(data: Dict, category: str) -> str:
    """Format policy document as a string."""
    lines = [
        f"POLICY ({category.upper()}): {data.get('name', 'Unknown')}",
        f"Description: {data.get('description', '')}",
    ]
    if data.get("targets"):
        lines.append(f"Targets: {data['targets']}")
    if data.get("funding"):
        lines.append(f"Funding: {data['funding']}")
    if data.get("url"):
        lines.append(f"URL: {data['url']}")
    return "\n".join(lines)


def _format_decision_framework_document(framework_id: str, data: Dict) -> str:
    """Format decision framework as a document string."""
    lines = [
        f"DECISION FRAMEWORK: {framework_id.replace('_', ' ').title()}",
        f"Description: {data.get('description', '')}",
    ]
    
    # Passive restoration conditions
    if "passive_restoration_conditions" in data:
        lines.append("Conditions favoring PASSIVE restoration (natural recovery):")
        for condition in data["passive_restoration_conditions"]:
            lines.append(f"  - {condition}")
    
    # Active restoration conditions
    if "active_restoration_conditions" in data:
        lines.append("Conditions requiring ACTIVE restoration (planting):")
        for condition in data["active_restoration_conditions"]:
            lines.append(f"  - {condition}")
    
    # Ecological criteria
    if "ecological_criteria" in data:
        lines.append("Ecological prioritization criteria:")
        for criterion in data["ecological_criteria"]:
            lines.append(f"  - {criterion}")
    
    # Social criteria
    if "social_criteria" in data:
        lines.append("Social prioritization criteria:")
        for criterion in data["social_criteria"]:
            lines.append(f"  - {criterion}")
    
    return "\n".join(lines)


def _format_law_document(law_id: str, data: Dict, framework_type: str) -> str:
    """Format a legal framework document."""
    level = framework_type.replace("_", " ").title()
    lines = [
        f"LEGAL FRAMEWORK ({level}): {data.get('name', law_id)}",
    ]
    
    if data.get("code"):
        lines.append(f"Legal code: {data['code']}")
    if data.get("agency"):
        lines.append(f"Agency: {data['agency']}")
    if data.get("type"):
        lines.append(f"Type: {data['type']}")
    
    # Constraints
    if "constraints" in data:
        lines.append("Legal constraints:")
        for c in data["constraints"]:
            lines.append(f"  - {c}")
    
    # Key provisions
    if "key_provisions" in data:
        lines.append("Key provisions:")
        for k, v in data["key_provisions"].items():
            lines.append(f"  - {k}: {v}")
    
    # Implications
    if "implications" in data:
        lines.append("Implications for restoration:")
        for imp in data["implications"]:
            lines.append(f"  - {imp}")
    
    if "implications_for_restoration" in data:
        lines.append("Implications for restoration:")
        for imp in data["implications_for_restoration"]:
            lines.append(f"  - {imp}")
    
    return "\n".join(lines)


def _format_indigenous_rights_document(right_id: str, data: Dict) -> str:
    """Format indigenous rights information."""
    lines = [
        f"INDIGENOUS RIGHTS: {data.get('name', right_id)}",
    ]
    
    if data.get("source"):
        lines.append(f"Source: {data['source']}")
    if data.get("definition"):
        lines.append(f"Definition: {data['definition']}")
    
    # Key principles
    if "key_principles" in data:
        lines.append("Key principles:")
        for p in data["key_principles"]:
            lines.append(f"  - {p}")
    
    # Absolute constraints
    if "constraints_absolute" in data:
        lines.append("ABSOLUTE CONSTRAINTS (non-negotiable):")
        for c in data["constraints_absolute"]:
            lines.append(f"  ⚠️ {c}")
    
    # General constraints
    if "constraints" in data:
        lines.append("Constraints:")
        for c in data["constraints"]:
            lines.append(f"  - {c}")
    
    # Requirements
    if "requirements" in data:
        lines.append("Requirements:")
        for r in data["requirements"]:
            lines.append(f"  - {r}")
    
    # Implications
    if "implications" in data:
        lines.append("Implications for restoration:")
        for imp in data["implications"]:
            lines.append(f"  - {imp}")
    
    return "\n".join(lines)


def _format_ethical_document(issue_id: str, data: Dict) -> str:
    """Format ethical consideration document."""
    lines = [
        f"ETHICAL CONSIDERATION: {issue_id.replace('_', ' ').title()}",
    ]
    
    if data.get("risk"):
        lines.append(f"Risk: {data['risk']}")
    if data.get("issue"):
        lines.append(f"Issue: {data['issue']}")
    
    # Examples
    if "examples" in data:
        lines.append("Examples:")
        for ex in data["examples"]:
            lines.append(f"  - {ex}")
    
    # Mitigation strategies
    if "mitigation" in data:
        lines.append("Mitigation strategies:")
        for m in data["mitigation"]:
            lines.append(f"  - {m}")
    
    # Handle nested structure for data_bias
    if isinstance(data.get("geographic_bias"), dict):
        lines.append("Geographic bias:")
        lines.append(f"  Issue: {data['geographic_bias'].get('issue', 'Unknown')}")
        lines.append(f"  Mitigation: {data['geographic_bias'].get('mitigation', 'Unknown')}")
    
    return "\n".join(lines)


def _format_site_legal_template(site_type: str, data: Dict) -> str:
    """Format site-specific legal template."""
    lines = [
        f"LEGAL TEMPLATE for {site_type.replace('_', ' ').upper()}",
    ]
    
    if "applicable_laws" in data:
        lines.append(f"Applicable laws: {', '.join(data['applicable_laws'])}")
    
    if "permits_likely_required" in data:
        lines.append("Permits likely required:")
        for p in data["permits_likely_required"]:
            lines.append(f"  - {p}")
    
    if "constraints" in data:
        lines.append("Key constraints:")
        for c in data["constraints"]:
            lines.append(f"  - {c}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 60)
    print("🔢 Gemini Embeddings Test")
    print("=" * 60)
    
    try:
        embeddings = GeminiEmbeddings()
        
        # Test single embedding
        test_text = "Ponderosa Pine grows at elevations of 1000-2500 meters in the Sierra Nevada."
        embedding = embeddings.embed_text(test_text)
        
        print(f"\n📝 Test text: {test_text[:50]}...")
        print(f"   Embedding dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        
    except ValueError as e:
        print(f"\n⚠️ {e}")
        print("Set GOOGLE_API_KEY to test embeddings.")
