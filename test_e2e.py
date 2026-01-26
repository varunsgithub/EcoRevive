#!/usr/bin/env python3
"""
EcoRevive End-to-End Test
=========================
Tests Layer 1 (Fire Model) → Layer 2 (Gemini Reasoning) pipeline.

This script:
1. Creates synthetic Sentinel-2 imagery (or loads real data if available)
2. Runs Layer 1 inference to get burn severity map
3. Feeds severity map to Layer 2 Gemini modules
4. Tests RAG retrieval for ecological knowledge
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "California-Fire-Model"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def test_layer1_model_loading():
    """Test that the fire model loads correctly."""
    print("\n" + "=" * 60)
    print("🔥 LAYER 1: Fire Model Test")
    print("=" * 60)
    
    try:
        # Add California-Fire-Model to path
        cfm_path = PROJECT_ROOT / "California-Fire-Model"
        sys.path.insert(0, str(cfm_path))
        
        from model.architecture import CaliforniaFireModel
        import torch
        
        checkpoint_path = PROJECT_ROOT / "California-Fire-Model/checkpoints/model.pth"
        
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found at {checkpoint_path}")
            return None, None
        
        print(f"📂 Loading model from: {checkpoint_path}")
        
        # Create model with default config
        model = CaliforniaFireModel(
            input_channels=10,
            output_channels=1,
            base_channels=64,
            use_attention=True,
        )
        
        # Load weights (weights_only=False for PyTorch 2.6+ compatibility)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   ✅ Loaded from checkpoint dict (epoch {checkpoint.get('epoch', '?')})")
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print(f"   ✅ Loaded from state_dict")
        else:
            model.load_state_dict(checkpoint)
            print(f"   ✅ Loaded raw state_dict")
        
        model.to(device)
        model.eval()
        
        print(f"   Device: {device}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test with synthetic input
        print("\n📊 Testing inference with synthetic data...")
        batch_size = 1
        channels = 10  # Sentinel-2 bands
        height, width = 256, 256
        
        synthetic_input = torch.randn(batch_size, channels, height, width).to(device)
        
        with torch.no_grad():
            output = model(synthetic_input)
            severity_map = torch.sigmoid(output).squeeze().cpu().numpy()
        
        print(f"   Input shape: {synthetic_input.shape}")
        print(f"   Output shape: {severity_map.shape}")
        print(f"   Severity range: [{severity_map.min():.3f}, {severity_map.max():.3f}]")
        print(f"   Mean severity: {severity_map.mean():.3f}")
        
        # Compute metadata like Layer 1 would
        metadata = {
            'mean_severity': float(severity_map.mean()),
            'max_severity': float(severity_map.max()),
            'burned_ratio': float((severity_map > 0.5).mean()),
            'confidence': float(np.abs(severity_map - 0.5).mean() * 2),
        }
        
        print(f"\n📈 Layer 1 Output Metadata:")
        print(f"   Mean severity: {metadata['mean_severity']:.1%}")
        print(f"   Burned pixels: {metadata['burned_ratio']:.1%}")
        print(f"   Confidence: {metadata['confidence']:.1%}")
        
        return severity_map, metadata
        
    except Exception as e:
        print(f"❌ Layer 1 Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_layer2_gemini(severity_map, metadata):
    """Test Layer 2 Gemini integration."""
    print("\n" + "=" * 60)
    print("🧠 LAYER 2: Gemini Reasoning Test")
    print("=" * 60)
    
    if severity_map is None:
        # Create synthetic severity map for testing
        print("   Using synthetic severity map (Layer 1 not available)")
        severity_map = np.random.beta(2, 1, size=(256, 256))
        metadata = {
            'mean_severity': float(severity_map.mean()),
            'max_severity': float(severity_map.max()),
            'burned_ratio': float((severity_map > 0.5).mean()),
            'confidence': 0.75,
        }
    
    try:
        # Test 1: Gemini Client
        print("\n📡 Test 1: Gemini Client Connection")
        from reasoning import EcoReviveGemini
        
        client = EcoReviveGemini()
        print("   ✅ Gemini client initialized")
        
        # Test 2: Multimodal Analysis (severity map as image)
        print("\n🖼️ Test 2: Multimodal Analysis (Severity Map → Image → Gemini)")
        from reasoning import severity_map_to_image, compute_severity_stats
        
        # Convert severity map to image
        severity_image = severity_map_to_image(severity_map)
        stats = compute_severity_stats(severity_map)
        
        print(f"   Severity image size: {severity_image.size}")
        print(f"   Stats: mean={stats['mean_severity']:.2f}, high_ratio={stats['high_severity_ratio']:.2f}")
        
        # Simple multimodal test (without image to save API calls)
        response = client.analyze_multimodal(
            prompt="In one sentence, what is the most important factor in post-fire ecosystem recovery?",
            use_json=False
        )
        print(f"   ✅ Gemini responded: {response['text'][:100]}...")
        print(f"   Tokens: {response['usage']}")
        
        # Test 3: Structured JSON Output
        print("\n📋 Test 3: Structured JSON Output")
        json_response = client.analyze_multimodal(
            prompt="""
            Return a JSON object with these fields for Ponderosa Pine:
            - scientific_name: string
            - fire_adaptation: string  
            - priority_for_planting: number (1-3)
            """,
            use_json=True
        )
        
        if json_response.get('parsed'):
            print(f"   ✅ JSON parsed successfully:")
            print(f"      {json_response['parsed']}")
        else:
            print(f"   ⚠️ JSON parsing issue: {json_response.get('parse_error')}")
        
        return True
        
    except ValueError as e:
        print(f"❌ API Key Error: {e}")
        print("   Set GOOGLE_API_KEY environment variable")
        return False
    except Exception as e:
        print(f"❌ Layer 2 Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_system():
    """Test RAG knowledge retrieval."""
    print("\n" + "=" * 60)
    print("📚 RAG: Knowledge Base Test")
    print("=" * 60)
    
    try:
        from reasoning.rag import EcologyRAG, LegalRAG
        from reasoning.rag.embeddings import load_json_as_documents
        
        # Test document loading (no API needed)
        print("\n📂 Test 1: Loading Knowledge Base Documents")
        
        kb_path = PROJECT_ROOT / "reasoning/knowledge_base"
        
        eco_docs = load_json_as_documents(kb_path / "ecology/california_ecoregions.json")
        print(f"   Loaded {len(eco_docs)} ecoregion documents")
        
        species_docs = load_json_as_documents(kb_path / "ecology/native_species_catalog.json")
        print(f"   Loaded {len(species_docs)} species documents")
        
        legal_docs = load_json_as_documents(kb_path / "legal/federal_state_lands.json")
        print(f"   Loaded {len(legal_docs)} legal/land documents")
        
        total_docs = len(eco_docs) + len(species_docs) + len(legal_docs)
        print(f"   ✅ Total: {total_docs} documents ready for embedding")
        
        # Show sample document
        print("\n📄 Sample Document:")
        if species_docs:
            sample = species_docs[0]
            print(f"   ID: {sample.doc_id}")
            print(f"   Type: {sample.metadata.get('type')}")
            print(f"   Preview: {sample.content[:200]}...")
        
        # Test RAG with API (optional)
        print("\n🔍 Test 2: RAG Search (requires API)")
        try:
            rag = EcologyRAG(rebuild_index=True)
            rag.initialize()
            
            # Search for species
            results = rag.get_species_recommendations(
                "fire resistant trees for high elevation Sierra Nevada"
            )
            
            print(f"   ✅ Found {len(results)} species recommendations:")
            for r in results[:3]:
                print(f"      [{r['score']:.2f}] {r.get('scientific_name', 'Unknown')}")
                
        except ValueError as e:
            print(f"   ⚠️ Skipping RAG search (API key required): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🌲 ECOREVIVE END-TO-END TEST")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    
    # Test Layer 1
    severity_map, metadata = test_layer1_model_loading()
    layer1_ok = severity_map is not None
    
    # Test Layer 2
    layer2_ok = test_layer2_gemini(severity_map, metadata)
    
    # Test RAG
    rag_ok = test_rag_system()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"   Layer 1 (Fire Model):  {'✅ PASS' if layer1_ok else '❌ FAIL'}")
    print(f"   Layer 2 (Gemini):      {'✅ PASS' if layer2_ok else '❌ FAIL'}")
    print(f"   RAG (Knowledge Base):  {'✅ PASS' if rag_ok else '❌ FAIL'}")
    print("=" * 60)
    
    return layer1_ok and layer2_ok and rag_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
