
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load env
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

try:
    from reasoning.rag.embeddings import GeminiEmbeddings
    
    print("Initializing GeminiEmbeddings...")
    embeddings = GeminiEmbeddings()
    
    test_text = "This is a test sentence for embedding verification."
    print(f"\nEmbedding text: '{test_text}'")
    
    vector = embeddings.embed_text(test_text)
    
    print(f"\nSuccess! Generated vector of length: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")
    
    if len(vector) == 768:
        print("\nVerification PASSED: Dimension is 768 as expected.")
    else:
        print(f"\nVerification FAILED: Expected 768 dimensions, got {len(vector)}")
        
except Exception as e:
    print(f"\nVerification FAILED with error: {e}")
    import traceback
    traceback.print_exc()
