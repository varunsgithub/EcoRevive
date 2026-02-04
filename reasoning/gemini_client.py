"""
EcoRevive Gemini Client
=======================
Unified Gemini API client using the NEW google.genai SDK.

Features:
- Gemini 3 Flash Preview
- Structured JSON output
- Google Search grounding
- Multimodal (images)

MIGRATED from deprecated google.generativeai to google.genai
"""

import os
import json
from typing import Optional, Dict, Any, List
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# New unified SDK
from google import genai
from google.genai import types


class EcoReviveGemini:
    """
    Gemini-powered reasoning engine for EcoRevive.
    
    Uses the new google.genai unified SDK (replacing deprecated google.generativeai).
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini client.
        
        Args:
            api_key: Google API key. If not provided, reads from GOOGLE_API_KEY env var.
        """
        # Get API key
        self.api_key = api_key or os.environ.get('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Create unified client
        self.client = genai.Client(api_key=self.api_key)
        
        # Model names
        self.model_name = 'gemini-3-flash-preview' 
        
        print("✅ EcoRevive Gemini client initialized")
        print(f"   - Model: {self.model_name}")
        print(f"   - SDK: google.genai (unified)")
    
    def analyze_multimodal(
        self,
        prompt: str,
        images: List[Any] = None,
        use_json: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze content using Gemini's multimodal capabilities.
        
        Args:
            prompt: Text prompt for analysis
            images: List of PIL Images or image paths to include
            use_json: Whether to request structured JSON output
            
        Returns:
            Response dict with 'text' and optionally 'parsed' (if JSON)
        """
        # Build content parts
        contents = []
        
        # Add images first if provided
        if images:
            for img in images:
                if isinstance(img, (str, Path)):
                    # Load image from path
                    import PIL.Image
                    img = PIL.Image.open(img)
                contents.append(img)
        
        # Add text prompt
        contents.append(prompt)
        
        # Configure generation
        config = types.GenerateContentConfig(
            temperature=0.1 if use_json else 0.2,
            max_output_tokens=8192,
        )
        
        # Add JSON response format if requested
        if use_json:
            config.response_mime_type = 'application/json'
        
        # Generate response
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config
        )
        
        result = {
            'text': response.text,
            'usage': {
                'prompt_tokens': getattr(response.usage_metadata, 'prompt_token_count', 0) if response.usage_metadata else 0,
                'response_tokens': getattr(response.usage_metadata, 'candidates_token_count', 0) if response.usage_metadata else 0,
            }
        }
        
        # Parse JSON if requested
        if use_json:
            try:
                result['parsed'] = json.loads(response.text)
            except json.JSONDecodeError:
                result['parsed'] = None
                result['parse_error'] = "Failed to parse JSON response"
        
        return result
    
    def search_grounded(self, query: str) -> Dict[str, Any]:
        """
        Query with Google Search grounding for real-time information.
        
        Args:
            query: The query to search and answer
            
        Returns:
            Response with grounded answer and sources
        """
        # Configure with Google Search tool
        config = types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=8192,
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=query,
            config=config
        )
        
        result = {
            'text': response.text,
            'grounding_metadata': None,
            'sources': []
        }
        
        # Extract grounding metadata if available
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                gm = candidate.grounding_metadata
                result['grounding_metadata'] = {
                    'search_queries': getattr(gm, 'web_search_queries', []),
                }
                # Extract source URLs
                if hasattr(gm, 'grounding_chunks') and gm.grounding_chunks:
                    for chunk in gm.grounding_chunks:
                        if hasattr(chunk, 'web') and chunk.web:
                            result['sources'].append({
                                'uri': chunk.web.uri,
                                'title': chunk.web.title
                            })
        
        return result
    
    def generate_with_functions(
        self,
        prompt: str,
        tools: List[Dict],
        auto_execute: bool = False,
        function_map: Dict = None
    ) -> Dict[str, Any]:
        """
        Generate response with function calling capabilities.
        
        Args:
            prompt: The user prompt
            tools: List of function declarations
            auto_execute: If True, automatically execute called functions
            function_map: Dict mapping function names to callables
            
        Returns:
            Response with text and/or function calls
        """
        # Configure with function tools
        config = types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=8192,
            tools=tools
        )
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        
        result = {
            'text': None,
            'function_calls': [],
            'function_results': []
        }
        
        # Check for function calls
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        result['text'] = part.text
                    elif hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        call_info = {
                            'name': fc.name,
                            'args': dict(fc.args) if fc.args else {}
                        }
                        result['function_calls'].append(call_info)
                        
                        # Auto-execute if requested
                        if auto_execute and function_map and fc.name in function_map:
                            try:
                                fn_result = function_map[fc.name](**dict(fc.args))
                                result['function_results'].append({
                                    'name': fc.name,
                                    'result': fn_result
                                })
                            except Exception as e:
                                result['function_results'].append({
                                    'name': fc.name,
                                    'error': str(e)
                                })
        
        return result
    
    def chat_session(self, system_instruction: str = None):
        """
        Start a multi-turn chat session.
        
        Args:
            system_instruction: Optional system prompt for the session
            
        Returns:
            Chat object for multi-turn conversation
        """
        config = types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=8192,
            system_instruction=system_instruction
        )
        
        # Return a chat wrapper
        return self.client.chats.create(
            model=self.model_name,
            config=config
        )


# Convenience function for quick initialization
def create_client(api_key: Optional[str] = None) -> EcoReviveGemini:
    """Create and return an EcoRevive Gemini client."""
    return EcoReviveGemini(api_key=api_key)


if __name__ == "__main__":
    # Quick test
    print("Testing EcoRevive Gemini Client (google.genai SDK)...")
    
    try:
        client = create_client()
        
        # Test basic generation
        response = client.analyze_multimodal(
            "What are the key factors in post-fire ecosystem recovery?",
            use_json=False
        )
        print(f"\n📝 Response preview: {response['text'][:200]}...")
        print(f"   Tokens used: {response['usage']}")
        
    except ValueError as e:
        print(f"⚠️ {e}")
        print("   Set GOOGLE_API_KEY to test the client.")
