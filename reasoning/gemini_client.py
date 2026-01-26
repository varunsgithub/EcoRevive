"""
EcoRevive Gemini Client
=======================
Unified Gemini API client showcasing multiple features:
- Gemini 1.5 Pro (multimodal reasoning)
- Structured JSON output
- Function calling
- Grounding with Google Search
- Imagen 3 for image generation

Built for the Gemini API Developer Competition.
"""

import os
import json
from typing import Optional, Dict, Any, List
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold


class EcoReviveGemini:
    """
    Gemini-powered reasoning engine for EcoRevive.
    
    This client demonstrates extensive use of Gemini API features
    for ecosystem restoration intelligence.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini client.
        
        Args:
            api_key: Google API key. If not provided, reads from GOOGLE_API_KEY env var.
        """
        # Configure API key
        self.api_key = api_key or os.environ.get('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        genai.configure(api_key=self.api_key)
        
        # Safety settings - allow ecological/environmental content
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        
        # Main reasoning model (Gemini 2.0 Flash - best free tier option)
        self.model = genai.GenerativeModel(
            model_name='models/gemini-2.0-flash',
            safety_settings=self.safety_settings,
            generation_config=genai.GenerationConfig(
                temperature=0.2,  # Low temp for factual responses
                top_p=0.95,
                max_output_tokens=8192,
            )
        )
        
        # Model configured for JSON output
        self.json_model = genai.GenerativeModel(
            model_name='models/gemini-2.0-flash',
            safety_settings=self.safety_settings,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                response_mime_type='application/json',
                max_output_tokens=8192,
            )
        )
        
        # Model with grounding (Google Search)
        self.grounded_model = genai.GenerativeModel(
            model_name='models/gemini-2.0-flash',
            safety_settings=self.safety_settings,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=8192,
            ),
            tools='google_search_retrieval'
        )
        
        print("✅ EcoRevive Gemini client initialized")
        print(f"   - Main model: gemini-2.0-flash")
        print(f"   - JSON output: enabled")
        print(f"   - Google Search grounding: enabled")
    
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
        content = [prompt]
        
        if images:
            for img in images:
                if isinstance(img, (str, Path)):
                    # Load image from path
                    import PIL.Image
                    img = PIL.Image.open(img)
                content.append(img)
        
        # Choose model based on output format
        model = self.json_model if use_json else self.model
        
        # Generate response
        response = model.generate_content(content)
        
        result = {
            'text': response.text,
            'usage': {
                'prompt_tokens': response.usage_metadata.prompt_token_count,
                'response_tokens': response.usage_metadata.candidates_token_count,
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
        
        Uses Gemini's grounding feature to search the web and provide
        cited, up-to-date information.
        
        Args:
            query: The query to search and answer
            
        Returns:
            Response with grounded answer and sources
        """
        response = self.grounded_model.generate_content(query)
        
        result = {
            'text': response.text,
            'grounding_metadata': None,
            'sources': []
        }
        
        # Extract grounding metadata if available
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'grounding_metadata'):
                gm = candidate.grounding_metadata
                result['grounding_metadata'] = {
                    'search_queries': getattr(gm, 'search_queries', []),
                    'grounding_supports': getattr(gm, 'grounding_supports', []),
                }
                # Extract source URLs
                if hasattr(gm, 'grounding_chunks'):
                    for chunk in gm.grounding_chunks:
                        if hasattr(chunk, 'web'):
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
        
        Allows Gemini to request function calls which can be executed
        and fed back for continued reasoning.
        
        Args:
            prompt: The user prompt
            tools: List of function declarations
            auto_execute: If True, automatically execute called functions
            function_map: Dict mapping function names to callables
            
        Returns:
            Response with text and/or function calls
        """
        # Create model with tools
        model_with_tools = genai.GenerativeModel(
            model_name='gemini-1.5-pro',
            safety_settings=self.safety_settings,
            tools=tools
        )
        
        response = model_with_tools.generate_content(prompt)
        
        result = {
            'text': None,
            'function_calls': [],
            'function_results': []
        }
        
        # Check for function calls
        if response.candidates:
            candidate = response.candidates[0]
            
            # Extract text if present
            if candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, 'text'):
                        result['text'] = part.text
                    elif hasattr(part, 'function_call'):
                        fc = part.function_call
                        call_info = {
                            'name': fc.name,
                            'args': dict(fc.args)
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
    
    def generate_image(
        self,
        prompt: str,
        aspect_ratio: str = "16:9",
        num_images: int = 1
    ) -> List[Any]:
        """
        Generate images using Imagen 3.
        
        Args:
            prompt: Description of the image to generate
            aspect_ratio: Image aspect ratio (e.g., "16:9", "1:1", "9:16")
            num_images: Number of images to generate (1-4)
            
        Returns:
            List of generated images
        """
        try:
            from google.generativeai import ImageGenerationModel
            
            imagen = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")
            
            response = imagen.generate_images(
                prompt=prompt,
                number_of_images=num_images,
                aspect_ratio=aspect_ratio,
                safety_filter_level="block_few",
                person_generation="allow_adult",
            )
            
            return [img for img in response.images]
            
        except Exception as e:
            print(f"⚠️ Imagen generation failed: {e}")
            print("   Imagen 3 may require additional API access.")
            return []
    
    def chat_session(self, system_instruction: str = None) -> 'ChatSession':
        """
        Start a multi-turn chat session.
        
        Args:
            system_instruction: Optional system prompt for the session
            
        Returns:
            ChatSession object for multi-turn conversation
        """
        model = genai.GenerativeModel(
            model_name='gemini-1.5-pro',
            safety_settings=self.safety_settings,
            system_instruction=system_instruction
        )
        return model.start_chat()


# Convenience function for quick initialization
def create_client(api_key: Optional[str] = None) -> EcoReviveGemini:
    """Create and return an EcoRevive Gemini client."""
    return EcoReviveGemini(api_key=api_key)


if __name__ == "__main__":
    # Quick test
    print("Testing EcoRevive Gemini Client...")
    
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
