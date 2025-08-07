import os
import time
import json
import requests
from typing import Dict, Any, Optional, List
from context_compressor.logging_config import production_logger

class OpenRouterIntegration:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or pass api_key parameter.")
        
        self.base_url = "https://openrouter.ai/api/v1"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
    
    def generate_response(self, 
                         prompt: str, 
                         model: str = "qwen/qwen3-8b:free",
                         max_tokens: int = 1024,
                         temperature: float = 0.1,
                         **kwargs) -> Dict[str, Any]:
        """
        Generate response using OpenRouter API with citation enforcement.
        """
        start_time = time.time()
        
        # Build citation-enforced prompt
        citation_prompt = self._build_citation_prompt(prompt)
        
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "user", "content": citation_prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    **kwargs
                },
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Extract citations from response
            citations = self._extract_citations(content)
            
            generation_time = time.time() - start_time
            
            return {
                "answer": content,
                "citations": citations,
                "model": model,
                "generation_time": generation_time,
                "tokens_used": result.get("usage", {}).get("total_tokens", 0),
                "prompt_tokens": result.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": result.get("usage", {}).get("completion_tokens", 0)
            }
            
        except requests.exceptions.RequestException as e:
            production_logger.error(f"OpenRouter API error: {e}")
            raise Exception(f"Failed to generate response: {e}")
    
    def _build_citation_prompt(self, context: str) -> str:
        """
        Build a prompt that enforces citation usage.
        """
        return f"""You are an expert document analyst. Answer the question based ONLY on the provided context. You MUST cite specific parts of the context using [citation_id] format where citation_id corresponds to the chunk number.

Context:
{context}

Instructions:
1. Answer the question using ONLY information from the provided context
2. Cite specific parts using [citation_id] format (e.g., [1], [2], [3])
3. If the context doesn't contain enough information, say so clearly
4. Be accurate and factual
5. Provide a comprehensive but concise answer

Question: {context.split('Question:')[-1].strip() if 'Question:' in context else 'Please analyze the provided context.'}

Answer:"""
    
    def _extract_citations(self, text: str) -> List[str]:
        """
        Extract citations from the generated text.
        """
        import re
        citations = re.findall(r'\[(\d+)\]', text)
        return list(set(citations))  # Remove duplicates
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models from OpenRouter.
        """
        try:
            response = self.session.get(f"{self.base_url}/models")
            response.raise_for_status()
            return response.json().get("data", [])
        except requests.exceptions.RequestException as e:
            production_logger.error(f"Failed to fetch models: {e}")
            return []
