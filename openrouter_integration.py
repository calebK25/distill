#!/usr/bin/env python3
"""
OpenRouter Integration for Context Compressor.
Provides fast LLM generation using OpenRouter's API.
"""

import os
import json
import time
import requests
from typing import Dict, Any, Optional, List
from pathlib import Path

from context_compressor.multimodal_schemas import MultimodalCompressionResponse


class OpenRouterIntegration:
    """OpenRouter integration for fast LLM inference."""
    
    def __init__(self, 
                 api_key: str,
                 model_id: str = "qwen/qwen3-8b:free",
                 base_url: str = "https://openrouter.ai/api/v1"):
        """
        Initialize OpenRouter integration.
        
        Args:
            api_key: OpenRouter API key
            model_id: Model ID to use
            base_url: OpenRouter API base URL
        """
        self.api_key = api_key
        self.model_id = model_id
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/context-compressor",
            "X-Title": "Context Compressor"
        }
    
    def generate_response(self, 
                         compression_response: MultimodalCompressionResponse,
                         question: str,
                         max_tokens: int = 512,
                         temperature: float = 0.1) -> Dict[str, Any]:
        """
        Generate LLM response using OpenRouter API.
        
        Args:
            compression_response: Compression response with context
            question: User question
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            
        Returns:
            Response dictionary with answer and metadata
        """
        start_time = time.time()
        
        try:
            # Build prompt with citation enforcement
            prompt = self._build_citation_prompt(compression_response, question)
            
            # Prepare request payload
            payload = {
                "model": self.model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max(max_tokens, 300),
                "temperature": temperature,
                "stream": False,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
            
            # Make API request
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                return {
                    "answer": f"API Error: {response.status_code} - {response.text}",
                    "citations": [],
                    "processing_time": time.time() - start_time,
                    "tokens_generated": 0,
                    "error": f"HTTP {response.status_code}"
                }
            
            # Parse response
            response_data = response.json()
            
            if "choices" not in response_data or not response_data["choices"]:
                return {
                    "answer": "No response generated",
                    "citations": [],
                    "processing_time": time.time() - start_time,
                    "tokens_generated": 0,
                    "error": "Empty response"
                }
            
            # Extract answer
            answer = response_data["choices"][0]["message"]["content"]
            
            # Extract usage information
            usage = response_data.get("usage", {})
            tokens_generated = usage.get("completion_tokens", 0)
            
            # Extract citations
            citations = self._extract_citations(answer, compression_response)
            
            processing_time = time.time() - start_time
            
            return {
                "answer": answer,
                "citations": citations,
                "processing_time": processing_time,
                "tokens_generated": tokens_generated,
                "model": self.model_id,
                "usage": usage
            }
            
        except requests.exceptions.Timeout:
            return {
                "answer": "Request timed out",
                "citations": [],
                "processing_time": time.time() - start_time,
                "tokens_generated": 0,
                "error": "Timeout"
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "answer": f"Request failed: {str(e)}",
                "citations": [],
                "processing_time": time.time() - start_time,
                "tokens_generated": 0,
                "error": str(e)
            }
            
        except Exception as e:
            return {
                "answer": f"Unexpected error: {str(e)}",
                "citations": [],
                "processing_time": time.time() - start_time,
                "tokens_generated": 0,
                "error": str(e)
            }
    
    def _build_citation_prompt(self, 
                              compression_response: MultimodalCompressionResponse, 
                              question: str) -> str:
        """
        Build prompt with citation enforcement.
        
        Args:
            compression_response: Compression response with context
            question: User question
            
        Returns:
            Formatted prompt string
        """
        context = compression_response.context
        
        prompt = f"""You are a helpful AI assistant. Answer the following question based on the provided context. 

IMPORTANT: You must cite specific parts of the context when providing information. Use [citation_id] format for citations.

Context:
{context}

Question: {question}

Instructions:
1. Answer the question based on the provided context
2. Use [citation_id] to cite specific parts of the context
3. If the context doesn't contain enough information, say so
4. Be concise but comprehensive
5. Focus on factual information from the context

Answer:"""

        return prompt
    
    def _extract_citations(self, 
                          response_text: str, 
                          compression_response: MultimodalCompressionResponse) -> List[Dict[str, Any]]:
        """
        Extract citations from response text.
        
        Args:
            response_text: LLM response text
            compression_response: Compression response for context
            
        Returns:
            List of citation dictionaries
        """
        citations = []
        
        # Simple citation extraction - look for [citation_id] patterns
        import re
        citation_pattern = r'\[citation_(\d+)\]'
        matches = re.findall(citation_pattern, response_text)
        
        for match in matches:
            try:
                citation_id = int(match)
                citations.append({
                    "id": citation_id,
                    "type": "context_reference",
                    "source": "compressed_context"
                })
            except ValueError:
                continue
        
        return citations


def integrate_with_openrouter(compression_response: MultimodalCompressionResponse,
                             question: str,
                             api_key: str,
                             model_id: str = "qwen/qwen3-8b:free") -> Dict[str, Any]:
    """
    Convenience function for OpenRouter integration.
    
    Args:
        compression_response: Compression response with context
        question: User question
        api_key: OpenRouter API key
        model_id: Model ID to use
        
    Returns:
        Response dictionary with answer and metadata
    """
    client = OpenRouterIntegration(api_key, model_id)
    return client.generate_response(compression_response, question)


if __name__ == "__main__":
    # Example usage
    import os
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Please set OPENROUTER_API_KEY environment variable")
        exit(1)
    
    # Create mock compression response
    mock_response = MultimodalCompressionResponse(
        context="This is a test context about AI and machine learning.",
        mapping=[],
        used_tokens=10,
        budget=10,
        saved_vs_pool=0,
        total_ms=0,
        low_context=False,
        text_chunks=1,
        image_chunks=0,
        table_chunks=0,
        lambda_=0.5,
        fusion_weights={},
        section_cap=2
    )
    
    # Test integration
    result = integrate_with_openrouter(
        compression_response=mock_response,
        question="What is AI?",
        api_key=api_key
    )
    
    print("Response:", result["answer"])
    print("Citations:", result["citations"])
    print("Processing time:", result["processing_time"])
