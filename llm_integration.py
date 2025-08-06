#!/usr/bin/env python3
"""
LLM Integration for Context Compressor.
Provides optimized LLM generation with citation enforcement.
"""

import os
import json
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from context_compressor.multimodal_schemas import MultimodalCompressionResponse


class OptimizedLLMIntegration:
    """Optimized LLM integration with performance improvements."""
    
    def __init__(self, 
                 model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
                 hf_token: Optional[str] = None,
                 device: str = "auto",
                 max_memory: Optional[Dict[str, str]] = None):
        """
        Initialize optimized LLM integration.
        
        Args:
            model_id: HuggingFace model ID
            hf_token: HuggingFace token for private models
            device: Device to use
            max_memory: Memory allocation for model offloading
        """
        self.model_id = model_id
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        
        # Optimize memory allocation
        if max_memory is None and self.device == "cuda":
            gpu_memory = "12GB" if torch.cuda.get_device_properties(0).total_memory > 12e9 else "8GB"
            self.max_memory = {
                "cuda:0": gpu_memory,
                "cpu": "16GB"
            }
        else:
            self.max_memory = max_memory
        
        self.tokenizer = None
        self.model = None
        self._load_model(hf_token)
    
    def _load_model(self, hf_token: Optional[str] = None):
        """Load model with optimized settings."""
        try:
            print(f"Loading LLM: {self.model_id}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                token=hf_token,
                trust_remote_code=True
            )
            
            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with simplified device handling
            try:
                if self.device == "cuda" and torch.cuda.is_available():
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        token=hf_token,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    ).cuda()
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        token=hf_token,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    )
            except Exception as e:
                print(f"GPU loading failed: {e}")
                print("Falling back to CPU...")
                # Fallback to CPU
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    token=hf_token,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
    
    def generate_response(self, 
                         context: str,
                         question: str,
                         max_tokens: int = 512,
                         temperature: float = 0.1) -> Dict[str, Any]:
        """
        Generate LLM response with citation enforcement.
        
        Args:
            context: Compressed context
            question: User question
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            
        Returns:
            Response dictionary with answer and metadata
        """
        start_time = time.time()
        
        try:
            # Build prompt
            prompt = self._build_citation_prompt(context, question)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            # Extract citations
            citations = self._extract_citations(response_text)
            
            processing_time = time.time() - start_time
            
            return {
                "answer": response_text,
                "citations": citations,
                "processing_time": processing_time,
                "tokens_generated": len(response_text.split()),
                "model": self.model_id
            }
            
        except Exception as e:
            return {
                "answer": f"Error generating response: {str(e)}",
                "citations": [],
                "processing_time": time.time() - start_time,
                "tokens_generated": 0,
                "error": str(e)
            }
    
    def _build_citation_prompt(self, context: str, question: str) -> str:
        """
        Build prompt with citation enforcement.
        
        Args:
            context: Compressed context
            question: User question
            
        Returns:
            Formatted prompt string
        """
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
    
    def _extract_citations(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Extract citations from response text.
        
        Args:
            response_text: LLM response text
            
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


def integrate_with_llm(context: str,
                      question: str,
                      hf_token: str,
                      model_id: str = "meta-llama/Llama-3.1-8B-Instruct") -> Dict[str, Any]:
    """
    Convenience function for LLM integration.
    
    Args:
        context: Compressed context
        question: User question
        hf_token: HuggingFace token
        model_id: Model ID to use
        
    Returns:
        Response dictionary with answer and metadata
    """
    client = OptimizedLLMIntegration(model_id, hf_token)
    return client.generate_response(context, question)


if __name__ == "__main__":
    # Example usage
    import os
    
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Please set HF_TOKEN environment variable")
        exit(1)
    
    # Test integration
    context = "This is a test context about AI and machine learning."
    question = "What is AI?"
    
    result = integrate_with_llm(
        context=context,
        question=question,
        hf_token=hf_token
    )
    
    print("Response:", result["answer"])
    print("Citations:", result["citations"])
    print("Processing time:", result["processing_time"])
