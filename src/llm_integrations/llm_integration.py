import os
import time
import torch
from typing import Dict, Any, Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from context_compressor.logging_config import production_logger

class OptimizedLLMIntegration:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.generator = None
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            production_logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            production_logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            production_logger.error(f"Failed to load model: {e}")
            raise Exception(f"Model loading failed: {e}")
    
    def generate_response(self, 
                         context: str, 
                         max_tokens: int = 1024,
                         temperature: float = 0.1,
                         **kwargs) -> Dict[str, Any]:
        """
        Generate response using local model with citation enforcement.
        """
        start_time = time.time()
        
        # Build citation-enforced prompt
        citation_prompt = self._build_citation_prompt(context)
        
        try:
            # Generate response
            outputs = self.generator(
                citation_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
            
            # Extract generated text
            generated_text = outputs[0]["generated_text"]
            answer = generated_text[len(citation_prompt):].strip()
            
            # Extract citations
            citations = self._extract_citations(answer)
            
            generation_time = time.time() - start_time
            
            return {
                "answer": answer,
                "citations": citations,
                "model": self.model_name,
                "generation_time": generation_time,
                "tokens_used": len(self.tokenizer.encode(generated_text)),
                "prompt_tokens": len(self.tokenizer.encode(citation_prompt)),
                "completion_tokens": len(self.tokenizer.encode(answer))
            }
            
        except Exception as e:
            production_logger.error(f"Generation error: {e}")
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
    
    def cleanup(self):
        """Clean up model resources."""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
