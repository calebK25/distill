#!/usr/bin/env python3
"""
Interactive PDF Question-Answering System.
Allows users to ask questions via terminal and get comprehensive analysis.
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional

from context_compressor.multimodal_extractor import MultimodalExtractor
from context_compressor.compressor import ContextCompressor
from context_compressor.schemas import CompressionRequest as CDCRequest
from openrouter_integration import OpenRouterIntegration


def get_user_input(prompt: str, default: str = "", required: bool = True) -> str:
    """Get user input with optional default value."""
    while True:
        if default:
            user_input = input(f"{prompt} (default: {default}): ").strip()
            if not user_input:
                user_input = default
        else:
            user_input = input(f"{prompt}: ").strip()
        
        if user_input or not required:
            return user_input
        print("This field is required. Please enter a value.")


def get_numeric_input(prompt: str, default: float, min_val: float = 0, max_val: float = float('inf')) -> float:
    """Get numeric user input with validation."""
    while True:
        try:
            user_input = get_user_input(prompt, str(default))
            value = float(user_input)
            if min_val <= value <= max_val:
                return value
            else:
                print(f"Value must be between {min_val} and {max_val}")
        except ValueError:
            print("Please enter a valid number.")


def get_boolean_input(prompt: str, default: bool = True) -> bool:
    """Get boolean user input."""
    default_str = "Y" if default else "N"
    while True:
        user_input = get_user_input(prompt, default_str).lower()
        if user_input in ['y', 'yes', 'true', '1']:
            return True
        elif user_input in ['n', 'no', 'false', '0']:
            return False
        else:
            print("Please enter Y/yes or N/no")


def analyze_pdf_interactive(pdf_path: str, question: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze PDF with given parameters and return comprehensive results.
    
    Args:
        pdf_path: Path to the PDF file
        question: Question to ask
        params: Analysis parameters
    
    Returns:
        Dictionary with analysis results
    """
    
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    start_time = time.time()
    
    try:
        # Initialize components
        print("Initializing components...")
        extractor = MultimodalExtractor()
        cdc_compressor = ContextCompressor()
        
        # Extract content
        print("Extracting content from PDF...")
        extraction_start = time.time()
        multimodal_candidates = extractor.extract_from_pdf(pdf_path, Path(pdf_path).stem)
        text_candidates = [c for c in multimodal_candidates if c.modality == "text"]
        extraction_time = time.time() - extraction_start
        
        print(f"Extracted {len(multimodal_candidates)} total candidates ({len(text_candidates)} text)")
        
        # Prepare CDC candidates
        cdc_candidates = []
        for i, candidate in enumerate(text_candidates):
            cdc_candidates.append({
                "text": candidate.text,
                "tokens": len(candidate.text.split()),
                "bm25": 0.5,
                "dense_sim": 0.5,
                "embedding": [0.1] * 768,
                "section": candidate.section,
                "page": candidate.page,
                "id": str(i),
                "doc_id": Path(pdf_path).stem
            })
        
        # Compress content
        print("Compressing content...")
        compression_start = time.time()
        
        cdc_request = CDCRequest(
            q=question,
            B=params["token_budget"],
            candidates=cdc_candidates,
            params={
                "lambda_": params["lambda"],
                "doc_cap": params["doc_cap"],
                "top_m": params["top_m"],
                "auto_router": params["auto_router"]
            }
        )
        
        compression_response = cdc_compressor.compress(cdc_request)
        compressed_context = compression_response.context
        compression_time = time.time() - compression_start
        
        # Generate LLM response
        print("Generating LLM response...")
        llm_start = time.time()
        
        api_key = params.get("api_key") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("No OpenRouter API key provided")
        
        llm_client = OpenRouterIntegration(api_key)
        
        # Build the prompt with context and question
        prompt = f"Context:\n{compressed_context}\n\nQuestion: {question}\n\nAnswer:"
        
        llm_response = llm_client.generate_response(
            prompt=prompt,
            model=params["model_id"],
            max_tokens=params["max_tokens"],
            temperature=params["temperature"]
        )
        
        llm_time = time.time() - llm_start
        total_time = time.time() - start_time
        
        # Calculate statistics
        original_tokens = sum(c["tokens"] for c in cdc_candidates)
        compressed_tokens = compression_response.stats.used
        reduction = 1 - (compressed_tokens / original_tokens) if original_tokens > 0 else 0
        
        return {
            "success": True,
            "question": question,
            "answer": llm_response["answer"],
            "citations": llm_response.get("citations", []),
            "compression_stats": {
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "reduction": reduction,
                "mode": compression_response.stats.mode,
                "candidates_selected": len(compression_response.context.split()),
                "total_candidates": len(cdc_candidates)
            },
            "timing": {
                "total_time": total_time,
                "extraction_time": extraction_time,
                "compression_time": compression_time,
                "llm_time": llm_time
            },
            "parameters": params,
            "compressed_context": compressed_context
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "question": question,
            "parameters": params
        }


def display_comprehensive_analysis(results: Dict[str, Any]):
    """Display comprehensive analysis results."""
    
    if not results.get("success", False):
        print(f"\nERROR: {results.get('error', 'Unknown error')}")
        return
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS RESULTS")
    print("="*80)
    
    # Question and Answer
    print(f"\nQUESTION:")
    print(f"   {results['question']}")
    
    print(f"\nANSWER:")
    print(f"   {results['answer']}")
    
    # Citations
    citations = results.get("citations", [])
    if citations:
        print(f"\nCITATIONS ({len(citations)}):")
        for i, citation in enumerate(citations, 1):
            print(f"   [{i}] {citation}")
    
    # Compression Analysis
    comp_stats = results["compression_stats"]
    print(f"\nCOMPRESSION ANALYSIS:")
    print(f"   Method: CDC (Cross-Document Compression)")
    print(f"   Mode: {comp_stats['mode']}")
    print(f"   Original tokens: {comp_stats['original_tokens']:,}")
    print(f"   Compressed tokens: {comp_stats['compressed_tokens']:,}")
    print(f"   Token reduction: {comp_stats['reduction']:.1%}")
    print(f"   Candidates processed: {comp_stats['total_candidates']:,}")
    print(f"   Candidates selected: {comp_stats['candidates_selected']:,}")
    print(f"   Selection ratio: {comp_stats['candidates_selected']/comp_stats['total_candidates']:.1%}")
    
    # Performance Metrics
    timing = results["timing"]
    print(f"\nPERFORMANCE METRICS:")
    print(f"   Total processing time: {timing['total_time']:.2f}s")
    print(f"   PDF extraction time: {timing['extraction_time']:.2f}s ({timing['extraction_time']/timing['total_time']:.1%})")
    print(f"   Compression time: {timing['compression_time']:.2f}s ({timing['compression_time']/timing['total_time']:.1%})")
    print(f"   LLM generation time: {timing['llm_time']:.2f}s ({timing['llm_time']/timing['total_time']:.1%})")
    
    # Parameters Used
    params = results["parameters"]
    print(f"\nPARAMETERS USED:")
    print(f"   Token budget: {params['token_budget']:,}")
    print(f"   MMR lambda: {params['lambda']}")
    print(f"   Document cap: {params['doc_cap']}")
    print(f"   Top M candidates: {params['top_m']}")
    print(f"   Auto-router: {params['auto_router']}")
    print(f"   Model: {params['model_id']}")
    print(f"   Max tokens: {params['max_tokens']}")
    print(f"   Temperature: {params['temperature']}")
    
    # Context Preview
    context = results.get("compressed_context", "")
    if context:
        print(f"\nCOMPRESSED CONTEXT PREVIEW:")
        context_lines = context.split('\n')
        for i, line in enumerate(context_lines[:8], 1):
            if line.strip():
                print(f"   {i:2d}. {line[:80]}{'...' if len(line) > 80 else ''}")
        if len(context_lines) > 8:
            print(f"   ... and {len(context_lines) - 8} more lines")
    
    # Efficiency Score
    efficiency_score = comp_stats['reduction'] * (1 - timing['total_time']/60)  # Normalize to 1 minute
    print(f"\nEFFICIENCY SCORE:")
    print(f"   Overall efficiency: {efficiency_score:.3f} (higher is better)")
    print(f"   Token efficiency: {comp_stats['reduction']:.1%}")
    print(f"   Time efficiency: {1 - timing['total_time']/60:.1%} (normalized to 1 minute)")
    
    print("\n" + "="*80)


def get_analysis_parameters() -> Dict[str, Any]:
    """Get analysis parameters from user input."""
    
    print("\n" + "="*60)
    print("CONFIGURE ANALYSIS PARAMETERS")
    print("="*60)
    
    # Compression parameters
    print("\nCOMPRESSION PARAMETERS:")
    token_budget = int(get_numeric_input("Token budget for compression", 800, 100, 2000))
    lambda_val = get_numeric_input("MMR lambda (diversity vs relevance)", 0.7, 0.0, 1.0)
    doc_cap = int(get_numeric_input("Document cap (max chunks per doc)", 8, 1, 20))
    top_m = int(get_numeric_input("Top M candidates for selection", 200, 50, 500))
    auto_router = get_boolean_input("Use auto-router for CDC/SDM", True)
    
    # LLM parameters
    print("\nLLM PARAMETERS:")
    model_id = get_user_input("Model ID", "qwen/qwen3-8b:free")
    max_tokens = int(get_numeric_input("Max tokens to generate", 1024, 50, 2000))
    temperature = get_numeric_input("Generation temperature", 0.1, 0.0, 1.0)
    
    # API key
    api_key = get_user_input("OpenRouter API key (or press Enter to use environment variable)", "", required=False)
    
    return {
        "token_budget": token_budget,
        "lambda": lambda_val,
        "doc_cap": doc_cap,
        "top_m": top_m,
        "auto_router": auto_router,
        "model_id": model_id,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "api_key": api_key
    }


def main():
    """Main interactive function."""
    
    print("INTERACTIVE PDF QUESTION-ANSWERING SYSTEM")
    print("="*60)
    
    # Get PDF path
    pdf_path = get_user_input("Enter PDF file path")
    
    if not Path(pdf_path).exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return
    
    print(f"PDF found: {pdf_path}")
    
    while True:
        print("\n" + "-"*60)
        
        # Get question
        question = get_user_input("\nEnter your question (or 'quit' to exit)")
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question.strip():
            print("Please enter a valid question.")
            continue
        
        # Get parameters
        try:
            params = get_analysis_parameters()
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        
        # Run analysis
        print(f"\nAnalyzing: {question}")
        print("Please wait...")
        
        try:
            results = analyze_pdf_interactive(pdf_path, question, params)
            display_comprehensive_analysis(results)
        except Exception as e:
            print(f"\nError during analysis: {e}")
        
        # Ask if user wants to continue
        continue_analysis = get_boolean_input("\nAsk another question?", True)
        if not continue_analysis:
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()
