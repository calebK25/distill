#!/usr/bin/env python3
"""
Integrated Question-Answering System.
Combines context compression, LLM generation, and citation handling.
"""

import os
import time
import json
import asyncio
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import tempfile
import shutil

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from context_compressor.enhanced_multimodal_compressor import EnhancedMultimodalCompressor
from context_compressor.multimodal_extractor import MultimodalExtractor
from context_compressor.multimodal_schemas import MultimodalCompressionRequest, MultimodalCompressionResponse
from context_compressor.compressor import ContextCompressor
from context_compressor.schemas import CompressionRequest as CDCRequest, CompressionResponse as CDCResponse
from context_compressor.logging_config import production_logger
from openrouter_integration import OpenRouterIntegration
from llm_integration import OptimizedLLMIntegration


class QARequest(BaseModel):
    """Integrated QA request model."""
    question: str = Field(..., description="User question")
    budget: int = Field(400, ge=100, le=2000, description="Token budget for compression")
    lambda_: float = Field(0.5, ge=0.0, le=1.0, description="MMR lambda parameter")
    enable_image_search: bool = Field(True, description="Enable image search")
    max_images: int = Field(3, ge=0, le=10, description="Maximum images to include")
    max_tables: int = Field(5, ge=0, le=20, description="Maximum tables to include")
    image_weight: float = Field(0.2, ge=0.0, le=1.0, description="Image weight in fusion")
    
    # LLM settings
    llm_provider: str = Field("openrouter", description="LLM provider: 'openrouter' or 'local'")
    model_id: str = Field("qwen/qwen3-8b:free", description="Model ID")
    max_tokens: int = Field(512, ge=50, le=1000, description="Maximum tokens to generate")
    temperature: float = Field(0.1, ge=0.0, le=1.0, description="Generation temperature")
    
    # CDC/SDM settings
    use_cdc: bool = Field(True, description="Use Cross-Document Compression")
    doc_cap: int = Field(5, ge=1, le=20, description="Document cap for CDC")
    top_m: int = Field(200, ge=50, le=500, description="Top M candidates for fusion")
    auto_router: bool = Field(True, description="Use auto-router for CDC/SDM")
    
    # API keys (optional for local models)
    openrouter_key: Optional[str] = None
    hf_token: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the main findings about model performance?",
                "budget": 500,
                "lambda_": 0.5,
                "enable_image_search": True,
                "max_images": 3,
                "max_tables": 5,
                "image_weight": 0.2,
                "llm_provider": "openrouter",
                "model_id": "qwen/qwen3-8b:free",
                "max_tokens": 512,
                "temperature": 0.1,
                "use_cdc": True,
                "doc_cap": 5,
                "top_m": 200,
                "auto_router": True
            }
        }


class QAResponse(BaseModel):
    """Integrated QA response model."""
    success: bool
    answer: str
    citations: List[Dict[str, Any]]
    context_summary: Dict[str, Any]
    processing_metrics: Dict[str, Any]
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    uptime: float
    components: Dict[str, str]
    performance_summary: Dict[str, Any]


class IntegratedQASystem:
    """Integrated question-answering system."""
    
    def __init__(self):
        """Initialize the integrated QA system."""
        self.extractor = None
        self.multimodal_compressor = None
        self.cdc_compressor = None
        self.openrouter_client = None
        self.local_llm = None
        self.startup_time = time.time()
        self.request_count = 0
        self.error_count = 0
        
        # Performance tracking
        self.avg_processing_time = 0.0
        self.total_processing_time = 0.0
        
        # Temporary file management
        self.temp_dir = Path(tempfile.gettempdir()) / "integrated_qa"
        self.temp_dir.mkdir(exist_ok=True)
    
    async def initialize_components(self):
        """Initialize all system components."""
        try:
            production_logger.info("Initializing Integrated QA System components")
            
            # Initialize extractors and compressors
            self.extractor = MultimodalExtractor()
            self.multimodal_compressor = EnhancedMultimodalCompressor()
            self.cdc_compressor = ContextCompressor()
            
            production_logger.info("All components initialized successfully")
            
        except Exception as e:
            production_logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _get_openrouter_client(self, api_key: str, model_id: str) -> OpenRouterIntegration:
        """Get OpenRouter client instance."""
        return OpenRouterIntegration(api_key)
    
    def _get_local_llm(self, model_id: str, hf_token: Optional[str] = None) -> OptimizedLLMIntegration:
        """Get local LLM instance."""
        return OptimizedLLMIntegration(model_id, hf_token)
    
    async def process_question(self, 
                             pdf_file: UploadFile,
                             request: QARequest) -> QAResponse:
        """Process a question with PDF document."""
        start_time = time.time()
        temp_file_path = None
        
        try:
            self.request_count += 1
            
            # Save uploaded file temporarily
            temp_file_path = self.temp_dir / f"upload_{int(time.time())}_{pdf_file.filename}"
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(pdf_file.file, buffer)
            
            # Step 1: Extract content from PDF
            extraction_start = time.time()
            multimodal_candidates = self.extractor.extract_from_pdf(
                str(temp_file_path), 
                pdf_file.filename.replace('.pdf', '')
            )
            extraction_time = time.time() - extraction_start
            
            # Step 2: Compress content
            compression_start = time.time()
            
            if request.use_cdc:
                # Use CDC compression
                text_candidates = [c for c in multimodal_candidates if c.modality == "text"]
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
                        "doc_id": pdf_file.filename.replace('.pdf', '')
                    })
                
                cdc_request = CDCRequest(
                    q=request.question,
                    B=request.budget,
                    candidates=cdc_candidates,
                    params=request.model_dump()
                )
                
                compression_response = self.cdc_compressor.compress(cdc_request)
                compressed_context = compression_response.context
                context_summary = {
                    "method": "cdc",
                    "original_tokens": sum(c["tokens"] for c in cdc_candidates),
                    "compressed_tokens": compression_response.stats.used,
                    "reduction": 1 - (compression_response.stats.used / sum(c["tokens"] for c in cdc_candidates)),
                    "candidates_selected": len(compression_response.context.split("\n")),
                    "mode": compression_response.stats.mode
                }
            else:
                # Use multimodal compression
                multimodal_request = MultimodalCompressionRequest(
                    q=request.question,
                    B=request.budget,
                    candidates=multimodal_candidates,
                    lambda_=request.lambda_,
                    enable_image_search=request.enable_image_search,
                    max_images=request.max_images,
                    max_tables=request.max_tables,
                    image_weight=request.image_weight
                )
                
                compression_response = self.multimodal_compressor.compress(multimodal_request)
                compressed_context = compression_response.context
                context_summary = {
                    "method": "multimodal",
                    "original_tokens": sum(len(c.text.split()) for c in multimodal_candidates if c.modality == "text"),
                    "compressed_tokens": compression_response.budget,
                    "candidates_selected": len(compression_response.mapping),
                    "modality_breakdown": {
                        "text": compression_response.text_chunks,
                        "image": compression_response.image_chunks,
                        "table": compression_response.table_chunks
                    }
                }
            
            compression_time = time.time() - compression_start
            
            # Step 3: Generate LLM response
            llm_start = time.time()
            
            if request.llm_provider == "openrouter":
                api_key = request.openrouter_key or os.getenv("OPENROUTER_API_KEY")
                if not api_key:
                    raise ValueError("OpenRouter API key not provided")
                
                llm_client = self._get_openrouter_client(api_key, request.model_id)
                
                # Create compression response for LLM
                mock_response = MultimodalCompressionResponse(
                    context=compressed_context,
                    mapping=[],
                    used_tokens=len(compressed_context.split()),
                    budget=len(compressed_context.split()),
                    saved_vs_pool=0,
                    total_ms=0,
                    low_context=False,
                    text_chunks=0,
                    image_chunks=0,
                    table_chunks=0,
                    lambda_=request.lambda_,
                    fusion_weights={},
                    section_cap=2
                )
                
                # Build the prompt with context and question
                prompt = f"Context:\n{compressed_context}\n\nQuestion: {request.question}\n\nAnswer:"
                
                llm_response = llm_client.generate_response(
                    prompt=prompt,
                    model=request.model_id,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                )
                
                answer = llm_response["answer"]
                citations = llm_response.get("citations", [])
                
            else:
                # Use local LLM
                hf_token = request.hf_token or os.getenv("HF_TOKEN")
                llm_client = self._get_local_llm(request.model_id, hf_token)
                
                # Build the prompt with context and question
                prompt = f"Context:\n{compressed_context}\n\nQuestion: {request.question}\n\nAnswer:"
                
                llm_response = llm_client.generate_response(
                    context=prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                )
                
                answer = llm_response["answer"]
                citations = llm_response.get("citations", [])
            
            llm_time = time.time() - llm_start
            total_time = time.time() - start_time
            
            # Update performance metrics
            self.total_processing_time += total_time
            self.avg_processing_time = self.total_processing_time / self.request_count
            
            # Prepare response
            processing_metrics = {
                "total_time": total_time,
                "extraction_time": extraction_time,
                "compression_time": compression_time,
                "llm_time": llm_time,
                "tokens_generated": len(answer.split()),
                "citations_count": len(citations)
            }
            
            return QAResponse(
                success=True,
                answer=answer,
                citations=citations,
                context_summary=context_summary,
                processing_metrics=processing_metrics
            )
            
        except Exception as e:
            self.error_count += 1
            production_logger.error(f"Error processing question: {e}")
            
            return QAResponse(
                success=False,
                answer="",
                citations=[],
                context_summary={},
                processing_metrics={},
                error=str(e)
            )
            
        finally:
            # Clean up temporary file
            if temp_file_path and temp_file_path.exists():
                temp_file_path.unlink()


# FastAPI application
app = FastAPI(
    title="Integrated QA System",
    description="End-to-end question-answering with context compression and citation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system instance
qa_system = IntegratedQASystem()


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    await qa_system.initialize_components()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - qa_system.startup_time
    
    components = {
        "extractor": "initialized" if qa_system.extractor else "not_initialized",
        "multimodal_compressor": "initialized" if qa_system.multimodal_compressor else "not_initialized",
        "cdc_compressor": "initialized" if qa_system.cdc_compressor else "not_initialized"
    }
    
    performance_summary = {
        "total_requests": qa_system.request_count,
        "error_rate": qa_system.error_count / max(qa_system.request_count, 1),
        "avg_processing_time": qa_system.avg_processing_time,
        "uptime_seconds": uptime
    }
    
    return HealthResponse(
        status="healthy" if qa_system.extractor else "initializing",
        version="1.0.0",
        uptime=uptime,
        components=components,
        performance_summary=performance_summary
    )


@app.post("/qa", response_model=QAResponse)
async def process_qa(
    background_tasks: BackgroundTasks,
    pdf_file: UploadFile = File(...),
    request: QARequest = Form(...)
):
    """Process a question with PDF document."""
    if not pdf_file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Add cleanup task
    background_tasks.add_task(lambda: None)  # Placeholder for cleanup
    
    return await qa_system.process_question(pdf_file, request)


@app.get("/metrics")
async def get_metrics():
    """Get system metrics."""
    return {
        "requests_total": qa_system.request_count,
        "errors_total": qa_system.error_count,
        "avg_processing_time": qa_system.avg_processing_time,
        "uptime_seconds": time.time() - qa_system.startup_time
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    production_logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    uvicorn.run(
        "integrated_qa_system:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    run_server()
