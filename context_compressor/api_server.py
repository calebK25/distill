#!/usr/bin/env python3
"""
Production FastAPI server for Context Compressor.
Provides REST API endpoints with proper error handling and monitoring.
"""

import os
import time
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import tempfile
import shutil

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from .enhanced_multimodal_compressor import EnhancedMultimodalCompressor
from .multimodal_extractor import MultimodalExtractor
from .multimodal_schemas import MultimodalCompressionRequest
from .compressor import ContextCompressor
from .schemas import CompressionRequest as CDCRequest, CompressionResponse as CDCResponse
from .logging_config import production_logger
from openrouter_integration import integrate_with_openrouter


class CompressionRequest(BaseModel):
    """API request model for compression."""
    question: str = Field(..., description="User question")
    budget: int = Field(400, ge=100, le=2000, description="Token budget")
    lambda_: float = Field(0.5, ge=0.0, le=1.0, description="MMR lambda parameter")
    enable_image_search: bool = Field(True, description="Enable image search")
    max_images: int = Field(3, ge=0, le=10, description="Maximum images to include")
    max_tables: int = Field(5, ge=0, le=20, description="Maximum tables to include")
    image_weight: float = Field(0.2, ge=0.0, le=1.0, description="Image weight in fusion")
    
    class Config:
        schema_extra = {
            "example": {
                "question": "What are the main findings about model performance?",
                "budget": 500,
                "lambda_": 0.5,
                "enable_image_search": True,
                "max_images": 3,
                "max_tables": 5,
                "image_weight": 0.2
            }
        }


class CompressionResponse(BaseModel):
    """API response model for compression."""
    success: bool
    context: str
    used_tokens: int
    total_tokens: int
    processing_time: float
    candidates_selected: int
    modality_breakdown: Dict[str, int]
    performance_metrics: Dict[str, Any]
    error: Optional[str] = None


class LLMRequest(BaseModel):
    """API request model for LLM integration."""
    compression_response: CompressionResponse
    openrouter_key: Optional[str] = None
    model_id: str = Field("qwen/qwen3-8b:free", description="OpenRouter model ID")
    max_tokens: int = Field(512, ge=50, le=1000, description="Maximum tokens to generate")
    temperature: float = Field(0.1, ge=0.0, le=1.0, description="Generation temperature")


class LLMResponse(BaseModel):
    """API response model for LLM integration."""
    success: bool
    answer: str
    citations: List[Dict[str, Any]]
    processing_time: float
    tokens_generated: int
    model: str
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    uptime: float
    performance_summary: Dict[str, Any]


# Initialize FastAPI app
app = FastAPI(
    title="Context Compressor API",
    description="Production API for multimodal context compression",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
startup_time = time.time()
multimodal_compressor = None
cdc_compressor = None
extractor = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global multimodal_compressor, cdc_compressor, extractor
    
    try:
        # Initialize multimodal compressor
        multimodal_compressor = EnhancedMultimodalCompressor()
        production_logger.info("Multimodal compressor initialized successfully")
        
        # Initialize CDC compressor
        cdc_compressor = ContextCompressor()
        production_logger.info("CDC compressor initialized successfully")
        
        # Initialize extractor
        extractor = MultimodalExtractor()
        production_logger.info("Multimodal extractor initialized successfully")
        
    except Exception as e:
        production_logger.error(f"Failed to initialize components: {e}")
        raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - startup_time
    
    performance_summary = {
        "cache_size": cdc_compressor.get_stats()["cache_size"] if cdc_compressor else 0,
        "components_ready": all([multimodal_compressor, cdc_compressor, extractor])
    }
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime=uptime,
        performance_summary=performance_summary
    )

@app.post("/compress", response_model=CompressionResponse)
async def compress_context(
    background_tasks: BackgroundTasks,
    pdf_file: UploadFile = File(...),
    request: CompressionRequest = Form(...)
):
    """Compress context from PDF with multimodal support."""
    start_time = time.time()
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            shutil.copyfileobj(pdf_file.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            # Extract content
            extraction_result = extractor.extract_from_pdf(tmp_path)
            
            # Create multimodal request
            multimodal_request = MultimodalCompressionRequest(
                question=request.question,
                budget=request.budget,
                lambda_=request.lambda_,
                enable_image_search=request.enable_image_search,
                max_images=request.max_images,
                max_tables=request.max_tables,
                image_weight=request.image_weight,
                extracted_content=extraction_result
            )
            
            # Compress
            result = multimodal_compressor.compress(multimodal_request)
            
            processing_time = time.time() - start_time
            
            return CompressionResponse(
                success=True,
                context=result.context,
                used_tokens=result.stats.used_tokens,
                total_tokens=result.stats.total_tokens,
                processing_time=processing_time,
                candidates_selected=len(result.mapping),
                modality_breakdown=result.stats.modality_breakdown,
                performance_metrics={
                    "fusion_time": result.stats.fusion_ms,
                    "mmr_time": result.stats.mmr_ms,
                    "trim_time": result.stats.trim_ms
                }
            )
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
            
    except Exception as e:
        production_logger.error(f"Compression failed: {e}")
        return CompressionResponse(
            success=False,
            context="",
            used_tokens=0,
            total_tokens=0,
            processing_time=time.time() - start_time,
            candidates_selected=0,
            modality_breakdown={},
            performance_metrics={},
            error=str(e)
        )

@app.post("/compress/cdc", response_model=CDCResponse)
async def compress_cdc(request: CDCRequest):
    """Compress context using CDC/SDM features."""
    try:
        result = cdc_compressor.compress(request)
        return result
    except Exception as e:
        production_logger.error(f"CDC compression failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/llm", response_model=LLMResponse)
async def generate_llm_response(request: LLMRequest):
    """Generate LLM response using OpenRouter."""
    start_time = time.time()
    
    try:
        # Extract context from compression response
        context = request.compression_response.context
        
        # Get OpenRouter key
        openrouter_key = request.openrouter_key or os.getenv("OPENROUTER_API_KEY")
        if not openrouter_key:
            raise ValueError("OpenRouter API key not provided")
        
        # Generate response
        response = integrate_with_openrouter(
            context=context,
            question=request.compression_response.question,
            api_key=openrouter_key,
            model_id=request.model_id,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        processing_time = time.time() - start_time
        
        return LLMResponse(
            success=True,
            answer=response["answer"],
            citations=response.get("citations", []),
            processing_time=processing_time,
            tokens_generated=response.get("tokens_generated", 0),
            model=request.model_id
        )
        
    except Exception as e:
        production_logger.error(f"LLM generation failed: {e}")
        return LLMResponse(
            success=False,
            answer="",
            citations=[],
            processing_time=time.time() - start_time,
            tokens_generated=0,
            model=request.model_id,
            error=str(e)
        )

@app.get("/metrics")
async def get_metrics():
    """Get performance metrics."""
    return {
        "uptime": time.time() - startup_time,
        "cache_stats": cdc_compressor.get_stats() if cdc_compressor else {}
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    production_logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    uvicorn.run(
        "context_compressor.api_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    run_server()
