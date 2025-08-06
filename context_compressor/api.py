"""
FastAPI endpoint for Context Compressor.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
import uvicorn

from .schemas import CompressionRequest, CompressionResponse
from .compressor import ContextCompressor

# Initialize FastAPI app
app = FastAPI(
    title="Context Compressor API",
    description="Intelligent context compression for LLM prompts",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize compressor
compressor = ContextCompressor()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Context Compressor API",
        "version": "0.1.0",
        "endpoints": {
            "POST /compress": "Compress context",
            "GET /health": "Health check",
            "GET /stats": "Get compressor statistics"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "compressor_ready": True}


@app.get("/stats")
async def get_stats():
    """Get compressor statistics."""
    return compressor.get_stats()


@app.post("/compress", response_model=CompressionResponse)
async def compress_context(request: CompressionRequest):
    """
    Compress context according to the request.
    
    This endpoint implements the core context compression algorithm:
    1. Rank fusion using z-score normalization
    2. MMR diversity selection with budget constraints
    3. Sentence-level trimming with anchor awareness
    """
    try:
        # Validate request
        if not request.candidates:
            raise HTTPException(
                status_code=400, 
                detail="At least one candidate is required"
            )
        
        if request.B <= 0:
            raise HTTPException(
                status_code=400, 
                detail="Budget must be positive"
            )
        
        # Perform compression
        response = compressor.compress(request)
        
        return response
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log the error in production
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/compress/batch")
async def compress_batch(requests: list[CompressionRequest]):
    """
    Compress multiple contexts in batch.
    
    This is useful for processing multiple queries efficiently.
    """
    try:
        responses = []
        for request in requests:
            try:
                response = compressor.compress(request)
                responses.append(response)
            except Exception as e:
                # Continue with other requests even if one fails
                responses.append({
                    "error": str(e),
                    "request_id": getattr(request, 'id', 'unknown')
                })
        
        return {"responses": responses}
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Batch processing error: {str(e)}"
        )


@app.delete("/cache")
async def clear_cache():
    """Clear the compressor cache."""
    compressor.clear_cache()
    return {"message": "Cache cleared successfully"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
