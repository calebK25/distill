# Context Compressor

A production-grade context compression system for intelligent document analysis and question answering with comprehensive PDF processing capabilities.

## Overview

Context Compressor is an advanced system that intelligently compresses and analyzes PDF documents using state-of-the-art techniques including Cross-Document Compression (CDC), Single-Document Mode (SDM), and multimodal processing. It achieves 90-95% token reduction while maintaining answer quality.

## Key Features

- **Cross-Document Compression (CDC)**: Intelligent selection across multiple documents with document caps
- **Single-Document Mode (SDM)**: Optimized for single-document analysis with relaxed constraints
- **Auto-Router**: Automatically switches between CDC and SDM based on content characteristics
- **Multimodal Processing**: Extracts and processes text, images, and tables from PDFs
- **LLM Integration**: Supports OpenRouter and local models with citation handling
- **Performance Optimization**: Caching, batching, and memory management
- **Interactive Analysis**: Terminal-based interface for real-time document analysis

## Quick Start

### Installation

```bash
git clone <repository>
cd context-compressor
pip install -e .
```

### Interactive Usage (Recommended)

```bash
# Set your OpenRouter API key
$env:OPENROUTER_API_KEY="your_api_key_here"

# Run interactive analysis
python interactive_pdf_qa.py
```

The interactive system will guide you through:
1. Selecting a PDF file
2. Entering your question
3. Configuring analysis parameters
4. Viewing comprehensive results

### Programmatic Usage

```python
from context_compressor.compressor import ContextCompressor
from context_compressor.schemas import CompressionRequest

# Initialize compressor
compressor = ContextCompressor()

# Create request
request = CompressionRequest(
    q="What are the main findings?",
    B=800,  # Token budget
    candidates=[...],  # Your document chunks
    params={"lambda_": 0.7, "doc_cap": 8}
)

# Compress
response = compressor.compress(request)
print(response.context)
```

## System Components

### Core Architecture

#### 1. **MultimodalExtractor**
- Extracts text, images, and tables from PDF documents
- Supports vector graphics and raster images
- Handles complex document layouts
- Provides content diagnosis and error handling

#### 2. **ContextCompressor**
- Main compression orchestrator
- Implements CDC/SDM logic with auto-routing
- Manages token budgets and constraints
- Provides comprehensive statistics

#### 3. **EnhancedMultimodalCompressor**
- Handles multimodal content compression
- Balances text, image, and table content
- Optimizes for different content types
- Provides modality-specific analysis

#### 4. **Auto-Router**
- Analyzes content characteristics
- Calculates `top1_doc_frac` and `entropy`
- Automatically selects CDC or SDM mode
- Optimizes for query type and document structure

### Compression Algorithms

#### Cross-Document Compression (CDC)
- **Purpose**: Multi-document scenarios with document diversity
- **Features**: Document caps, section constraints, cross-document selection
- **Use Case**: Research papers, multi-source analysis, comparative studies

#### Single-Document Mode (SDM)
- **Purpose**: Single-document deep analysis
- **Features**: Relaxed document constraints, focused selection
- **Use Case**: Detailed paper analysis, book chapters, technical documents

#### MMR (Maximal Marginal Relevance)
- **Algorithm**: Greedy selection balancing relevance and diversity
- **Parameters**: `lambda_` controls diversity vs relevance trade-off
- **Benefits**: Reduces redundancy while maintaining coverage

## Configuration Parameters

### Compression Settings

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `token_budget` | 100-2000 | 800 | Maximum tokens for compressed output |
| `lambda_` | 0.0-1.0 | 0.7 | MMR diversity parameter (higher = more diverse) |
| `doc_cap` | 1-20 | 8 | Maximum chunks per document |
| `top_m` | 50-500 | 200 | Initial candidate selection pool |
| `auto_router` | bool | True | Enable automatic CDC/SDM selection |

### LLM Settings

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `model_id` | string | "qwen/qwen3-8b:free" | OpenRouter model identifier |
| `max_tokens` | 50-2000 | 1024 | Maximum tokens to generate |
| `temperature` | 0.0-1.0 | 0.1 | Generation creativity (lower = more factual) |

## Analysis Output

### Comprehensive Results

The system provides detailed analysis including:

#### 1. **Compression Analysis**
- Original vs compressed token counts
- Token reduction percentage
- Candidates processed vs selected
- Compression mode used (CDC/SDM)

#### 2. **Performance Metrics**
- Total processing time breakdown
- Extraction, compression, and LLM timing
- Percentage allocation across components

#### 3. **Context Preview**
- First 8 lines of compressed context
- Shows exactly what content was selected
- Enables quality verification

#### 4. **Efficiency Score**
- Overall efficiency metric
- Token efficiency percentage
- Time efficiency (normalized to 1 minute)

### Example Output

```
COMPREHENSIVE ANALYSIS RESULTS
================================================================================

QUESTION:
   What are the main findings about model performance?

ANSWER:
   The main findings highlight that Large Language Models (LLMs) exhibit significant 
   challenges in maintaining long-term coherence when managing straightforward, 
   long-running tasks...

COMPRESSION ANALYSIS:
   Method: CDC (Cross-Document Compression)
   Mode: single_doc
   Original tokens: 7,469
   Compressed tokens: 387
   Token reduction: 94.8%
   Candidates processed: 308
   Candidates selected: 321
   Selection ratio: 104.2%

PERFORMANCE METRICS:
   Total processing time: 15.19s
   PDF extraction time: 6.05s (39.8%)
   Compression time: 0.15s (1.0%)
   LLM generation time: 8.99s (59.2%)

EFFICIENCY SCORE:
   Overall efficiency: 0.237 (higher is better)
   Token efficiency: 94.8%
   Time efficiency: 74.7% (normalized to 1 minute)
```

## Advanced Features

### Fine-Tuning Framework

The system includes a comprehensive fine-tuning framework:

#### Components
- **AnchorExtractor**: Extracts key information (entities, numbers, keyphrases)
- **OracleCreator**: Creates optimal selections using set-cover algorithms
- **TrainingDataGenerator**: Generates synthetic training data
- **FineTuner**: Trains bi-encoder and cross-encoder models

#### Usage
```python
from context_compressor.fine_tuning import FineTuner

# Initialize fine-tuner
fine_tuner = FineTuner()

# Train bi-encoder
fine_tuner.train_bi_encoder(training_data, model_name="custom-bi-encoder")

# Train cross-encoder
fine_tuner.train_cross_encoder(training_data, model_name="custom-cross-encoder")
```

### Web API Integration

For production deployment, use the integrated QA system:

```bash
# Start the server
python integrated_qa_system.py

# Access API endpoints
curl -X POST "http://localhost:8000/qa" \
  -F "pdf_file=@document.pdf" \
  -F "request={\"question\":\"What are the main findings?\"}"
```

## Performance Optimization

### Caching Strategy
- **Model Caching**: Reuses loaded models across requests
- **Embedding Cache**: Stores computed embeddings
- **Result Cache**: Caches compression results for similar queries

### Memory Management
- **Automatic Cleanup**: Releases unused resources
- **Batch Processing**: Efficient candidate processing
- **Memory Monitoring**: Tracks and optimizes memory usage

### Typical Performance Metrics
- **Token Reduction**: 90-95% typical compression
- **Processing Time**: 10-20 seconds for full pipeline
- **Memory Usage**: Optimized for production deployment
- **Accuracy**: Maintains answer quality despite compression

## Development

### Project Structure

```
context_compressor/
├── compressor.py                    # Main compression logic
├── multimodal_extractor.py         # PDF extraction
├── enhanced_multimodal_compressor.py # Multimodal processing
├── schemas.py                      # Data models
├── router.py                       # CDC/SDM routing
├── mmr.py                          # MMR selection
├── fusion.py                       # Rank fusion
├── fine_tuning/                    # Model fine-tuning framework
│   ├── fine_tuner.py              # Training orchestration
│   ├── anchor_extractor.py        # Key information extraction
│   ├── oracle_creator.py          # Optimal selection creation
│   └── data_generator.py          # Training data generation
└── logging_config.py              # Production logging

integrated_qa_system.py            # Web API server
interactive_pdf_qa.py              # Interactive terminal interface
openrouter_integration.py          # OpenRouter LLM integration
llm_integration.py                 # Local LLM integration
```

### Testing

```bash
# Run interactive analysis
python interactive_pdf_qa.py

# Test specific components
python -m pytest tests/

# Performance testing
python -m pytest tests/test_performance.py
```

## Troubleshooting

### Common Issues

1. **PDF Extraction Errors**
   - Ensure PDF is not password-protected
   - Check for corrupted PDF files
   - Verify sufficient disk space for temporary files

2. **LLM API Errors**
   - Verify OpenRouter API key is valid
   - Check API rate limits and quotas
   - Ensure stable internet connection

3. **Memory Issues**
   - Reduce `top_m` parameter for large documents
   - Lower `token_budget` for memory-constrained environments
   - Use smaller models for local deployment

### Performance Tuning

1. **For Faster Processing**
   - Reduce `token_budget` to 400-600
   - Lower `top_m` to 100-150
   - Use smaller LLM models

2. **For Better Quality**
   - Increase `token_budget` to 1000-1200
   - Raise `top_m` to 300-400
   - Use larger, more capable models

3. **For Memory Optimization**
   - Enable model caching
   - Use batch processing
   - Implement memory monitoring

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review the configuration parameters
- Test with the interactive interface
- Open an issue on GitHub
