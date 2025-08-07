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

#### Interactive Workflow

The system guides you through a step-by-step process:

**Step 1: PDF Selection**
- Enter the path to your PDF file (default: "2502.15840v1.pdf")
- System validates the file exists and is accessible

**Step 2: Question Input**
- Type your question about the document
- Examples: "What are the main findings?", "How does the methodology work?"
- Type 'quit' to exit the system

**Step 3: Parameter Configuration**
- **Compression Parameters**: Control how content is selected and compressed
- **LLM Parameters**: Control answer generation and model selection
- **API Configuration**: Set up authentication for language models

**Step 4: Analysis & Results**
- System processes your request and displays comprehensive results
- Includes answer, citations, compression stats, and performance metrics
- Option to ask follow-up questions or start over

#### Example Interactive Session

```
INTERACTIVE PDF QUESTION-ANSWERING SYSTEM
============================================================
Enter PDF file path (default: 2502.15840v1.pdf): topology_paper.pdf
PDF found: topology_paper.pdf

------------------------------------------------------------

Enter your question (or 'quit' to exit): What are the key topological concepts discussed?

============================================================
CONFIGURE ANALYSIS PARAMETERS
============================================================

COMPRESSION PARAMETERS:
Token budget for compression (default: 800): 1000
MMR lambda (diversity vs relevance) (default: 0.7): 0.8
Document cap (max chunks per doc) (default: 8): 10
Top M candidates for selection (default: 200): 250
Use auto-router for CDC/SDM (default: Y): Y

LLM PARAMETERS:
Model ID (default: qwen/qwen3-8b:free): anthropic/claude-3.5-sonnet
Max tokens to generate (default: 1024): 1200
Generation temperature (default: 0.1): 0.1

OpenRouter API key (or press Enter to use environment variable): 

Analyzing: What are the key topological concepts discussed?
Please wait...
```

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

#### 2. **Document Chunks (The Building Blocks)**
- **What they are**: Segmented pieces of text extracted from PDF documents
- **How they're created**: The extractor breaks down PDF content into meaningful units:
  - Paragraphs or sections of text
  - Sentences grouped together
  - Logical content blocks (headers, body text, captions)
  - Page-based segments with location information
- **What each chunk contains**:
  ```python
  {
      "text": "The study found a 25% improvement in accuracy...",
      "tokens": 25,
      "section": "Results",
      "page": 14,
      "id": "chunk_001",
      "doc_id": "paper_2024"
  }
  ```
- **Why they matter**: 
  - Enable selective compression (only relevant chunks are kept)
  - Allow for 90-95% token reduction while maintaining quality
  - Provide traceable citations back to specific document sections
  - Enable parallel processing and memory optimization

#### 3. **ContextCompressor**
- Main compression orchestrator
- Implements CDC/SDM logic with auto-routing
- Manages token budgets and constraints
- Provides comprehensive statistics

#### 4. **EnhancedMultimodalCompressor**
- Handles multimodal content compression
- Balances text, image, and table content
- Optimizes for different content types
- Provides modality-specific analysis

#### 5. **Auto-Router**
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

### Compression Pipeline Explained

Here's how the system processes your document and question:

**Stage 1: Document Extraction**
```
PDF Document → MultimodalExtractor → 300+ Document Chunks
```

**Stage 2: Initial Ranking**
```
All Chunks → BM25 + Dense Similarity Scoring → Top M Candidates (200 by default)
```

**Stage 3: MMR Selection**
```
Top M Candidates → MMR Algorithm (λ=0.7) → Selected Chunks (respecting Document Cap)
```

**Stage 4: Context Assembly**
```
Selected Chunks → Token Budget Check (800 tokens) → Final Compressed Context
```

**Stage 5: LLM Generation**
```
Compressed Context + Question → Language Model → Answer with Citations
```

**Key Parameters in Action:**
- **Top M (200)**: Limits initial candidates for performance
- **MMR Lambda (0.7)**: Balances relevance vs diversity in selection
- **Document Cap (8)**: Ensures representation across sections
- **Token Budget (800)**: Final limit on context size
- **Auto-Router**: Chooses CDC vs SDM based on document structure

## Configuration Parameters

### Interactive Script Options

When using `python interactive_pdf_qa.py`, you'll be prompted to configure these parameters:

#### Compression Parameters

**Token Budget (100-2000, default: 800)**
- **What it does**: Sets the maximum number of tokens allowed in the compressed context sent to the LLM
- **Technical details**: This is the hard limit on how much content the system can include in the final compressed context. The system will stop adding chunks once this limit is reached, even if more relevant content exists.
- **Lower values (400-600)**: Faster processing, less detailed answers, lower cost, may miss important context
- **Higher values (1000-1200)**: More comprehensive answers, slower processing, higher cost, includes more context
- **Recommendation**: Start with 800, increase for complex questions, decrease for quick summaries

**MMR Lambda (0.0-1.0, default: 0.7)**
- **What it does**: Controls the balance between relevance and diversity in the Maximal Marginal Relevance algorithm
- **Technical details**: MMR uses this formula: `score = λ × relevance + (1-λ) × diversity`. Lambda controls the trade-off:
  - λ = 0.0: Only diversity matters (maximally diverse selection)
  - λ = 1.0: Only relevance matters (most relevant content only)
  - λ = 0.7: Balanced approach (70% relevance, 30% diversity)
- **Lower values (0.3-0.5)**: Focus on most relevant content, may miss important context, less redundancy
- **Higher values (0.8-1.0)**: Prioritize diverse content, covers more topics but may include less relevant info, more redundancy
- **Recommendation**: 0.7 provides good balance, use 0.5 for focused questions, 0.9 for broad analysis

**Document Cap (1-20, default: 8)**
- **What it does**: Maximum number of text chunks that can be selected from each document section (e.g., Introduction, Methods, Results)
- **Technical details**: This prevents the system from selecting too many chunks from a single section, ensuring representation across the entire document. For example, if set to 8, the system can select at most 8 chunks from the "Introduction" section, 8 from "Methods", etc.
- **Lower values (3-5)**: More focused selection, faster processing, may miss important details in long sections
- **Higher values (12-15)**: More comprehensive coverage, includes more context, may include redundant information
- **Recommendation**: 8 works well for most documents, increase for complex papers with long sections, decrease for simple documents

**Top M Candidates (50-500, default: 200)**
- **What it does**: Number of initial candidates considered in the first stage of selection before applying MMR
- **Technical details**: The system first ranks all document chunks by relevance (using BM25 + dense similarity scores), then takes the top M candidates for MMR selection. This is a performance optimization - instead of running MMR on all 1000+ chunks, it runs on the top 200 most relevant ones.
- **Lower values (100-150)**: Faster processing, may miss some relevant content that wasn't in the top candidates
- **Higher values (300-400)**: More thorough analysis, slower processing, considers more candidates
- **Recommendation**: 200 is optimal for most cases, increase for large documents with many sections, decrease for memory constraints

**Auto-Router (Y/N, default: Y)**
- **What it does**: Automatically chooses between Cross-Document Compression (CDC) and Single-Document Mode (SDM) based on content analysis
- **Technical details**: The router analyzes the document structure and calculates:
  - `top1_doc_frac`: Fraction of top candidates from the same document (1.0 = single document)
  - `entropy`: Diversity measure of candidate distribution across documents
  - If `top1_doc_frac > 0.8` and `entropy < 0.3`, it switches to SDM mode
- **Yes**: System analyzes content and picks the best mode automatically, optimizes for your specific document
- **No**: Forces CDC mode (useful for multi-document scenarios or when you want consistent behavior)
- **Recommendation**: Keep enabled unless you have specific requirements or want to force CDC mode

#### LLM Parameters

**Model ID (default: "qwen/qwen3-8b:free")**
- **What it does**: Specifies which language model to use for generating answers
- **Free models**: "qwen/qwen3-8b:free", "meta-llama/llama-3.1-8b:free"
- **Paid models**: "openai/gpt-4o", "anthropic/claude-3.5-sonnet", "google/gemini-pro"
- **Recommendation**: Start with free models, upgrade to paid for better quality

**Max Tokens (50-2000, default: 1024)**
- **What it does**: Maximum length of the generated answer
- **Lower values (300-500)**: Concise answers, faster generation
- **Higher values (1500-2000)**: Detailed explanations, longer generation time
- **Recommendation**: 1024 provides good detail, adjust based on question complexity

**Temperature (0.0-1.0, default: 0.1)**
- **What it does**: Controls creativity vs factual accuracy in responses
- **Lower values (0.0-0.2)**: More factual, consistent answers
- **Higher values (0.7-1.0)**: More creative, varied responses
- **Recommendation**: 0.1 for academic/research questions, 0.3-0.5 for creative analysis

#### API Configuration

**OpenRouter API Key**
- **What it does**: Authentication for accessing language models
- **Required**: Yes (unless using local models)
- **How to get**: Sign up at openrouter.ai and generate an API key
- **Security**: Can use environment variable `OPENROUTER_API_KEY` instead of entering directly

### Parameter Optimization Guide

#### For Speed Optimization
```
Token Budget: 400-600
Top M Candidates: 100-150
Max Tokens: 512
Model: qwen/qwen3-8b:free
```

#### For Quality Optimization
```
Token Budget: 1000-1200
Top M Candidates: 300-400
Max Tokens: 1500
Model: openai/gpt-4o
Temperature: 0.1
```

#### For Memory-Constrained Systems
```
Token Budget: 600
Top M Candidates: 100
Document Cap: 5
Max Tokens: 800
```

#### For Academic Research
```
Token Budget: 1000
MMR Lambda: 0.8
Document Cap: 10
Temperature: 0.1
Model: anthropic/claude-3.5-sonnet
```

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


### Testing

```bash
# Run interactive analysis
python interactive_pdf_qa.py

# Test specific components
python -m pytest tests/

# Performance testing
python -m pytest tests/test_performance.py
```

## License

MIT License - see LICENSE file for details.

