"""
Minimal CLI for Context Compressor.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any

from .schemas import CompressionRequest, Candidate
from .compressor import ContextCompressor


def load_candidates_from_file(file_path: str) -> list[Candidate]:
    """Load candidates from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return [Candidate(**item) for item in data]
        elif isinstance(data, dict) and 'candidates' in data:
            return [Candidate(**item) for item in data['candidates']]
        else:
            raise ValueError("File must contain a list of candidates or a dict with 'candidates' key")
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{file_path}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading candidates: {e}", file=sys.stderr)
        sys.exit(1)


def create_sample_candidates() -> list[Candidate]:
    """Create sample candidates for testing."""
    return [
        Candidate(
            id="c_001",
            doc_id="d_01",
            section="Results",
            page=14,
            text="The study found a significant improvement in performance with a 25% increase in accuracy. The results were consistent across all test conditions.",
            tokens=25,
            bm25=7.2,
            dense_sim=0.81,
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
        ),
        Candidate(
            id="c_002",
            doc_id="d_01",
            section="Methods",
            page=8,
            text="We employed a randomized controlled trial design with 200 participants. The intervention group received the new treatment protocol.",
            tokens=22,
            bm25=6.8,
            dense_sim=0.75,
            embedding=[0.2, 0.3, 0.4, 0.5, 0.6]
        ),
        Candidate(
            id="c_003",
            doc_id="d_02",
            section="Discussion",
            page=20,
            text="These findings suggest that the proposed approach may have broader applications. Future research should explore scalability.",
            tokens=18,
            bm25=5.9,
            dense_sim=0.68,
            embedding=[0.3, 0.4, 0.5, 0.6, 0.7]
        )
    ]


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Context Compressor CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compress with sample data
  python -m context_compressor.cli --query "What were the main findings?" --budget 50

  # Compress with custom candidates file
  python -m context_compressor.cli --query "Analyze the results" --budget 100 --candidates candidates.json

  # Use custom parameters
  python -m context_compressor.cli --query "Summarize the study" --budget 75 --lambda 0.8 --section-cap 3
        """
    )
    
    parser.add_argument(
        "--query", "-q",
        required=True,
        help="Query string"
    )
    
    parser.add_argument(
        "--budget", "-b",
        type=int,
        required=True,
        help="Token budget"
    )
    
    parser.add_argument(
        "--candidates", "-c",
        help="Path to JSON file containing candidates (optional, uses sample data if not provided)"
    )
    
    parser.add_argument(
        "--lambda",
        type=float,
        default=0.7,
        help="MMR diversity parameter (default: 0.7)"
    )
    
    parser.add_argument(
        "--section-cap",
        type=int,
        default=2,
        help="Maximum chunks per section (default: 2)"
    )
    
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=0.7,
        help="Weight for dense similarity in fusion (default: 0.7)"
    )
    
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=0.3,
        help="Weight for BM25 in fusion (default: 0.3)"
    )
    
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Use cross-encoder reranker"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass caching"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: stdout)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.budget <= 0:
        print("Error: Budget must be positive", file=sys.stderr)
        sys.exit(1)
    
    lambda_val = getattr(args, 'lambda')
    if not 0.0 <= lambda_val <= 1.0:
        print("Error: Lambda must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)
    
    if args.section_cap < 1:
        print("Error: Section cap must be at least 1", file=sys.stderr)
        sys.exit(1)
    
    if not 0.0 <= args.dense_weight <= 1.0 or not 0.0 <= args.bm25_weight <= 1.0:
        print("Error: Fusion weights must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)
    
    # Load candidates
    if args.candidates:
        candidates = load_candidates_from_file(args.candidates)
    else:
        candidates = create_sample_candidates()
        if args.verbose:
            print("Using sample candidates", file=sys.stderr)
    
    # Create request
    request = CompressionRequest(
        q=args.query,
        B=args.budget,
        candidates=candidates,
        params={
            "fusion_weights": {"dense": args.dense_weight, "bm25": args.bm25_weight},
            "lambda_": lambda_val,
            "section_cap": args.section_cap,
            "use_reranker": args.use_reranker
        },
        no_cache=args.no_cache
    )
    
    # Perform compression
    try:
        compressor = ContextCompressor()
        response = compressor.compress(request)
        
        # Prepare output
        output_data = {
            "context": response.context,
            "mapping": [item.dict() for item in response.mapping],
            "stats": response.stats.dict()
        }
        
        # Output result
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Results written to {args.output}", file=sys.stderr)
        else:
            print(json.dumps(output_data, indent=2))
        
        # Verbose output
        if args.verbose:
            stats = response.stats
            print(f"\nCompression Statistics:", file=sys.stderr)
            print(f"  Budget: {stats.budget} tokens", file=sys.stderr)
            print(f"  Used: {stats.used} tokens", file=sys.stderr)
            print(f"  Saved: {stats.saved_vs_pool} tokens", file=sys.stderr)
            print(f"  Compression ratio: {stats.saved_vs_pool / (stats.used + stats.saved_vs_pool) * 100:.1f}%", file=sys.stderr)
            print(f"  Processing time: {stats.total_ms:.2f} ms", file=sys.stderr)
            print(f"  Low context flag: {stats.low_context}", file=sys.stderr)
    
    except Exception as e:
        print(f"Error during compression: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
