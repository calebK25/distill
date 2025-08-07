#!/usr/bin/env python3
"""
Quick script to run the API server.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    from src.api.integrated_qa_system import run_server
    run_server()
