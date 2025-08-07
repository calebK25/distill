#!/usr/bin/env python3
"""
Quick script to run the interactive PDF QA system.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    from src.examples.interactive_pdf_qa import main
    main()
