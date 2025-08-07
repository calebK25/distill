#!/usr/bin/env python3
"""
Main entry point for the Context Compressor system.
Provides easy access to different components and examples.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Main entry point with interactive menu."""
    print("Context Compressor System")
    print("=" * 40)
    print("1. Run Interactive PDF QA")
    print("2. Start API Server")
    print("3. Run Tests")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nSelect an option (1-4): ").strip()
            
            if choice == "1":
                print("\nStarting Interactive PDF QA...")
                from src.examples.interactive_pdf_qa import main as run_interactive
                run_interactive()
                break
                
            elif choice == "2":
                print("\nStarting API Server...")
                from src.api.integrated_qa_system import run_server
                run_server()
                break
                
            elif choice == "3":
                print("\nRunning tests...")
                import subprocess
                subprocess.run([sys.executable, "-m", "pytest", "tests/"])
                break
                
            elif choice == "4":
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice. Please select 1-4.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    main()
