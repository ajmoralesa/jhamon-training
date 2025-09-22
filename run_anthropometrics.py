#!/usr/bin/env python3
"""
Simple script to run anthropometric data extraction.
Run this from the command line or import the function in your code.
"""

from pathlib import Path
import sys
import os

# Add the project root to the Python path so we can import our modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from extract_anthropometrics import main as extract_main
    
    def run_extraction():
        """Run the anthropometric data extraction."""
        return extract_main()
    
    if __name__ == "__main__":
        print("=== Anthropometric Data Extraction ===")
        print()
        
        try:
            df = run_extraction()
            if df is not None:
                print(f"\nExtracted data for {len(df)} participants.")
                print("\nFirst few rows:")
                print(df.head())
            
        except Exception as e:
            print(f"An error occurred: {e}")
            print("\nPlease check:")
            print("1. That the data path '/Volumes/AJMA/' exists and contains participant folders")
            print("2. That the participant folders contain 'antro' or 'anthro' Excel files")
            print("3. That you have the required Python packages installed (pandas, openpyxl)")

except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure all required packages are installed.")
    print("You may need to install openpyxl: pip install openpyxl")
