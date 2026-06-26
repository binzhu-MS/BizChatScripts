"""
JSON File Merger Tool

This tool merges multiple JSON files containing categorized text data.
It uses the utility functions from the utils.json_utils module.

Author: Bin Zhu
Date: July 23, 2025
"""

import os
import sys
from typing import List
import logging
import fire

# Import from the Utils package
from utils.json_utils import merge_json_files, merge_json_files_advanced

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def merge_files(
    input_files,  # Can be a list or a single string
    output_file: str,
    strategy: str = "combine",
    advanced: bool = False,
    remove_duplicates: bool = True,
    sort_categories: bool = True
) -> None:
    """
    Merge multiple JSON files containing categorized text data.
    
    Args:
        input_files: List of input JSON file paths to merge, or comma-separated string
        output_file: Path for the merged output file
        strategy: Merge strategy - 'combine', or 'prefix' (default: 'combine')
        advanced: Use advanced merge with additional processing (default: False)
        remove_duplicates: Remove duplicate entries within categories (advanced mode only)
        sort_categories: Sort categories alphabetically (advanced mode only)
    
    Example usage:
        # Using comma-separated string (recommended for Fire)
        python merge_json_files.py merge_files "file1.json,file2.json,file3.json" output.json
        
        # Using multiple --input_files (may have parsing issues with Fire)
        python merge_json_files.py --input_files=file1.json --input_files=file2.json --output_file=merged.json
        
        # Advanced merge with duplicate removal and sorting
        python merge_json_files.py merge_files "file1.json,file2.json" output.json --advanced=True
        
        # Merge with prefix strategy to avoid category conflicts
        python merge_json_files.py --input_files=file1.json --input_files=file2.json --output_file=merged.json --strategy=prefix
    """
    try:
        # Debug: Check the type and content of input_files
        logger.info(f"Type of input_files: {type(input_files)}")
        logger.info(f"Raw input_files: {repr(input_files)}")
        
        # Handle different input formats
        if isinstance(input_files, str):
            if ',' in input_files:
                # Comma-separated string
                logger.info("input_files is a comma-separated string, splitting...")
                input_files = [f.strip() for f in input_files.split(',')]
            else:
                # Single file
                logger.info("input_files is a single string, converting to list")
                input_files = [input_files]
        elif isinstance(input_files, list):
            # Already a list
            logger.info("input_files is already a list")
        else:
            # Try to convert to list
            logger.info(f"input_files is {type(input_files)}, converting to list")
            input_files = list(input_files)
        
        # Print out input files for debugging 
        logger.info(f"Total input files: {len(input_files)}")
        for k, file_path in enumerate(input_files):
            logger.info(f"{k}-th input file: {file_path}")

        # Validate input files exist
        for file_path in input_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Input file not found: {file_path}")
        
        print(f"Merging {len(input_files)} JSON files...")
        for i, file_path in enumerate(input_files, 1):
            print(f"  {i}. {file_path}")
        
        # Choose merge function based on advanced flag
        if advanced:
            print(f"Using advanced merge (remove_duplicates={remove_duplicates}, sort_categories={sort_categories})")
            merged_data = merge_json_files_advanced(
                input_files, 
                output_file, 
                remove_duplicates=remove_duplicates,
                sort_categories=sort_categories
            )
        else:
            print(f"Using basic merge with strategy: {strategy}")
            merged_data = merge_json_files(input_files, output_file, strategy)
        
        # Print summary
        total_categories = len(merged_data)
        total_entries = sum(len(texts) for texts in merged_data.values())
        
        print("\nMerge completed successfully!")
        print(f"  Total categories: {total_categories}")
        print(f"  Total text entries: {total_entries}")
        print(f"  Output saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def quick_merge(file1: str, file2: str, output: str = "merged_output.json") -> None:
    """
    Quick merge of exactly two files with default settings.
    
    Args:
        file1: First JSON file to merge
        file2: Second JSON file to merge  
        output: Output file name (default: merged_output.json)
    
    Example:
        python merge_json_files.py quick_merge file1.json file2.json
    """
    merge_files([file1, file2], output)


if __name__ == "__main__":
    fire.Fire({
        'merge_files': merge_files,
        'quick_merge': quick_merge
    })
