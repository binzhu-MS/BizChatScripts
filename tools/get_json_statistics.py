"""
JSON Statistics Analyzer

This module provides functionality to analyze statistics from JSON files containing 
categorized text data. It calculates various metrics, including category-wise breakdowns.

Author: Bin Zhu
Date: July 22, 2025
"""

import os
from typing import Dict, List, Any, Union
import fire  # For command line interface

# Import from the Utils package (sibling folder)
from utils.json_utils import read_json_file, write_json_file

def analyze_statistics(data: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Analyzes statistics for the JSON data structure.
    
    Parameters:
        data (Dict[str, List[str]]): The JSON data containing categories and text entries.
        
    Returns:
        Dict[str, Any]: A dictionary containing various statistics.
    """
    stats = {}
    
    # Basic counts
    stats['total_categories'] = len(data)
    
    # Text entry statistics
    all_texts = []
    categories = []
    
    for category, texts in data.items():
        count = len(texts)
        categories.append({
            "category_name": category,
            "count": count
        })
        all_texts.extend(texts)
    
    stats['categories'] = categories
    stats['total_text_entries'] = len(all_texts)
    
    return stats


def print_statistics(stats: Dict[str, Any]) -> None:
    """
    Prints the statistics in a readable format.
    
    Parameters:
        stats (Dict[str, Any]): The statistics dictionary.
        
    Returns:
        None
    """
    print("=" * 60)
    print("FILE STATISTICS REPORT")
    print("=" * 60)
    
    print("\nBASIC INFORMATION:")
    print(f"  Total Categories: {stats['total_categories']}")
    print(f"  Total Text Entries: {stats['total_text_entries']}")
    
    print("\nCATEGORIES:")
    for category in stats['categories']:
        print(f"  - {category['category_name']} - {category['count']} entries")
    
    print("=" * 60)


def save_statistics_to_file(stats: Dict[str, Any], output_file: str) -> None:
    """
    Saves the statistics to a JSON file.
    
    Parameters:
        stats (Dict[str, Any]): The statistics dictionary.
        output_file (str): The output file path.
        
    Returns:
        None
    """
    write_json_file(stats, output_file)
    print(f"\nStatistics saved to: {output_file}")


def main(
    input_path: Union[str, None] = None,
    output_path: Union[str, None] = None,
    silent: bool = False
) -> None:
    """
    Analyze JSON file statistics and generate a comprehensive report.
    
    Args:
        input_path: Path to the input JSON file (defaults to multiple_tool_call_utterance_for_all_segments.json in current directory)
        output_path: Path to the output statistics JSON file (defaults to input filename with _statistics.json extension)
        silent: If True, only saves to file without printing to console
    
    Example usage:
        # Use default file in current directory
        python get_JSON_statistics.py
        
        # Analyze a specific file
        python get_JSON_statistics.py --input_path=data.json
        
        # Specify both input and output
        python get_JSON_statistics.py --input_path=data.json --output_path=results.json
        
        # Silent mode (no console output)
        python get_JSON_statistics.py --input_path=data.json --silent=True
    """
    try:
        # Validate input_path was provided (no defaults allowed)
        if input_path is None:
            raise ValueError("Error: input_path is required. Please specify a valid input file path.")
        
        # Validate input file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Set default output file path if not provided
        if output_path is None or output_path.strip() == "":
            base, ext = os.path.splitext(input_path)
            output_path = base + "_statistics.json"
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Read and analyze the JSON file
        if not silent:
            print(f"Reading file: {input_path}")
        
        data = read_json_file(input_path)
        
        if not silent:
            print("Analyzing statistics...")
        
        # Check data type and convert if necessary
        if isinstance(data, list):
            # Convert list to a dictionary with a single key
            data = {"items": data}
            if not silent:
                print("Note: Input data was in list format, converted to dictionary with 'items' key.")
        
        stats = analyze_statistics(data)
        
        # Print statistics to console unless in silent mode
        if not silent:
            print_statistics(stats)
        
        # Save statistics to file
        save_statistics_to_file(stats, output_path)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    # Use fire for command line arguments
    fire.Fire(main)