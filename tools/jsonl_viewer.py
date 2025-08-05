#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
JSON/JSONL Viewer - A tool to read JSON or JSONL files and output selected entries in a nicely formatted JSON.

This script allows you to:
1. Read JSON or JSONL files (auto-detects format)
2. Select specific entries by index or filter by content
3. Output the selected entries to a file in a human-readable JSON format
4. Optionally extract only specific fields from each entry

Example usage:
    # View entry at index 5 from input.jsonl
    python jsonl_viewer.py --input_path=input.jsonl --index=5 --output_path=view.json
    
    # View the first 3 entries from a JSON file
    python jsonl_viewer.py --input_path=data.json --range=0,3 --output_path=view.json
    
    # View entries containing a specific text in any field
    python jsonl_viewer.py --input_path=input.jsonl --filter="error" --output_path=view.json
    
    # Extract only specific fields
    python jsonl_viewer.py --input_path=data.json --fields=id,name,score --output_path=view.json
"""

import os
import sys
import json
import fire
from typing import List, Dict, Any, Optional, Union, Tuple
from loguru import logger

def read_json(path: str) -> dict:
    """Read a JSON file and return the parsed data."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_jsonl(file_path: str) -> list:
    """
    Read a JSONL file and return a list of JSON objects.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of JSON objects
    """
    data = []
    with open(file_path, 'r', encoding='utf8') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    json_obj = json.loads(line)
                    data.append(json_obj)
                except json.JSONDecodeError as e:
                    logger.warning(f"Warning: Failed to parse JSON on line {line_num}: {e}")
                    continue
    return data


def read_input_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Read input file, automatically detecting if it's JSON or JSONL format.
    
    Args:
        file_path: Path to the input file
        
    Returns:
        List of JSON objects
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.jsonl':
        logger.info(f"Reading JSONL file: {file_path}")
        return read_jsonl(file_path)
    elif file_extension == '.json':
        logger.info(f"Reading JSON file: {file_path}")
        data = read_json(file_path)
        # If it's a single object, wrap it in a list
        if isinstance(data, dict):
            return [data]
        elif isinstance(data, list):
            return data
        else:
            logger.warning(f"Unexpected JSON structure in {file_path}")
            return []
    else:
        # Try to auto-detect based on content
        try:
            logger.info(f"Auto-detecting format for: {file_path}")
            # First try to read as JSONL
            return read_jsonl(file_path)
        except:
            try:
                # If that fails, try as JSON
                data = read_json(file_path)
                if isinstance(data, dict):
                    return [data]
                elif isinstance(data, list):
                    return data
                else:
                    return []
            except:
                logger.error(f"Could not parse file as JSON or JSONL: {file_path}")
                return []


def filter_jsonl_entries(
    json_data: List[Dict[str, Any]], 
    index: int = None,
    entry_range: Tuple[int, int] = None,
    filter_text: str = None,
    filter_field: str = None,
    fields: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Filter and process JSONL entries based on various criteria.
    
    Args:
        json_data: List of JSON objects from the input JSONL file
        index: Specific entry index to extract (0-based)
        entry_range: Tuple of (start, end) indices to extract a range of entries
        filter_text: Text to search for in any field
        filter_field: Specific field to apply the filter_text to (if None, searches all fields)
        fields: List of specific fields to extract from each entry (if None, includes all fields)
        
    Returns:
        List of filtered and processed JSON objects
    """
    result = []
    
    # Apply filtering logic
    if index is not None:
        if 0 <= index < len(json_data):
            result = [json_data[index]]
        else:
            logger.warning(f"Index {index} is out of range (0-{len(json_data)-1})")
    elif entry_range:
        start, end = entry_range
        if start < 0:
            start = 0
        if end > len(json_data):
            end = len(json_data)
        result = json_data[start:end]
    elif filter_text:
        if filter_field:
            # Filter by specific field
            result = [item for item in json_data if filter_field in item and 
                      isinstance(item[filter_field], str) and 
                      filter_text in item[filter_field]]
        else:
            # Filter by any field (convert everything to string for searching)
            result = []
            for item in json_data:
                item_str = json.dumps(item, ensure_ascii=False).lower()
                if filter_text.lower() in item_str:
                    result.append(item)
    else:
        # No filtering, use all data
        result = json_data
    
    # Extract specific fields if requested
    if fields and result:
        filtered_result = []
        for item in result:
            filtered_item = {field: item.get(field, None) for field in fields}
            filtered_result.append(filtered_item)
        return filtered_result
    
    return result


def main(
    input_path: str, 
    output_path: str = None, 
    index: int = None,
    range: str = None,
    filter: str = None,
    filter_field: str = None,
    fields: str = None,
    indent: int = 2
) -> None:
    """
    Read a JSON/JSONL file, select specific entries, and output them in nicely formatted JSON.
    
    Args:
        input_path: Path to the input JSON/JSONL file
        output_path: Path to the output JSON file (defaults to input filename with _view.json extension)
        index: Specific entry index to extract (0-based)
        range: Range of entries to extract (format: "start,end")
        filter: Text to search for in any field
        filter_field: Specific field to apply the filter to
        fields: Comma-separated list of fields to extract from each entry
        indent: Indentation level for the output JSON (default: 2)
    """
    try:
        # Validate input file
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Set default output path if not provided
        if output_path is None:
            base_name = os.path.splitext(input_path)[0]
            output_path = f"{base_name}_view.json"
        
        # Parse range parameter if provided
        entry_range = None
        if range:
            try:
                start, end = map(int, range.split(','))
                entry_range = (start, end)
            except ValueError:
                logger.error("Invalid range format. Use 'start,end' (e.g., '0,5')")
                sys.exit(1)
        
        # Parse fields parameter if provided
        fields_list = None
        if fields:
            fields_list = [field.strip() for field in fields.split(',')]
        
        # Read the input file (JSON or JSONL)
        json_data = read_input_file(input_path)
        
        if not json_data:
            logger.warning("No valid JSON objects found in the input file.")
            return
        
        # Filter and process the entries
        logger.info(f"Processing {len(json_data)} entries from: {input_path}")
        filtered_data = filter_jsonl_entries(
            json_data,
            index=index,
            entry_range=entry_range,
            filter_text=filter,
            filter_field=filter_field,
            fields=fields_list
        )
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Write the filtered data to the output file
        with open(output_path, 'w', encoding='utf8') as f:
            if len(filtered_data) == 1 and index is not None:
                # If specifically requesting one entry by index, don't wrap in an array
                json.dump(filtered_data[0], f, indent=indent, ensure_ascii=False)
            else:
                json.dump(filtered_data, f, indent=indent, ensure_ascii=False)
        
        logger.info(f"Successfully wrote {len(filtered_data)} entries to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Use fire for command line arguments
    fire.Fire(main)
