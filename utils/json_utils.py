"""
JSON Utilities Module

This module provides general-purpose utilities for JSON file and data handling.
- Functions for reading and writing JSON files.


Author: Bin Zhu
Last Update Date: July 21, 2025
"""
import os
import json
from typing import Dict, List, Any, Union, Optional


def read_json_file(file_path: str) -> Union[Dict, List, Any]:
    """
    Reads a JSON file and returns the corresponding Python data structure.
    
    Parameters:
        file_path (str): The path to the JSON file.
        
    Returns:
        Union[Dict, List, Any]: The JSON data as a Python object.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def write_json_file(data: Any, output_file: str, indent: int = 4) -> None:
    """
    Writes data to a JSON file with optional formatting.
    
    Parameters:
        data (Any): The data to write to the JSON file.
        output_file (str): The output file path.
        indent (int): Number of spaces for indentation (default: 4).
        
    Returns:
        None
        
    Raises:
        IOError: If the file cannot be written.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def read_json_string(json_string: str) -> Any:
    """
    Parses a JSON string and returns the corresponding Python data structure.
    
    Parameters:
        json_string (str): The JSON string to parse.
        
    Returns:
        Any: The parsed JSON data as a Python object.
        
    Raises:
        json.JSONDecodeError: If the string contains invalid JSON.
    """
    return json.loads(json_string)


def write_json_string(data: Any, indent: Optional[int] = None) -> str:
    """
    Converts Python data to a JSON string.
    
    Parameters:
        data (Any): The data to convert to JSON string.
        indent (Optional[int]): Number of spaces for indentation (default: None for compact).
        
    Returns:
        str: The JSON string representation of the data.
    """
    return json.dumps(data, indent=indent, ensure_ascii=False)


def merge_json_files(file_paths: List[str], output_file: Optional[str] = None, merge_strategy: str = "combine") -> Dict[str, List[str]]:
    """
    Merges multiple JSON files with categorized text data.
    
    Parameters:
        file_paths (List[str]): List of paths to JSON files to merge.
        output_file (str, optional): Path to save the merged result. If None, doesn't save.
        merge_strategy (str): How to handle duplicate categories:
            - "combine": Combine lists from duplicate categories
            - "prefix": Add file prefix to category names to avoid conflicts
            
    Returns:
        Dict[str, List[str]]: The merged JSON data.
        
    Raises:
        FileNotFoundError: If any input file doesn't exist.
        json.JSONDecodeError: If any file contains invalid JSON.
        ValueError: If merge_strategy is not recognized.
    """
    if merge_strategy not in ["combine", "prefix"]:
        raise ValueError("merge_strategy must be 'combine', or 'prefix'")
    
    merged_data = {}
    
    for i, file_path in enumerate(file_paths):
        try:
            data = read_json_file(file_path)
            
            # Convert list format to dict if necessary
            if isinstance(data, list):
                data = {"items": data}
            
            # Handle different merge strategies
            for category, texts in data.items():
                if merge_strategy == "combine":
                    # Combine lists from duplicate categories
                    if category in merged_data:
                        merged_data[category].extend(texts)
                    else:
                        merged_data[category] = texts[:]  # Create a copy
                        
                elif merge_strategy == "prefix":
                    # Add file prefix to avoid conflicts
                    file_prefix = os.path.splitext(os.path.basename(file_path))[0]
                    prefixed_category = f"{file_prefix}_{category}"
                    merged_data[prefixed_category] = texts[:]
                    
        except Exception as e:
            raise Exception(f"Error processing file {file_path}: {str(e)}")
    
    # Save to output file if specified
    if output_file:
        write_json_file(merged_data, output_file)
        print(f"Merged data saved to: {output_file}")
    
    return merged_data


def merge_json_files_advanced(file_paths: List[str], output_file: Optional[str] = None,
                             remove_duplicates: bool = True,
                             sort_categories: bool = True) -> Dict[str, List[str]]:
    """
    Advanced merge function with additional processing options.
    
    Parameters:
        file_paths (List[str]): List of paths to JSON files to merge.
        output_file (str, optional): Path to save the merged result.
        remove_duplicates (bool): Remove duplicate entries within each category.
        sort_categories (bool): Sort categories alphabetically.
        
    Returns:
        Dict[str, List[str]]: The merged and processed JSON data.
    """
    merged_data = merge_json_files(file_paths, merge_strategy="combine")
    
    # Remove duplicates within each category if requested
    if remove_duplicates:
        for category, texts in merged_data.items():
            # Preserve order while removing duplicates
            seen = set()
            unique_texts = []
            for text in texts:
                if text not in seen:
                    seen.add(text)
                    unique_texts.append(text)
            merged_data[category] = unique_texts
    
    # Sort categories if requested
    if sort_categories:
        merged_data = dict(sorted(merged_data.items()))
    
    # Save to output file if specified
    if output_file:
        write_json_file(merged_data, output_file)
        print(f"Advanced merged data saved to: {output_file}")
    
    return merged_data


def merge_json_files_general(file_paths: List[str], output_file: Optional[str] = None,
                           merge_strategy: str = "combine") -> Any:
    """
    Merges multiple JSON files with any JSON content structure.
    
    Parameters:
        file_paths (List[str]): List of paths to JSON files to merge.
        output_file (Optional[str]): Path to save the merged result. If None, doesn't save.
        merge_strategy (str): How to handle merging:
            - "combine": Merge dictionaries, combine lists, overwrite primitives
            - "list": Put all file contents into a list
            
    Returns:
        Any: The merged JSON data.
        
    Raises:
        FileNotFoundError: If any input file doesn't exist.
        json.JSONDecodeError: If any file contains invalid JSON.
        ValueError: If merge_strategy is not recognized.
    """
    if merge_strategy not in ["combine", "list"]:
        raise ValueError("merge_strategy must be 'combine' or 'list'")
    
    if merge_strategy == "list":
        # Put all file contents into a list
        merged_data = []
        for file_path in file_paths:
            data = read_json_file(file_path)
            merged_data.append(data)
    
    else:  # combine strategy
        merged_data = None
        for file_path in file_paths:
            data = read_json_file(file_path)
            if merged_data is None:
                # First file - deep copy the data
                merged_data = _deep_copy_json(data)
            else:
                # Merge with existing data
                merged_data = _merge_json_recursive(merged_data, data)
    
    # Save to output file if specified
    if output_file:
        write_json_file(merged_data, output_file)
        print(f"General merged data saved to: {output_file}")
    
    return merged_data


def _deep_copy_json(data: Any) -> Any:
    """
    Creates a deep copy of JSON-compatible data.
    
    Parameters:
        data (Any): The data to copy.
        
    Returns:
        Any: A deep copy of the data.
    """
    if isinstance(data, dict):
        return {key: _deep_copy_json(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_deep_copy_json(item) for item in data]
    # Primitives (str, int, float, bool, None) are immutable
    return data


def _merge_json_recursive(target: Any, source: Any) -> Any:
    """
    Recursively merges two JSON-compatible data structures.
    
    Parameters:
        target (Any): The target data structure to merge into.
        source (Any): The source data structure to merge from.
        
    Returns:
        Any: The merged data structure.
        
    Raises:
        ValueError: If incompatible types are being merged.
    """
    # If both are dictionaries, merge them
    if isinstance(target, dict) and isinstance(source, dict):
        result = target.copy()
        for key, value in source.items():
            if key in result:
                result[key] = _merge_json_recursive(result[key], value)
            else:
                result[key] = _deep_copy_json(value)
        return result

    # If both are lists, combine them
    if isinstance(target, list) and isinstance(source, list):
        return target + source

    # If types don't match or are primitives, source overwrites target
    return _deep_copy_json(source)


def merge_json_files_advanced_general(file_paths: List[str], 
                                    output_file: Optional[str] = None,
                                    merge_strategy: str = "combine",
                                    remove_duplicate_dicts: bool = False,
                                    sort_dict_keys: bool = False) -> Any:
    """
    Advanced general merge function with additional processing options.
    
    Parameters:
        file_paths (List[str]): List of paths to JSON files to merge.
        output_file (Optional[str]): Path to save the merged result.
        merge_strategy (str): Merge strategy for general content.
        remove_duplicate_dicts (bool): Remove duplicate dictionary entries from lists.
        sort_dict_keys (bool): Sort dictionary keys alphabetically.
        
    Returns:
        Any: The merged and processed JSON data.
    """
    merged_data = merge_json_files_general(file_paths, merge_strategy=merge_strategy)

    # Post-process the merged data
    if remove_duplicate_dicts or sort_dict_keys:
        merged_data = _post_process_json(merged_data, remove_duplicate_dicts, sort_dict_keys)

    # Save to output file if specified
    if output_file:
        write_json_file(merged_data, output_file)
        print(f"Advanced general merged data saved to: {output_file}")

    return merged_data


def _post_process_json(data: Any, remove_duplicate_dicts: bool, sort_dict_keys: bool) -> Any:
    """
    Post-processes JSON data with various options.
    
    Parameters:
        data (Any): The data to process.
        remove_duplicate_dicts (bool): Remove duplicate dictionaries from lists.
        sort_dict_keys (bool): Sort dictionary keys.
        
    Returns:
        Any: The processed data.
    """
    if isinstance(data, dict):
        result = {}
        keys = sorted(data.keys()) if sort_dict_keys else data.keys()
        for key in keys:
            result[key] = _post_process_json(data[key], remove_duplicate_dicts, sort_dict_keys)
        return result

    if isinstance(data, list):
        # Process each item in the list
        processed_list = [_post_process_json(item, remove_duplicate_dicts, sort_dict_keys) for item in data]

        # Remove duplicate dictionaries if requested
        if remove_duplicate_dicts:
            seen_dicts = set()
            unique_list = []
            for item in processed_list:
                if isinstance(item, dict):
                    # Convert dict to a hashable representation for comparison
                    dict_str = json.dumps(item, sort_keys=True)
                    if dict_str not in seen_dicts:
                        seen_dicts.add(dict_str)
                        unique_list.append(item)
                else:
                    unique_list.append(item)
            return unique_list
        else:
            return processed_list

    # Primitives remain unchanged
    return data
