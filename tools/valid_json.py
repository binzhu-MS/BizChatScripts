"""
JSON Validator Tool

This module provides functionality to validate JSON files in multiple formats (JSON, JSONC, JSON5).
It attempts to parse files using different JSON parsers and reports which format is valid.

Author: Bin Zhu
Date: July 30, 2025
"""

import os
from typing import Dict, Any, Optional
import json
import json5
import commentjson
import fire  # For command line interface

# Import from the Utils package (when PYTHONPATH includes the code directory)
from utils.json_utils import write_json_file


def validate_json_content(json_string: str) -> Dict[str, Any]:
    """
    Validates JSON content using multiple parsers.

    Parameters:
        json_string (str): The JSON string to validate.

    Returns:
        Dict[str, Any]: A dictionary containing validation results.
    """
    results = {
        "valid_formats": [],
        "errors": {},
        "parsed_data": None,
        "format_used": None,
    }

    # Try to parse as standard JSON
    try:
        json_obj = json.loads(json_string)
        results["valid_formats"].append("JSON")
        if results["parsed_data"] is None:
            results["parsed_data"] = json_obj
            results["format_used"] = "JSON"
    except (ValueError, json.JSONDecodeError) as e:
        results["errors"]["JSON"] = str(e)

    # Try to parse as JSONC (JSON with comments)
    try:
        jsonc_obj = commentjson.loads(json_string)
        results["valid_formats"].append("JSONC")
        if results["parsed_data"] is None:
            results["parsed_data"] = jsonc_obj
            results["format_used"] = "JSONC"
    except (ValueError, json.JSONDecodeError) as e:
        results["errors"]["JSONC"] = str(e)

    # Try to parse as JSON5
    try:
        json5_obj = json5.loads(json_string)
        results["valid_formats"].append("JSON5")
        if results["parsed_data"] is None:
            results["parsed_data"] = json5_obj
            results["format_used"] = "JSON5"
    except (ValueError, json.JSONDecodeError) as e:
        results["errors"]["JSON5"] = str(e)

    return results


def print_validation_results(results: Dict[str, Any], file_path: str) -> None:
    """
    Prints the validation results in a readable format.

    Parameters:
        results (Dict[str, Any]): The validation results dictionary.
        file_path (str): The path of the validated file.

    Returns:
        None
    """
    print("=" * 60)
    print("JSON VALIDATION REPORT")
    print("=" * 60)

    print(f"\nFILE: {file_path}")

    if results["valid_formats"]:
        print(f"\n✅ VALID FORMATS:")
        for fmt in results["valid_formats"]:
            print(f"  - {fmt}")

        print(f"\n📋 PARSING RESULT:")
        print(f"  Format used: {results['format_used']}")

        if isinstance(results["parsed_data"], dict):
            print(f"  Data type: Dictionary")
            print(f"  Keys count: {len(results['parsed_data'])}")
            if len(results["parsed_data"]) <= 10:
                print(f"  Keys: {list(results['parsed_data'].keys())}")
        elif isinstance(results["parsed_data"], list):
            print(f"  Data type: List")
            print(f"  Items count: {len(results['parsed_data'])}")
        else:
            print(f"  Data type: {type(results['parsed_data']).__name__}")
    else:
        print(f"\n❌ NO VALID FORMATS FOUND")

    if results["errors"]:
        print(f"\n🚫 ERRORS:")
        for fmt, error in results["errors"].items():
            if not error.endswith("library not installed"):
                print(f"  {fmt}: {error}")

    print("=" * 60)


def save_validation_results(results: Dict[str, Any], output_file: str) -> None:
    """
    Saves the validation results to a JSON file.

    Parameters:
        results (Dict[str, Any]): The validation results dictionary.
        output_file (str): The output file path.

    Returns:
        None
    """
    # Remove the parsed_data from results for the saved file (might be large)
    save_results = results.copy()
    if "parsed_data" in save_results:
        del save_results["parsed_data"]

    write_json_file(save_results, output_file)
    print(f"\nValidation results saved to: {output_file}")


def main(
    input_path: str,
    output_path: Optional[str] = None,
    silent: bool = False,
    save_parsed: bool = False,
) -> None:
    """
    Validate JSON file formats and generate a comprehensive report.

    Args:
        input_path: Path to the input JSON file to validate (required)
        output_path: Path to save validation results (defaults to input filename with _validation.json extension)
        silent: If True, only saves to file without printing to console
        save_parsed: If True, includes parsed data in the output file

    Example usage:
        # Validate a specific file
        python valid_json.py data.json

        # Validate and save results
        python valid_json.py data.json --output_path=validation_results.json

        # Silent mode (no console output)
        python valid_json.py data.json --silent=True

        # Include parsed data in output
        python valid_json.py data.json --save_parsed=True
    """
    try:
        # Validate input_path was provided
        if input_path is None:
            raise ValueError(
                "Error: input_path is required. Please specify a valid input file path."
            )

        # Validate input file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Set default output file path if not provided
        if output_path is None or output_path.strip() == "":
            base, ext = os.path.splitext(input_path)
            output_path = base + "_validation.json"

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Read and validate the JSON file
        if not silent:
            print(f"Reading file: {input_path}")

        with open(input_path, "r", encoding="utf-8") as file:
            json_string = file.read()

        if not silent:
            print("Validating JSON formats...")

        results = validate_json_content(json_string)

        # Print validation results to console unless in silent mode
        if not silent:
            print_validation_results(results, input_path)

        # Save validation results to file if output_path is provided
        if output_path:
            save_results = results.copy()
            if not save_parsed and "parsed_data" in save_results:
                del save_results["parsed_data"]
            save_validation_results(save_results, output_path)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    # Use fire for command line arguments
    fire.Fire(main)
