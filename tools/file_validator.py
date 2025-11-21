#!/usr/bin/env python3
"""
File Validator - Modular Validation Tool

Provides validation utilities for JSON, JSONC, JSON5, and JSONL files
with detailed error reporting and multi-format support.

Available Commands:
    validate_json     - Validate JSON files (JSON/JSONC/JSON5)
    validate_jsonl    - Validate JSONL (JSON Lines) files line-by-line
    validate_auto     - Auto-detect format and validate accordingly

Supported Formats:
    - JSON: Standard JSON format
    - JSONC: JSON with comments
    - JSON5: Extended JSON with more flexible syntax
    - JSONL: JSON Lines (one JSON object per line)

Usage Examples:
    # Validate JSONL file
    python file_validator.py validate_jsonl --input_file="data.jsonl"
    
    # Validate JSON with multiple format support
    python file_validator.py validate_json --input_file="data.json"
    
    # Auto-detect and validate
    python file_validator.py validate_auto --input_file="data.json"
    
    # Save validation results
    python file_validator.py validate_json --input_file="data.json" \
        --output_file="results.json"

Programmatic Usage:
    from tools.file_validator import validate_jsonl, validate_json
    
    # JSONL validation
    results = validate_jsonl(input_file="data.jsonl")
    
    # JSON validation (supports JSON/JSONC/JSON5)
    results = validate_json(input_file="data.json")

Author: Bin Zhu
Date: November 21, 2025
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import fire

# Try importing optional parsers for JSONC and JSON5 support
try:
    import commentjson
    HAS_COMMENTJSON = True
except ImportError:
    HAS_COMMENTJSON = False

try:
    import json5
    HAS_JSON5 = True
except ImportError:
    HAS_JSON5 = False

# Configure logging with default WARNING level
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==========================================================================
# Base Classes
# ==========================================================================

class BaseValidator:
    """Base class providing common functionality for all validators."""
    
    def __init__(self):
        self.results = {}
        self.errors = []
    
    def _setup_logging(self, verbose: bool = False):
        """Configure logging level."""
        if isinstance(verbose, str):
            verbose = verbose.lower() in ('true', '1', 'yes')
        
        level = logging.DEBUG if verbose else logging.WARNING
        logging.getLogger().setLevel(level)
    
    def _print_header(self, title: str):
        """Print a formatted header."""
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def _save_results(self, output_file: str):
        """Save validation results to a JSON file."""
        output_path = Path(output_file)
        
        if output_path.parent != Path('.'):
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Validation results saved to: {output_file}")
        print(f"\n💾 Results saved to: {output_file}")


# ==========================================================================
# JSONL Validation Module
# ==========================================================================

class JSONLValidator(BaseValidator):
    """Validates JSONL files line-by-line."""
    
    def validate(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        verbose: bool = False,
        stats_only: bool = False,
        max_errors: int = 100,
        sample_size: int = 3
    ) -> Dict[str, Any]:
        """
        Validate a JSONL file by parsing each line as JSON.
        
        Args:
            input_file: Path to the JSONL file to validate
            output_file: Optional path to save validation results as JSON
            verbose: Enable verbose logging (show each line validation)
            stats_only: Only show statistics, skip detailed error reporting
            max_errors: Maximum number of errors to collect (default: 100)
            sample_size: Number of sample valid objects to collect (default: 3)
            
        Returns:
            Dictionary with validation results
        """
        self._setup_logging(verbose)
        
        # Validate input file exists
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Print header
        self._print_header("JSONL VALIDATION")
        print(f"Input file:  {input_file}")
        print(f"File size:   {self._format_size(input_path.stat().st_size)}")
        
        # Reset results
        self.results = {
            "is_valid": True,
            "total_lines": 0,
            "valid_lines": 0,
            "error_count": 0,
            "empty_lines": 0,
            "errors": [],
            "sample_data": []
        }
        
        # Process file line by line
        logger.info(f"Validating JSONL file: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                self.results["total_lines"] += 1
                
                # Skip empty lines
                if not line.strip():
                    self.results["empty_lines"] += 1
                    if verbose:
                        logger.debug(f"Line {line_num}: Empty line (skipped)")
                    continue
                
                # Try to parse as JSON
                try:
                    obj = json.loads(line)
                    self.results["valid_lines"] += 1
                    
                    # Collect sample data
                    if len(self.results["sample_data"]) < sample_size:
                        self.results["sample_data"].append({
                            "line_number": line_num,
                            "keys": list(obj.keys()) if isinstance(obj, dict) else None,
                            "type": type(obj).__name__
                        })
                    
                    if verbose:
                        logger.debug(f"Line {line_num}: Valid JSON")
                
                except json.JSONDecodeError as e:
                    self.results["error_count"] += 1
                    self.results["is_valid"] = False
                    
                    # Store error details (up to max_errors)
                    if len(self.results["errors"]) < max_errors:
                        error_detail = {
                            "line_number": line_num,
                            "error_type": "JSONDecodeError",
                            "error_message": str(e),
                            "line_preview": line[:200] + "..." if len(line) > 200 else line
                        }
                        self.results["errors"].append(error_detail)
                    
                    logger.error(f"Line {line_num}: JSON parse error - {e}")
                
                except Exception as e:
                    self.results["error_count"] += 1
                    self.results["is_valid"] = False
                    
                    # Store error details (up to max_errors)
                    if len(self.results["errors"]) < max_errors:
                        error_detail = {
                            "line_number": line_num,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "line_preview": line[:200] + "..." if len(line) > 200 else line
                        }
                        self.results["errors"].append(error_detail)
                    
                    logger.error(f"Line {line_num}: Unexpected error - {e}")
                
                # Progress reporting for large files
                if self.results["total_lines"] % 1000 == 0:
                    logger.info(
                        f"Progress: {self.results['total_lines']} lines processed "
                        f"({self.results['valid_lines']} valid, "
                        f"{self.results['error_count']} errors)"
                    )
        
        # Print results
        self._print_results(stats_only)
        
        # Save results to file if requested
        if output_file:
            self._save_results(output_file)
        
        return self.results
    
    def _print_results(self, stats_only: bool = False):
        """Print validation results."""
        self._print_header("VALIDATION RESULTS")
        
        # Overall status
        if self.results["is_valid"]:
            print("✅ Status: VALID JSONL")
        else:
            print("❌ Status: INVALID JSONL")
        
        print()
        
        # Statistics
        print("📊 STATISTICS:")
        print(f"  Total lines:     {self.results['total_lines']}")
        print(f"  Valid lines:     {self.results['valid_lines']}")
        print(f"  Error lines:     {self.results['error_count']}")
        print(f"  Empty lines:     {self.results['empty_lines']}")
        
        if self.results['total_lines'] > 0:
            valid_pct = (self.results['valid_lines'] / self.results['total_lines']) * 100
            print(f"  Success rate:    {valid_pct:.1f}%")
        
        # Error details (if not stats_only)
        if not stats_only and self.results["errors"]:
            print()
            print("🚫 ERRORS (first 10):")
            for error in self.results["errors"][:10]:
                print(f"  Line {error['line_number']}: {error['error_type']}")
                print(f"    Message: {error['error_message']}")
                if len(error['line_preview']) < 150:
                    print(f"    Content: {error['line_preview']}")
                print()
            
            if len(self.results["errors"]) > 10:
                print(f"  ... and {len(self.results['errors']) - 10} more errors")
        
        print("=" * 70)


# ==========================================================================
# JSON Validation Module
# ==========================================================================

class JSONValidator(BaseValidator):
    """Validates standard JSON files with multiple format support."""
    
    def validate(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        verbose: bool = False,
        save_parsed: bool = False
    ) -> Dict[str, Any]:
        """
        Validate a JSON file (supports JSON, JSONC, JSON5).
        
        Args:
            input_file: Path to the JSON file to validate
            output_file: Optional path to save validation results
            verbose: Enable verbose logging
            save_parsed: Include parsed data in output file
            
        Returns:
            Dictionary with validation results
        """
        self._setup_logging(verbose)
        
        # Validate input file exists
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Print header
        self._print_header("JSON VALIDATION")
        print(f"Input file:  {input_file}")
        print(f"File size:   {self._format_size(input_path.stat().st_size)}")
        
        # Read file content
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try parsing with different parsers
        self.results = {
            "is_valid": False,
            "valid_formats": [],
            "format_used": None,
            "data_type": None,
            "parsed_data": None,
            "errors": {}
        }
        
        # Try standard JSON
        try:
            data = json.loads(content)
            self.results["valid_formats"].append("JSON")
            if self.results["parsed_data"] is None:
                self.results["is_valid"] = True
                self.results["format_used"] = "JSON"
                self.results["parsed_data"] = data
                self.results["data_type"] = type(data).__name__
                
                if isinstance(data, dict):
                    self.results["keys_count"] = len(data.keys())
                    if len(data.keys()) <= 10:
                        self.results["keys"] = list(data.keys())
                elif isinstance(data, list):
                    self.results["items_count"] = len(data)
        
        except (ValueError, json.JSONDecodeError) as e:
            self.results["errors"]["JSON"] = str(e)
        
        # Try JSONC (JSON with comments)
        if HAS_COMMENTJSON:
            try:
                data = commentjson.loads(content)
                self.results["valid_formats"].append("JSONC")
                if self.results["parsed_data"] is None:
                    self.results["is_valid"] = True
                    self.results["format_used"] = "JSONC"
                    self.results["parsed_data"] = data
                    self.results["data_type"] = type(data).__name__
                    
                    if isinstance(data, dict):
                        self.results["keys_count"] = len(data.keys())
                        if len(data.keys()) <= 10:
                            self.results["keys"] = list(data.keys())
                    elif isinstance(data, list):
                        self.results["items_count"] = len(data)
            
            except (ValueError, json.JSONDecodeError) as e:
                self.results["errors"]["JSONC"] = str(e)
        else:
            self.results["errors"]["JSONC"] = "commentjson not installed"
        
        # Try JSON5
        if HAS_JSON5:
            try:
                data = json5.loads(content)
                self.results["valid_formats"].append("JSON5")
                if self.results["parsed_data"] is None:
                    self.results["is_valid"] = True
                    self.results["format_used"] = "JSON5"
                    self.results["parsed_data"] = data
                    self.results["data_type"] = type(data).__name__
                    
                    if isinstance(data, dict):
                        self.results["keys_count"] = len(data.keys())
                        if len(data.keys()) <= 10:
                            self.results["keys"] = list(data.keys())
                    elif isinstance(data, list):
                        self.results["items_count"] = len(data)
            
            except (ValueError, json.JSONDecodeError) as e:
                self.results["errors"]["JSON5"] = str(e)
        else:
            self.results["errors"]["JSON5"] = "json5 library not installed"
        
        # Print results
        self._print_results()
        
        # Save results to file if requested
        if output_file:
            if not save_parsed and "parsed_data" in self.results:
                # Remove parsed data before saving (might be large)
                save_results = self.results.copy()
                del save_results["parsed_data"]
                original_results = self.results
                self.results = save_results
                self._save_results(output_file)
                self.results = original_results
            else:
                self._save_results(output_file)
        
        return self.results
    
    def _print_results(self):
        """Print validation results."""
        self._print_header("VALIDATION RESULTS")
        
        if self.results["is_valid"]:
            print("✅ Status: VALID")
            print()
            print("📋 VALID FORMATS:")
            for fmt in self.results["valid_formats"]:
                print(f"  - {fmt}")
            
            print()
            print("📊 PARSING RESULT:")
            print(f"  Format used: {self.results['format_used']}")
            print(f"  Data type:   {self.results['data_type']}")
            
            if "keys_count" in self.results:
                print(f"  Keys count:  {self.results['keys_count']}")
                if "keys" in self.results:
                    print(f"  Keys: {self.results['keys']}")
            elif "items_count" in self.results:
                print(f"  Items count: {self.results['items_count']}")
        else:
            print("❌ Status: NO VALID FORMATS FOUND")
            print()
            print("🚫 ERRORS:")
            for fmt, error in self.results["errors"].items():
                if not error.endswith("not installed"):
                    print(f"  {fmt}: {error}")
        
        print("=" * 70)


# ==========================================================================
# Module-Level Functions (for Fire CLI)
# ==========================================================================

def validate_jsonl(
    input_file: str,
    output_file: Optional[str] = None,
    verbose: bool = False,
    stats_only: bool = False,
    max_errors: int = 100,
    sample_size: int = 3
):
    """
    Validate JSONL file line-by-line.
    
    Args:
        input_file: Path to the JSONL file
        output_file: Optional output file for results
        verbose: Enable verbose logging
        stats_only: Only show statistics
        max_errors: Max errors to collect
        sample_size: Number of samples to collect
    """
    validator = JSONLValidator()
    validator.validate(
        input_file=input_file,
        output_file=output_file,
        verbose=verbose,
        stats_only=stats_only,
        max_errors=max_errors,
        sample_size=sample_size
    )


def validate_json(
    input_file: str,
    output_file: Optional[str] = None,
    verbose: bool = False,
    save_parsed: bool = False
):
    """
    Validate JSON file with multi-format support (JSON/JSONC/JSON5).
    
    Args:
        input_file: Path to the JSON file
        output_file: Optional output file for results
        verbose: Enable verbose logging
        save_parsed: Include parsed data in output file
    """
    validator = JSONValidator()
    validator.validate(
        input_file=input_file,
        output_file=output_file,
        verbose=verbose,
        save_parsed=save_parsed
    )


def validate_auto(
    input_file: str,
    output_file: Optional[str] = None,
    verbose: bool = False
):
    """
    Auto-detect format and validate (JSONL vs JSON).
    
    Detects JSONL by checking if first line is valid JSON
    and file has multiple lines.
    
    Args:
        input_file: Path to the file
        output_file: Optional output file for results
        verbose: Enable verbose logging
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Read first few lines to detect format
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [f.readline() for _ in range(5)]
    
    # If first line is valid JSON and we have multiple lines, assume JSONL
    try:
        json.loads(lines[0])
        non_empty_lines = [line for line in lines if line.strip()]
        if len(non_empty_lines) > 1:
            print("🔍 Auto-detected format: JSONL")
            validate_jsonl(
                input_file=input_file,
                output_file=output_file,
                verbose=verbose
            )
            return
    except (ValueError, json.JSONDecodeError):
        pass
    
    # Otherwise, try standard JSON
    print("🔍 Auto-detected format: JSON")
    validate_json(
        input_file=input_file,
        output_file=output_file,
        verbose=verbose
    )


# ==========================================================================
# CLI Entry Point
# ==========================================================================

if __name__ == "__main__":
    # Fire automatically exposes all module-level functions as CLI commands
    fire.Fire({
        'validate_jsonl': validate_jsonl,
        'validate_json': validate_json,
        'validate_auto': validate_auto,
    })

