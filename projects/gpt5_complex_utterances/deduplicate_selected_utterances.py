#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Duplicate Utterance Remover - Remove duplicate utterances from selected_utterances.json files.
It applies a position-based removal strategy: a later-positioned utterance is removed when duplication occurs.

The output file preserves the same JSON formatting (indentation) as the input file.

Example usage:
    python deduplicate_selected_utterances.py input.json output.json
"""

import os
import sys
import json
import fire
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import logging

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def read_utterances_file(file_path: str) -> Tuple[Dict[str, Any], str]:
    """
    Read a selected_utterances.json file and return the parsed data and original text.

    Args:
        file_path: Path to the selected_utterances.json file

    Returns:
        Tuple of (parsed_data, original_text)
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            original_text = f.read()
        data = json.loads(original_text)
        logger.info(f"Successfully loaded data from: {file_path}")
        return data, original_text
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise


def find_duplicates(utterances: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """
    Find duplicate utterances and return their indices.

    Args:
        utterances: List of utterance dictionaries

    Returns:
        Dictionary mapping utterance text to list of indices where it appears
    """
    utterance_indices = defaultdict(list)

    for i, utterance in enumerate(utterances):
        utterance_text = utterance.get("utterance", "")
        if utterance_text:  # Only consider non-empty utterances
            utterance_indices[utterance_text].append(i)

    # Filter to only return duplicates (utterances that appear more than once)
    duplicates = {
        text: indices for text, indices in utterance_indices.items() if len(indices) > 1
    }

    return duplicates


def apply_removal_strategy(
    utterances: List[Dict[str, Any]], duplicates: Dict[str, List[int]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Apply the duplicate removal strategy and return the cleaned utterances.

    Strategy:
    - For each group of duplicate utterances, keep the first occurrence (earliest position)
    - Remove all later occurrences regardless of their selection round

    Args:
        utterances: List of utterance dictionaries
        duplicates: Dictionary mapping utterance text to list of indices

    Returns:
        Tuple of (cleaned_utterances_list, removal_stats)
    """
    indices_to_remove = set()
    removal_stats = {
        "total_removed": 0,
        "removed_by_round": defaultdict(int),
    }

    for utterance_text, indices in duplicates.items():
        if len(indices) <= 1:
            continue  # Skip non-duplicates

        # Sort indices to identify the first occurrence (keep) and later ones (remove)
        sorted_indices = sorted(indices)
        indices_to_remove.update(sorted_indices[1:])  # Remove all but the first
        
        # Track removals by round
        for idx in sorted_indices[1:]:
            round_num = utterances[idx].get("selected_round", "unknown")
            removal_stats["removed_by_round"][round_num] += 1
        
        logger.debug(
            f"Removing {len(sorted_indices) - 1} later duplicates of: {utterance_text[:50]}..."
        )

    # Create cleaned utterances list
    cleaned_utterances = [
        utterance
        for i, utterance in enumerate(utterances)
        if i not in indices_to_remove
    ]

    removal_stats["total_removed"] = len(indices_to_remove)

    return cleaned_utterances, removal_stats


def print_removal_summary(
    original_count: int,
    final_count: int,
    removal_stats: Dict[str, Any],
    duplicates: Dict[str, List[int]],
):
    """
    Print a summary of the duplicate removal process.

    Args:
        original_count: Original number of utterances
        final_count: Final number of utterances after removal
        removal_stats: Dictionary with removal statistics
        duplicates: Dictionary of found duplicates
    """
    print("\n" + "=" * 60)
    print("DUPLICATE REMOVAL SUMMARY")
    print("=" * 60)

    print(f"📊 Original Utterances: {original_count}")
    print(f"🧹 Final Utterances: {final_count}")
    print(f"🗑️  Total Removed: {removal_stats['total_removed']}")
    print(
        f"📉 Reduction: {(removal_stats['total_removed'] / original_count * 100):.1f}%"
    )

    print(f"\n📋 Duplicate Groups Found: {len(duplicates)}")

    # Show removals by round
    if removal_stats.get("removed_by_round"):
        print(f"\n� Removed utterances by round:")
        for round_num in sorted(removal_stats["removed_by_round"].keys()):
            count = removal_stats["removed_by_round"][round_num]
            print(f"   Round {round_num}: {count} utterances removed")

    print("=" * 60 + "\n")


def main(
    input_path: str,
    output_path: str,
) -> None:
    """
    Remove duplicate utterances from a selected_utterances.json file.

    Args:
        input_path: Path to the input selected_utterances.json file (required)
        output_path: Path to save the cleaned file (required)
    """
    try:
        # Validate input file
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Validate output path is provided
        if not output_path:
            raise ValueError("Output path must be specified")

        # Read the input file
        logger.info(f"Reading utterance data from: {input_path}")
        data, original_text = read_utterances_file(input_path)

        selected_utterances = data.get("selected_utterances", [])
        if not selected_utterances:
            logger.warning("No selected_utterances found in the data")
            return

        original_count = len(selected_utterances)
        logger.info(f"Found {original_count} utterances to process")

        # Find duplicates
        logger.info("Analyzing duplicate utterances...")
        duplicates = find_duplicates(selected_utterances)

        if not duplicates:
            logger.info("No duplicate utterances found. No changes needed.")
            return

        logger.info(f"Found {len(duplicates)} groups of duplicate utterances")

        # Apply removal strategy
        logger.info("Applying duplicate removal strategy...")
        cleaned_utterances, removal_stats = apply_removal_strategy(
            selected_utterances, duplicates
        )

        final_count = len(cleaned_utterances)

        # Print summary
        print_removal_summary(original_count, final_count, removal_stats, duplicates)

        # Update the data with cleaned utterances
        data["selected_utterances"] = cleaned_utterances

        # Add metadata about the cleaning process
        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["duplicate_removal"] = {
            "original_count": original_count,
            "final_count": final_count,
            "removed_count": removal_stats["total_removed"],
            "removal_stats": removal_stats,
            "strategy": "Position-based removal: keep first occurrence, remove later duplicates",
        }

        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Detect original indentation style from the input file
        indent = 2  # default value
        if '"selected_utterances"' in original_text:
            # Try to detect indentation by looking at the structure
            lines = original_text.split("\n")
            for line in lines:
                if '"selected_utterances"' in line and line.strip().endswith("["):
                    # Look at the next line for indentation
                    next_idx = lines.index(line) + 1
                    if next_idx < len(lines):
                        next_line = lines[next_idx]
                        leading_spaces = len(next_line) - len(next_line.lstrip())
                        if leading_spaces > 0:
                            indent = leading_spaces
                            break

        # Save the cleaned data
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

        logger.info(f"Successfully saved cleaned data to: {output_path}")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Use fire for command line arguments
    fire.Fire(main)
