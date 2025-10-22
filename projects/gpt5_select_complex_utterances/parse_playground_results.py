#!/usr/bin/env python3
"""
Playground Results Parser

This program parses playground output files to extract input data and results
where the evaluator name starts with "v11". The output is formatted similar to
classification data structure with "switching_class" as the evaluation result.

Author: Generated for BizChatScripts project
Date: August 6, 2025
"""

import json
import logging
import os
import sys
import fire
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Add parent directories to path to import utils
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from utils.json_utils import read_json_file, write_json_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlaygroundResultsParser:
    """
    Parser to extract input data and v11 evaluation results from playground output files.
    """

    def __init__(self):
        self.v11_prefix = "v11"

    def parse_playground_results(
        self,
        input_file: str,
        output_file: str,
        show_stats: bool = True,
    ) -> Dict[str, Any]:
        """
        Parse playground results and extract v11 evaluations.

        Args:
            input_file: Path to playground output JSON file
            output_file: Path to save parsed results JSON file
            show_stats: Whether to display parsing statistics

        Returns:
            Dictionary with parsing statistics
        """
        logger.info(f"Loading playground results from: {input_file}")

        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Load playground results data
        playground_data = read_json_file(input_file)
        if not isinstance(playground_data, list):
            raise ValueError("Expected playground data to be a list of entries")

        logger.info(f"Loaded {len(playground_data)} playground entries")

        # Parse entries
        parsed_entries = []
        stats = {
            "total_entries": len(playground_data),
            "entries_with_v11": 0,
            "entries_without_v11": 0,
            "v11_evaluators_found": set(),
        }

        for entry in playground_data:
            parsed_entry = self._parse_playground_entry(entry, stats)
            if parsed_entry:
                parsed_entries.append(parsed_entry)

        # Group parsed entries by category (similar to classification data structure)
        grouped_results = self._group_parsed_entries(parsed_entries)

        # Save results
        write_json_file(grouped_results, output_file)
        logger.info(f"Saved {len(parsed_entries)} parsed entries to: {output_file}")

        # Update final statistics
        stats["parsed_entries"] = len(parsed_entries)
        stats["v11_evaluators_found"] = list(stats["v11_evaluators_found"])

        # Display statistics if requested
        if show_stats:
            self._display_parsing_stats(stats, input_file, output_file)

        return stats

    def _parse_playground_entry(
        self, entry: Dict[str, Any], stats: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Parse a single playground entry to extract input data and v11 results.

        Args:
            entry: Single playground entry
            stats: Statistics dictionary to update

        Returns:
            Parsed entry dictionary or None if no v11 results found
        """
        try:
            # Extract input parameters
            if "input" not in entry or "parameters" not in entry["input"]:
                logger.warning("Entry missing input parameters, skipping")
                return None

            # Parse the parameters JSON string
            params_str = entry["input"]["parameters"]
            if isinstance(params_str, str):
                params = json.loads(params_str)
            else:
                params = params_str

            # Extract utterance (required field)
            utterance = params.get("utterance")
            if not utterance:
                logger.warning("Entry missing utterance, skipping")
                return None

            # Look for v11 results
            v11_results = self._extract_v11_results(entry.get("results", []), stats)
            if not v11_results:
                stats["entries_without_v11"] += 1
                return None

            stats["entries_with_v11"] += 1

            # Get the segment name for grouping
            segment_name = params.get("segment", "unknown")

            # Build parsed entry similar to classification data structure
            parsed_entry = {
                "utterance": utterance,
                "switching_class": v11_results.get("output", "unknown"),
                "confidence": params.get("confidence", ""),
                "reasoning": params.get("reasoning", ""),
                "_segment": segment_name,  # Internal field for grouping
            }

            # Add complexity_indicators if available
            if "complexity_indicators" in params:
                parsed_entry["complexity_indicators"] = params["complexity_indicators"]

            # Add any other fields from original data (except the ones we've already handled)
            for key, value in params.items():
                if key not in [
                    "utterance",
                    "segment",
                    "classification",  # This becomes switching_class
                    "confidence",
                    "reasoning",
                    "complexity_indicators",
                ]:
                    parsed_entry[key] = value

            return parsed_entry

        except Exception as e:
            logger.warning(f"Error parsing entry: {e}")
            return None

    def _extract_v11_results(
        self, results: List[Dict[str, Any]], stats: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract results from evaluators whose name starts with v11.

        Args:
            results: List of result dictionaries
            stats: Statistics dictionary to update

        Returns:
            First v11 result found or None
        """
        for result in results:
            evaluator_name = result.get("name", "")
            if evaluator_name.startswith(self.v11_prefix):
                stats["v11_evaluators_found"].add(evaluator_name)
                return result

        return None

    def _extract_confidence_score(self, v11_result: Dict[str, Any]) -> float:
        """
        Extract confidence score from v11 evaluation results.

        Args:
            v11_result: V11 evaluation result

        Returns:
            Confidence score (0.0-1.0) or 0.0 if not found
        """
        try:
            # Look for confidence in evaluations
            evaluations = v11_result.get("evaluations", {})
            for eval_name, eval_data in evaluations.items():
                if isinstance(eval_data, dict):
                    for sub_name, sub_data in eval_data.items():
                        if "score" in str(sub_name).lower():
                            score = sub_data
                            if isinstance(score, (int, float)):
                                return min(
                                    max(score / 100.0, 0.0), 1.0
                                )  # Normalize to 0-1
            return 0.0
        except Exception:
            return 0.0

    def _group_parsed_entries(
        self, parsed_entries: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group parsed entries by segment, similar to classification data structure.

        Args:
            parsed_entries: List of parsed entries

        Returns:
            Dictionary grouped by segment
        """
        grouped = {}

        for entry in parsed_entries:
            segment = entry.pop(
                "_segment", "unknown"
            )  # Remove internal field and use for grouping
            if segment not in grouped:
                grouped[segment] = []
            grouped[segment].append(entry)

        # If no segments found, create a default group
        if not grouped and parsed_entries:
            grouped["playground_parsed"] = parsed_entries

        return grouped

    def _display_parsing_stats(
        self, stats: Dict[str, Any], input_file: str, output_file: str
    ) -> None:
        """Display parsing statistics."""
        print("\n" + "=" * 80)
        print("PLAYGROUND RESULTS PARSING STATISTICS")
        print("=" * 80)

        print(f"\n📊 INPUT DATA SUMMARY:")
        print(f"  Input File: {os.path.basename(input_file)}")
        print(f"  Total Playground Entries: {stats['total_entries']}")

        print(f"\n🎯 V11 EVALUATOR ANALYSIS:")
        print(f"  Entries with V11 Results: {stats['entries_with_v11']}")
        print(f"  Entries without V11 Results: {stats['entries_without_v11']}")

        if stats["v11_evaluators_found"]:
            print(f"  V11 Evaluators Found:")
            for evaluator in sorted(stats["v11_evaluators_found"]):
                print(f"    - {evaluator}")

        print(f"\n✅ PARSING RESULTS:")
        print(f"  Successfully Parsed Entries: {stats['parsed_entries']}")
        if stats["total_entries"] > 0:
            success_rate = round(
                stats["parsed_entries"] / stats["total_entries"] * 100, 1
            )
            print(f"  Success Rate: {success_rate}%")

        print(f"\n📁 OUTPUT:")
        print(f"  Results File: {output_file}")
        print(f"  Format: Classification-style JSON")
        print("=" * 80)


def parse_playground_results(
    input_file: str,
    output_file: str,
    show_stats: bool = True,
) -> None:
    """
    Parse playground output file to extract input data and v11 evaluation results.

    Args:
        input_file: Path to playground output JSON file
        output_file: Path to save parsed results JSON file
        show_stats: Whether to display parsing statistics

    Examples:
        # Parse playground results
        parse_playground_results("playground_output.json", "parsed_results.json")

        # Parse without showing statistics
        parse_playground_results("playground_output.json", "parsed_results.json", show_stats=False)

    Notes:
        - Only extracts results from evaluators whose name starts with "v11"
        - Output "switching_class" contains the v11 evaluator's output
        - Preserves all original input parameters as "original_*" fields
        - Groups results by original segment similar to classification data structure
    """
    try:
        # Validate input file
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Create output directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        # Create parser and process data
        parser = PlaygroundResultsParser()
        stats = parser.parse_playground_results(
            input_file=input_file,
            output_file=output_file,
            show_stats=show_stats,
        )

        logger.info("Parsing completed successfully!")

    except Exception as e:
        logger.error(f"Error during parsing: {e}")
        raise


def main():
    """Main entry point for the parser."""
    fire.Fire(parse_playground_results)


if __name__ == "__main__":
    main()
