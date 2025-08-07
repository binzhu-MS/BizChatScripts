"""
Simplified Iterative Utterance Selector with LLM analysis and simple resume functionality.
Only uses two files: utterance_selector.py and utterance_selector_core.py
"""

import json
import logging
import os
import re
import sys
from typing import Dict, List, Any, Union, Optional
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import LLM utilities
from llms import ChatCompletionLLMApplier, prompts, with_retries

logger = logging.getLogger(__name__)


class IterativeUtteranceSelector(ChatCompletionLLMApplier):
    """
    Iterative utterance selector with LLM analysis and resume.
    """

    # Model configuration - easy to adjust at class level
    DEFAULT_MODEL_CONFIG = {
        "model": "dev-gpt-41-longco-2025-04-14",
        "temperature": 0.1,  # Low temperature for consistency
        "max_tokens": 4000,  # Sufficient for selection analysis
    }

    # Load prompt that includes enterprise search context
    DEFAULT_PROMPT = prompts.get("utterance_selector", "0.1.0")

    # Required parent class attributes
    DEFAULT_THREADS = 1  # Single-threaded for simplicity
    DEFAULT_RETRIES = 3  # Retry failed LLM calls
    APPLICATION_MODE = None  # Not used in our case

    # Selection Configuration
    DEFAULT_INCREMENT_PER_CATEGORY = 2  # Select 2 utterances per category per round

    def __init__(
        self,
        model_config=None,
        prompt=None,
        increment_per_category=None,
    ):
        """Initialize the iterative selector with configurable parameters."""
        # Initialize parent ChatCompletionLLMApplier with all required parameters
        super().__init__(
            model_config=model_config or self.DEFAULT_MODEL_CONFIG,
            prompt=prompt or self.DEFAULT_PROMPT,
            threads=self.DEFAULT_THREADS,
            retries=self.DEFAULT_RETRIES,
        )

        # Set up selection parameters
        self.increment_per_category = (
            increment_per_category
            if increment_per_category is not None
            else self.DEFAULT_INCREMENT_PER_CATEGORY
        )

    def select_utterances_iteratively(
        self,
        data: Union[List, Dict],
        target_count: int = 2000,
        use_llm_analysis: bool = True,
        output_file: Optional[str] = None,
        resume: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Select utterances using simple iterative approach with LLM analysis.

        Args:
            data: Input data (list of utterances or dict with segments)
            target_count: Target number of utterances to select
            use_llm_analysis: Whether to use LLM for complexity/diversity analysis
            output_file: Path to save results (also used for resume state)
            resume: Whether to attempt resuming from existing output file
            verbose: Whether to show detailed category info

        Returns:
            Dictionary with selected utterances and selection metadata
        """
        # Try to resume from existing output file if requested
        if resume and output_file and os.path.exists(output_file):
            logger.info(f"Attempting to resume from existing output: {output_file}")
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)

                # Simple calculation from existing data - no complex metadata needed
                existing_utterances = existing_data.get("selected_utterances", [])
                existing_selected = len(existing_utterances)

                # Case 1: Already reached or exceeded target
                if existing_selected >= target_count:
                    logger.info(
                        f"🎯 Already completed: {existing_selected}/{target_count}"
                    )
                    return {
                        "selected_utterances": existing_utterances[:target_count],
                        "total_selected": min(existing_selected, target_count),
                        "target_count": target_count,
                        "rounds_completed": max(
                            1, existing_selected // 628
                        ),  # Estimate rounds from utterance count
                    }

                # Case 2: Partial completion - resume with existing data
                elif existing_selected > 0:
                    logger.info(
                        f"📂 Found {existing_selected} existing selections, "
                        f"need {target_count - existing_selected} more"
                    )
                    # Resume with existing data
                    return self._resume_from_existing(
                        existing_data, data, target_count, output_file, verbose
                    )

                # Case 3: Empty file - start fresh
                else:
                    logger.info("Empty output file - starting fresh")
            except Exception as e:
                logger.warning(f"Resume failed: {e}, starting fresh")

        # Organize data by categories
        category_data = self._organize_by_categories(data)

        # Sort categories by popularity (utterance count)
        sorted_categories = sorted(
            category_data.items(), key=lambda x: len(x[1]), reverse=True
        )

        logger.info(f"Found {len(sorted_categories)} categories")
        if verbose:
            logger.info(
                f"Category sizes: {[(cat, len(utts)) for cat, utts in sorted_categories[:5]]}"
            )

        # Calculate rounds needed - simple and accurate estimation
        total_utterances = sum(
            len(utterances) for _, utterances in category_data.items()
        )
        utterances_per_round = len(category_data) * self.increment_per_category
        estimated_rounds = (
            target_count + utterances_per_round - 1
        ) // utterances_per_round  # Ceiling division

        logger.info(f"Target: {target_count}, Total available: {total_utterances}")
        logger.info(
            f"Categories: {len(category_data)}, Max per round: {utterances_per_round} ({len(category_data)} categories × {self.increment_per_category} each)"
        )
        logger.info(f"Estimated rounds: {estimated_rounds}")

        # Initialize selection tracking
        selected_utterances = []
        category_selected_counts = defaultdict(int)
        round_selections = []

        # Iterative selection
        for round_num in range(1, estimated_rounds + 1):
            if len(selected_utterances) >= target_count:
                logger.info(f"Target reached at round {round_num-1}")
                break

            logger.info(f"Starting round {round_num}/{estimated_rounds}")

            round_selected = []
            remaining_target = target_count - len(selected_utterances)

            # Select from each category in order of popularity
            for category_key, category_utterances in sorted_categories:
                if len(selected_utterances) >= target_count:
                    break

                # Determine how many to select from this category this round
                already_selected = category_selected_counts[category_key]
                available = len(category_utterances) - already_selected

                if available <= 0:
                    continue  # Category exhausted

                # Select up to increment_per_category based on availability and remaining target
                to_select = min(
                    self.increment_per_category, available, remaining_target
                )

                if to_select <= 0:
                    continue

                # Select utterances using LLM guidance or simple selection
                if use_llm_analysis and len(selected_utterances) > 0:
                    category_selections = self._llm_guided_selection(
                        category_key,
                        category_utterances[
                            already_selected : already_selected + min(10, available)
                        ],
                        selected_utterances,
                        to_select,
                    )
                else:
                    # Simple selection - take next utterances
                    category_selections = category_utterances[
                        already_selected : already_selected + to_select
                    ]

                # Add to selections
                for utterance in category_selections:
                    utterance["selected_round"] = round_num
                    utterance["selected_category"] = category_key
                    selected_utterances.append(utterance)
                    round_selected.append(utterance)
                    category_selected_counts[category_key] += 1

                logger.info(
                    f"  {category_key}: selected {len(category_selections)}, total from category: {category_selected_counts[category_key]}"
                )

                remaining_target = target_count - len(selected_utterances)

            round_selections.append(
                {
                    "round": round_num,
                    "selected_count": len(round_selected),
                    "total_selected": len(selected_utterances),
                    "selections": round_selected,
                    "category_breakdown": dict(category_selected_counts),
                    "remaining_target": remaining_target,
                }
            )

            logger.info(
                f"Round {round_num} complete: {len(round_selected)} selected, total: {len(selected_utterances)}"
            )

            # Save results after each round for resume capability
            if output_file:
                self._save_results_to_file(
                    output_file,
                    selected_utterances,
                    category_selected_counts,
                    round_selections,
                    target_count,
                    round_num,
                )

            if len(round_selected) == 0:
                logger.warning(f"⚠️  EARLY TERMINATION - Round {round_num}")
                logger.warning(f"   Target: {target_count} utterances")
                logger.warning(
                    f"   Actually selected: {len(selected_utterances)} utterances"
                )
                logger.warning(
                    f"   Shortfall: {target_count - len(selected_utterances)} utterances"
                )
                logger.warning(
                    f"   Reason: No utterances could be selected in this round"
                )
                logger.warning(
                    f"   This typically means all categories have been exhausted"
                )
                break

        # Prepare results
        results = {
            "selected_utterances": selected_utterances,
            "total_selected": len(selected_utterances),
            "target_count": target_count,
            "rounds_completed": len(round_selections),
            "category_distribution": dict(category_selected_counts),
            "round_selections": round_selections,
            "selection_summary": {
                "categories_used": len(
                    [c for c in category_selected_counts.values() if c > 0]
                ),
                "avg_per_category": sum(category_selected_counts.values())
                / max(1, len([c for c in category_selected_counts.values() if c > 0])),
                "max_from_single_category": (
                    max(category_selected_counts.values())
                    if category_selected_counts
                    else 0
                ),
                "min_from_used_category": (
                    min([c for c in category_selected_counts.values() if c > 0])
                    if category_selected_counts
                    else 0
                ),
            },
        }

        logger.info(
            f"Selection complete: {results['total_selected']} utterances selected"
        )

        # Check if target was reached
        if results["total_selected"] < target_count:
            shortfall = target_count - results["total_selected"]
            completion_rate = (results["total_selected"] / target_count) * 100
            logger.warning(f"🎯 TARGET NOT REACHED:")
            logger.warning(f"   Requested: {target_count} utterances")
            logger.warning(f"   Selected: {results['total_selected']} utterances")
            logger.warning(f"   Shortfall: {shortfall} utterances")
            logger.warning(f"   Completion: {completion_rate:.1f}%")
        else:
            logger.info(
                f"🎯 TARGET REACHED: {results['total_selected']}/{target_count}"
            )

        # Save final results
        if output_file:
            self._save_results_to_file(
                output_file,
                selected_utterances,
                category_selected_counts,
                round_selections,
                target_count,
                len(round_selections),
            )

        return results

    def _organize_by_categories(self, data: Union[List, Dict]) -> Dict[str, List[Dict]]:
        """Organize data by categories (popularity-based)."""
        category_data = defaultdict(list)

        if isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
            # Check if this is pre-limited data with proper category keys (containing |)
            sample_key = next(iter(data.keys()))
            if "|" in sample_key:
                # Direct dictionary format - each key is already a category
                logger.info(
                    f"Using direct dictionary format with {len(data)} categories"
                )
                return dict(data)
            else:
                # Original format - data is dict with segments, need to organize by segment|switching_class
                for segment_name, utterances in data.items():
                    # Group by switching_class within each segment
                    switching_class_groups = defaultdict(list)
                    for utterance in utterances:
                        switching_class = utterance.get("switching_class", "unknown")
                        switching_class_groups[switching_class].append(utterance)

                    # Create categories (segment + switching_class combination)
                    for (
                        switching_class,
                        category_utterances,
                    ) in switching_class_groups.items():
                        category_key = f"{segment_name}|{switching_class}"
                        category_data[category_key] = category_utterances

        elif isinstance(data, list):
            # Data is a list of utterance objects
            for utterance in data:
                if isinstance(utterance, dict) and "utterance" in utterance:
                    classification = utterance.get("classification", "general")
                    category_key = f"general|{classification}"
                    category_data[category_key].append(utterance)

        return dict(category_data)

    @with_retries
    def _llm_guided_selection(
        self,
        category_key: str,
        candidates: List[Dict],
        selected_utterances: List[Dict],
        to_select: int,
    ) -> List[Dict]:
        """
        Use LLM to guide selection of utterances from candidates.

        Args:
            category_key: Current category being processed
            candidates: List of candidate utterances to select from
            selected_utterances: Previously selected utterances for context
            to_select: Number of utterances to select

        Returns:
            List of selected utterances
        """
        if not candidates:
            return []

        try:
            # Get reference examples (limited to prevent token overflow)
            reference_examples = self._get_reference_examples(selected_utterances)

            # Format candidates for LLM
            candidates_text = []
            for i, candidate in enumerate(candidates):
                text = candidate.get("utterance", candidate.get("text", ""))
                candidates_text.append(f"{i+1}. {text}")

            # Create prompt
            prompt_data = {
                "category": category_key,
                "to_select": to_select,
                "candidates": "\n".join(candidates_text),
                "reference_examples": (
                    "\n".join(reference_examples)
                    if reference_examples
                    else "None selected yet"
                ),
            }

            # Format the prompt with variables
            formatted_messages = prompts.formatting.render_messages(
                self.prompt, prompt_data
            )

            # Get LLM response
            completion = self.llmapi.chat_completion(
                self.model_config, formatted_messages
            )

            # Extract response
            response = completion["choices"][0]["message"]["content"].strip()

            # Parse response to get selected indices
            selected_indices = self._parse_llm_response(response)

            # Return selected utterances
            selected = []
            for idx in selected_indices:
                if 0 <= idx < len(candidates):
                    selected.append(candidates[idx])
                if len(selected) >= to_select:
                    break

            # If LLM selection failed, fall back to simple selection
            if not selected:
                logger.warning(
                    f"LLM selection failed for {category_key}, using simple selection"
                )
                selected = candidates[:to_select]

            return selected

        except Exception as e:
            logger.warning(
                f"LLM guidance failed for {category_key}: {e}, using simple selection"
            )
            return candidates[:to_select]

    def _get_reference_examples(
        self, selected_utterances: List[Dict], max_examples: int = 5
    ) -> List[str]:
        """Get reference examples from previously selected utterances."""
        if not selected_utterances:
            return []

        # Get a diverse sample of recent selections
        sample_size = min(max_examples, len(selected_utterances))
        if len(selected_utterances) <= max_examples:
            sample = selected_utterances
        else:
            # Take recent selections for diversity
            sample = selected_utterances[-sample_size:]

        examples = []
        for utterance in sample:
            text = utterance.get("utterance", utterance.get("text", ""))
            category = utterance.get("selected_category", "unknown")
            examples.append(f"- [{category}] {text}")

        return examples

    def _parse_llm_response(self, response: str) -> List[int]:
        """Parse LLM response to extract selected indices."""
        selected_indices = []
        try:
            # Look for numbers in the response
            import re

            numbers = re.findall(r"\b(\d+)\b", response)
            for num_str in numbers:
                idx = int(num_str) - 1  # Convert to 0-based index
                if idx not in selected_indices:
                    selected_indices.append(idx)
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")

        return selected_indices

    def _save_results_to_file(
        self,
        output_file: str,
        selected_utterances: List[Dict],
        category_selected_counts: Dict[str, int],
        round_selections: List[Dict],
        target_count: int,
        rounds_completed: int,
    ):
        """Save results to output file with metadata for resume capability."""
        try:
            # Minimal output format - only essential resuming info
            results = {
                "selected_utterances": selected_utterances,
                # Only store timestamp for basic tracking
                "timestamp": str(__import__("datetime").datetime.now()),
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved to {output_file} ({len(selected_utterances)} selected)")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def _resume_from_existing(
        self,
        existing_data: Dict,
        input_data: Union[List, Dict],
        target_count: int,
        output_file: str,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Resume selection from existing data, continuing from where we left off."""

        # Extract existing state
        existing_utterances = existing_data.get("selected_utterances", [])
        existing_metadata = existing_data.get("metadata", {})
        existing_rounds = existing_data.get("round_selections", [])
        existing_category_dist = existing_data.get("category_distribution", {})

        existing_selected = len(existing_utterances)
        remaining_needed = target_count - existing_selected
        completed_rounds = len(existing_rounds)

        # Organize input data by categories
        category_data = self._organize_by_categories(input_data)
        sorted_categories = sorted(
            category_data.items(), key=lambda x: len(x[1]), reverse=True
        )

        # Calculate remaining rounds needed
        utterances_per_round = len(category_data) * self.increment_per_category
        remaining_rounds = (
            remaining_needed + utterances_per_round - 1
        ) // utterances_per_round

        logger.info(f"Found {len(sorted_categories)} categories")
        logger.info(f"Target: {target_count}, Remaining needed: {remaining_needed}")
        logger.info(
            f"Categories: {len(category_data)}, Max per round: {utterances_per_round}"
        )
        logger.info(
            f"Completed rounds: {completed_rounds}, Additional rounds needed: {remaining_rounds}"
        )

        # Initialize tracking with existing data
        selected_utterances = existing_utterances.copy()
        category_selected_counts = defaultdict(int)

        # Rebuild category counts from existing data
        for utterance in existing_utterances:
            category_key = f"{utterance.get('category', 'unknown')}|{utterance.get('segment', 'unknown')}"
            category_selected_counts[category_key] += 1

        # Continue selection from next round
        round_selections = existing_rounds.copy()

        # Continue selection until target is reached (not limited by estimated rounds)
        round_num = completed_rounds + 1

        while len(selected_utterances) < target_count:
            logger.info(
                f"Starting round {round_num} (continuing until {target_count} reached)"
            )
            round_selected = []

            # Same selection logic as original method...
            for category_key, category_utterances in sorted_categories:
                if len(selected_utterances) >= target_count:
                    break

                already_selected = category_selected_counts[category_key]
                available = len(category_utterances) - already_selected
                remaining_target = target_count - len(selected_utterances)

                if available <= 0:
                    continue

                to_select = min(
                    self.increment_per_category, available, remaining_target
                )

                if to_select > 0:
                    # Use LLM-guided selection like main method
                    if (
                        len(selected_utterances) > 0
                    ):  # We have existing utterances for context
                        category_selections = self._llm_guided_selection(
                            category_key,
                            category_utterances[
                                already_selected : already_selected + min(10, available)
                            ],
                            selected_utterances,
                            to_select,
                        )
                    else:
                        # Simple selection for first selections
                        category_selections = category_utterances[
                            already_selected : already_selected + to_select
                        ]

                    # Add selection metadata
                    for utterance in category_selections:
                        utterance["selected_round"] = round_num
                        utterance["selected_category"] = category_key
                        selected_utterances.append(utterance)
                        round_selected.append(utterance)
                        category_selected_counts[category_key] += 1

                    logger.info(
                        f"  {category_key}: selected {len(category_selections)}, total from category: {category_selected_counts[category_key]}"
                    )

            if round_selected:
                round_selections.append(
                    {
                        "round": round_num,
                        "selected": round_selected,
                        "total_selected": len(round_selected),
                    }
                )
                logger.info(
                    f"Round {round_num} complete: {len(round_selected)} selected, total: {len(selected_utterances)}"
                )
                # Save after each round
                if output_file:
                    self._save_results_to_file(
                        output_file,
                        selected_utterances,
                        category_selected_counts,
                        [],
                        target_count,
                        round_num,
                    )

                # Increment round for next iteration
                round_num += 1
            else:
                logger.warning("No utterances selected in this round - stopping")
                break

        # Build final results
        total_rounds_completed = (
            completed_rounds + (round_num - completed_rounds - 1)
            if round_num > completed_rounds + 1
            else completed_rounds + 1
        )
        results = {
            "selected_utterances": selected_utterances,
            "total_selected": len(selected_utterances),
            "target_count": target_count,
            "rounds_completed": total_rounds_completed,  # Total rounds including previous + new
        }

        # Save results
        self._save_results_to_file(
            output_file,
            selected_utterances,
            category_selected_counts,
            round_selections,
            target_count,
            len(round_selections),
        )

        if len(selected_utterances) >= target_count:
            logger.info(f"🎯 TARGET REACHED: {len(selected_utterances)}/{target_count}")
        else:
            logger.warning(
                f"🎯 TARGET NOT REACHED: {len(selected_utterances)}/{target_count}"
            )

        return results
