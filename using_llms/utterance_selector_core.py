"""
Iterative Utterance Selector with LLM analysis and simple resume functionality.
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
from llms import ChatCompletionLLMApplier, ApplicationModes, prompts, with_retries

logger = logging.getLogger(__name__)


class IterativeUtteranceSelector(ChatCompletionLLMApplier):
    """
    Iterative utterance selector with LLM analysis and resume.
    """

    # Model configuration - easy to adjust at class level
    DEFAULT_MODEL_CONFIG = {
        "model": "dev-gpt-41-longco-2025-04-14",
        "temperature": 0.1,  # Low temperature for consistency
        "max_tokens": 2000,  # Sufficient for selection analysis
    }

    # Load prompt that includes enterprise search context
    DEFAULT_PROMPT = prompts.get("utterance_selector", "0.1.0")

    # Required parent class attributes
    DEFAULT_THREADS = 1  # Single-threaded for simplicity
    DEFAULT_RETRIES = 3  # Retry failed LLM calls
    APPLICATION_MODE = ApplicationModes.PerItem  # Process items individually

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
                        "rounds_completed": 0,  # No new rounds were completed in this run
                    }

                # Case 2: Partial completion - resume with existing data
                elif existing_selected > 0:
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
        selected_utterance_texts = set()  # Track utterance texts to avoid duplicates

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

                # Get candidates for this category
                candidate_start_idx = already_selected
                candidate_end_idx = (
                    candidate_start_idx + min(10, available)
                    if use_llm_analysis
                    else candidate_start_idx + to_select
                )
                candidates = category_utterances[candidate_start_idx:candidate_end_idx]

                # Filter out any candidates that have already been selected (avoid duplicates)
                filtered_candidates = []
                for candidate in candidates:
                    candidate_text = candidate.get(
                        "utterance", candidate.get("text", "")
                    )
                    if (
                        candidate_text
                        and candidate_text not in selected_utterance_texts
                    ):
                        filtered_candidates.append(candidate)

                if not filtered_candidates:
                    logger.info(
                        f"  {category_key}: No new candidates available (all may be duplicates)"
                    )
                    continue

                # Select utterances using LLM guidance or simple selection
                if use_llm_analysis and len(selected_utterances) > 0:
                    try:
                        category_selections = self._llm_guided_selection(
                            category_key,
                            filtered_candidates,
                            selected_utterances,
                            to_select,
                        )
                    except Exception as e:
                        # LLM failed after all retries - save progress and exit gracefully
                        logger.error(f"❌ LLM FAILURE: {e}")
                        logger.error(f"❌ Category: {category_key}")
                        logger.error(f"❌ Failed after {self.retries} retry attempts")

                        # Save current progress to output file
                        if output_file:
                            self._save_results_to_file(
                                output_file,
                                selected_utterances,
                                category_selected_counts,
                                round_selections,
                                target_count,
                                round_num,
                            )
                            logger.info(f"💾 Progress saved to {output_file}")
                            logger.info(
                                f"📊 Current progress: {len(selected_utterances)}/{target_count} utterances selected"
                            )

                        # Create results for partial completion
                        results = {
                            "selected_utterances": selected_utterances,
                            "total_selected": len(selected_utterances),
                            "target_count": target_count,
                            "rounds_completed": len(round_selections),
                            "status": "llm_failure",
                            "failure_info": {
                                "failed_category": category_key,
                                "error": str(e),
                                "retries_attempted": self.retries,
                            },
                        }

                        # Exit with detailed information
                        logger.error("\n" + "=" * 60)
                        logger.error("🚨 LLM QUALITY SELECTION FAILED")
                        logger.error("=" * 60)
                        logger.error(
                            f"Reason: LLM failed after {self.retries} retries for category '{category_key}'"
                        )
                        logger.error(f"Error: {str(e)}")
                        logger.error(f"\n📈 PROGRESS SUMMARY:")
                        logger.error(
                            f"   • Selected: {len(selected_utterances)}/{target_count} utterances"
                        )
                        logger.error(f"   • Completed rounds: {round_num}")
                        logger.error(f"   • Progress saved to: {output_file}")
                        logger.error(f"\n🔧 RECOMMENDED ACTIONS:")
                        logger.error(
                            f"   1. Check LLM API authentication and connectivity"
                        )
                        logger.error(
                            f"   2. Verify prompt template exists: prompts/utterance_selector/0.1.0/"
                        )
                        logger.error(
                            f"   3. Review LLM model configuration: {self.model_config}"
                        )
                        logger.error(
                            f"   4. Re-run with same parameters to resume from saved progress"
                        )
                        logger.error(f"\n🔄 RESUME COMMAND:")
                        logger.error(
                            f"   Use the same command - the system will automatically resume from {output_file}"
                        )
                        logger.error("=" * 60)

                        return results
                else:
                    # First selections: take first utterances sequentially (no LLM context available yet)
                    category_selections = filtered_candidates[:to_select]

                # Add to selections
                for utterance in category_selections:
                    utterance["selected_round"] = round_num
                    utterance["selected_category"] = category_key
                    selected_utterances.append(utterance)
                    round_selected.append(utterance)
                    category_selected_counts[category_key] += 1

                    # Add to duplicate avoidance set
                    utterance_text = utterance.get(
                        "utterance", utterance.get("text", "")
                    )
                    if utterance_text:
                        selected_utterance_texts.add(utterance_text)

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

        return results

    def _organize_by_categories(self, data: Union[List, Dict]) -> Dict[str, List[Dict]]:
        """Organize data by categories (popularity-based)."""
        category_data = defaultdict(list)

        if isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
            # Check if this is pre-limited data with proper category keys (containing |)
            sample_key = next(iter(data.keys()))
            if "|" in sample_key:
                # Direct dictionary format - each key is already a category
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

        Raises:
            Exception: When LLM fails after all retries (no fallback to simple selection)
        """
        if not candidates:
            return []

        # Get reference examples (limited to prevent token overflow)
        reference_examples = self._get_reference_examples(selected_utterances)

        # Format candidates for LLM
        candidates_text = []
        for i, candidate in enumerate(candidates):
            text = candidate.get("utterance", candidate.get("text", ""))
            candidates_text.append(f"{i+1}. {text}")

        # Validate candidates
        if not candidates_text or not any(candidates_text):
            logger.error(f"No valid candidates found for category {category_key}")
            raise Exception("No valid candidate utterances found")

        # Create prompt
        prompt_data = {
            "category_name": category_key,  # Fixed: template expects category_name
            "target_count": to_select,  # Fixed: template expects target_count
            "candidates_list": "\n".join(
                candidates_text
            ),  # Fixed: template expects candidates_list
            "selected_examples": (  # Fixed: template expects selected_examples
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
        completion = self.llmapi.chat_completion(self.model_config, formatted_messages)

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

        # If LLM response parsing failed, raise an exception (no fallback)
        if not selected:
            raise Exception(
                f"LLM response parsing failed - no valid selections found in response: {response[:200]}..."
            )

        return selected

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
        """Save results to output file with minimal info for resume capability."""
        try:
            # Minimal output format - only essential data for resuming
            results = {
                "selected_utterances": selected_utterances,
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

        # Calculate completed rounds from the highest selected_round in existing utterances
        completed_rounds = 0
        if existing_utterances:
            max_round = max(
                utterance.get("selected_round", 1) for utterance in existing_utterances
            )
            completed_rounds = max_round
        else:
            logger.info("No existing utterances found")

        # Organize input data by categories
        category_data = self._organize_by_categories(input_data)
        sorted_categories = sorted(
            category_data.items(), key=lambda x: len(x[1]), reverse=True
        )

        # Consolidated status message
        if existing_utterances:
            logger.info(
                f"📂 Found {existing_selected} utterances already selected in {completed_rounds} rounds with {len(category_data)} categories"
            )

        # Calculate remaining rounds needed
        utterances_per_round = len(category_data) * self.increment_per_category
        remaining_rounds = (
            remaining_needed + utterances_per_round - 1
        ) // utterances_per_round

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

        # Rebuild category counts from existing data using the correct selected_category field
        for utterance in existing_utterances:
            category_key = utterance.get("selected_category", "unknown")
            category_selected_counts[category_key] += 1

        # Build set of already selected utterance texts to avoid duplicates
        selected_utterance_texts = set()
        for utterance in existing_utterances:
            utterance_text = utterance.get("utterance", utterance.get("text", ""))
            if utterance_text:
                selected_utterance_texts.add(utterance_text)

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
                    # Get candidates starting from where we left off for this category
                    candidate_start_idx = already_selected
                    candidate_end_idx = candidate_start_idx + min(10, available)
                    candidates = category_utterances[
                        candidate_start_idx:candidate_end_idx
                    ]

                    # Filter out any candidates that have already been selected (avoid duplicates)
                    filtered_candidates = []
                    for candidate in candidates:
                        candidate_text = candidate.get(
                            "utterance", candidate.get("text", "")
                        )
                        if (
                            candidate_text
                            and candidate_text not in selected_utterance_texts
                        ):
                            filtered_candidates.append(candidate)

                    if not filtered_candidates:
                        logger.info(
                            f"  {category_key}: No new candidates available (all may be duplicates)"
                        )
                        continue

                    # Use LLM-guided selection (we always have existing utterances during resume)
                    try:
                        category_selections = self._llm_guided_selection(
                            category_key,
                            filtered_candidates,
                            selected_utterances,
                            to_select,
                        )
                    except Exception as e:
                        # LLM failed during resume - save progress and exit gracefully
                        logger.error(f"❌ LLM FAILURE DURING RESUME: {e}")
                        logger.error(f"❌ Category: {category_key}")
                        logger.error(f"❌ Failed after {self.retries} retry attempts")

                        # Save current progress to output file
                        if output_file:
                            self._save_results_to_file(
                                output_file,
                                selected_utterances,
                                category_selected_counts,
                                round_selections,
                                target_count,
                                round_num,
                            )
                            logger.info(f"💾 Resume progress saved to {output_file}")
                            logger.info(
                                f"📊 Current progress: {len(selected_utterances)}/{target_count} utterances selected"
                            )

                        # Create results for partial completion during resume
                        results = {
                            "selected_utterances": selected_utterances,
                            "total_selected": len(selected_utterances),
                            "target_count": target_count,
                            "rounds_completed": len(round_selections),
                            "status": "llm_failure_during_resume",
                            "failure_info": {
                                "failed_category": category_key,
                                "error": str(e),
                                "retries_attempted": self.retries,
                            },
                        }

                        # Exit with detailed information
                        logger.error("\n" + "=" * 60)
                        logger.error("🚨 LLM QUALITY SELECTION FAILED DURING RESUME")
                        logger.error("=" * 60)
                        logger.error(
                            f"Reason: LLM failed after {self.retries} retries for category '{category_key}'"
                        )
                        logger.error(f"Error: {str(e)}")
                        logger.error(f"\n📈 PROGRESS SUMMARY:")
                        logger.error(
                            f"   • Selected: {len(selected_utterances)}/{target_count} utterances"
                        )
                        logger.error(f"   • Completed rounds: {round_num}")
                        logger.error(f"   • Progress saved to: {output_file}")
                        logger.error(f"\n🔧 RECOMMENDED ACTIONS:")
                        logger.error(
                            f"   1. Check LLM API authentication and connectivity"
                        )
                        logger.error(
                            f"   2. Verify prompt template exists: prompts/utterance_selector/0.1.0/"
                        )
                        logger.error(
                            f"   3. Review LLM model configuration: {self.model_config}"
                        )
                        logger.error(
                            f"   4. Re-run with same parameters to resume from saved progress"
                        )
                        logger.error("=" * 60)

                        return results

                    # Add selection metadata
                    for utterance in category_selections:
                        utterance["selected_round"] = round_num
                        utterance["selected_category"] = category_key
                        selected_utterances.append(utterance)
                        round_selected.append(utterance)
                        category_selected_counts[category_key] += 1

                        # Add to duplicate avoidance set
                        utterance_text = utterance.get(
                            "utterance", utterance.get("text", "")
                        )
                        if utterance_text:
                            selected_utterance_texts.add(utterance_text)

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

        if len(selected_utterances) >= target_count:
            logger.info(f"🎯 TARGET REACHED: {len(selected_utterances)}/{target_count}")
        else:
            logger.warning(
                f"🎯 TARGET NOT REACHED: {len(selected_utterances)}/{target_count}"
            )

        return results
