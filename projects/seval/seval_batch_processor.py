#!/usr/bin/env python3
"""
SEVAL Batch Processor - Modular Architecture

Provides batch processing utilities for SEVAL data analysis with
extensible modular design.

Available Commands:
    extract_conversations           - Extract conversation details from SEVAL
    merge_citescg_scores            - Merge conversations with CiteDCG scores
    process_seval_job_multihop_citedcg - End-to-end CiteDCG processing
    process_seval_job_with_statistics_plots - Calculate statistics for
                                             multiple top-k and generate plots

Usage Examples:
    # Process with multiple top-k values, calculate statistics, and plot
    python seval_batch_processor.py process_seval_job_with_statistics_plots \\
        --job_id=130949 --top_k_list=1,3,5
    
    # Complete multi-hop CiteDCG processing (single top-k)
    python seval_batch_processor.py process_seval_job_multihop_citedcg \\
        --job_id=130949
    
    # Extract conversations only
    python seval_batch_processor.py extract_conversations \\
        --input_dir "seval_data/130949_scraping_raw_data_output" \\
        --output_dir "results/conversation_details" \\
        --experiment control --threads 8

    # Merge with CiteDCG scores only
    python seval_batch_processor.py merge_citescg_scores \\
        --conv_dir "results/conversation_details" \\
        --citedcg_control_file "results/130949_citedcg_scores_control.json" \\
        --output_dir "results/merged"

Workflow:
    1. Extract CiteDCG scores from metrics
    2. Extract conversation details from raw data
    3. Merge conversations with CiteDCG scores (once, no statistics)
    4. Calculate statistics for different top-k values
    5. Generate comparison plots
"""

import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import fire

# Configure logging with default WARNING level (quiet by default)
# IMPORTANT: Must be done BEFORE importing seval_analysis_toolkit
# so that toolkit inherits this configuration instead of setting its own
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import after logging configuration
from merge_seval_results import (
    generate_plot_statistics_from_utterance_details,
    merge_citedcg_and_calculate_stats)
from seval_analysis_toolkit import SEVALAnalysisToolkit
from seval_plotting import (generate_comparison_plots,
                            generate_paired_utterances_plot,
                            generate_statistics_plots)

# ==========================================================================
# Base Classes
# ==========================================================================

class BaseProcessor:
    """Base class providing common functionality for all processors."""
    
    def __init__(self):
        self.results: List[Any] = []
        self.errors: List[Dict[str, str]] = []
    
    def _setup_logging(self, verbose: bool = False):
        """Configure logging level.
        
        Args:
            verbose: If True, show DEBUG level. If False, show WARNING level only.
        """
        # Handle string-to-boolean conversion (Fire may pass "false" as string)
        if isinstance(verbose, str):
            verbose = verbose.lower() in ('true', '1', 'yes')
        
        level = logging.DEBUG if verbose else logging.WARNING
        logging.getLogger().setLevel(level)
    
    def _clean_directory(
        self,
        directory: Path,
        description: str = "directory",
        silent: bool = False
    ):
        """Clean all files in a directory.
        
        Args:
            directory: Path to directory to clean
            description: Human-readable description for logging
            silent: If True, don't print cleaning messages
        """
        if directory.exists():
            file_count = len(list(directory.rglob('*')))
            if file_count > 0:
                if not silent:
                    print(f"  Cleaning {description}: {directory}")
                    print(f"  → Removing {file_count} items...")
                shutil.rmtree(directory)
                directory.mkdir(parents=True, exist_ok=True)
                if not silent:
                    print(f"  ✓ Cleaned")
            # Skip message if directory is already empty
    
    def _validate_path(
        self,
        path: Path,
        must_exist: bool = True,
        create: bool = False
    ) -> bool:
        """Validate and optionally create directory."""
        if path.exists():
            return True
        
        if create:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")
            return True
        
        if must_exist:
            logger.error(f"Path not found: {path}")
            return False
        
        return True
    
    def _parallel_process(
        self,
        items: List[Any],
        process_func: Callable,
        max_workers: int = 8,
        verbose: bool = False,
        description: str = "items"
    ) -> List[Any]:
        """Generic parallel processing with progress tracking."""
        results = []
        total = len(items)
        
        logger.info(f"Processing {total} {description} with {max_workers} threads...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {
                executor.submit(process_func, item): item
                for item in items
            }
            
            completed = 0
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                completed += 1
                
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                    
                    if completed % 10 == 0 or verbose:
                        logger.info(
                            f"Progress: {completed}/{total} {description}"
                        )
                
                except Exception as e:
                    logger.error(f"Error processing {item}: {e}")
                    self.errors.append({
                        "item": str(item),
                        "error": str(e)
                    })
        
        logger.info(f"Completed: {len(results)}/{total} successful")
        return results


# ==========================================================================
# Conversation Extraction Module
# ==========================================================================

class ConversationExtractor(BaseProcessor):
    """
    Extract conversation details from SEVAL files.
    
    IMPORTANT - "Non-empty hop" definition in this phase:
    A hop is considered "non-empty" if it EXISTS in the data structure,
    regardless of whether it has invocations (search attempts) or results.
    This counts hop data structures that may be empty placeholders.
    
    See MergeCiteDCGProcessor for a different definition used in merge phase.
    """
    
    def __init__(self):
        super().__init__()
        self.toolkit = SEVALAnalysisToolkit()
        self.multi_turn_or_hop_conversations: List[Dict] = []
        self.input_dir = None
        self.output_dir = None
        self.verbose = False
    
    def vprint(self, *args, **kwargs):
        """Print only when verbose=True (for detailed logs)"""
        if self.verbose:
            print(*args, **kwargs)
    
    def process(
        self,
        input_dir: str,
        output_dir: str,
        experiment: str = "both",
        threads: int = 8,
        verbose: bool = False,
        clean: bool = False
    ) -> Dict[str, Any]:
        """
        Extract conversation details from SEVAL files.
        
        Args:
            input_dir: Directory with SEVAL JSON files
            output_dir: Directory for extracted conversation details
            experiment: Experiment type: 'control', 'treatment', or 'both' (default: 'both')
            threads: Number of parallel threads
            verbose: Enable verbose logging
            clean: If True, delete and recreate output directory
            
        Returns:
            Summary statistics
        """
        self._setup_logging(verbose)
        
        self.verbose = verbose  # Store for use in helper methods
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not self._validate_path(input_path):
            return {}
        
        # Clean output directory if requested
        if clean and output_path.exists():
            import shutil
            logger.info(f"Cleaning output directory: {output_path}")
            shutil.rmtree(output_path)
        
        self._validate_path(output_path, create=True)
        
        # Find files based on experiment type
        files = []
        control_files = []
        treatment_files = []
        control_count = 0
        treatment_count = 0
        
        if experiment.lower() in ["control", "both"]:
            control_files = sorted(input_path.glob("control_*.json"))
            files.extend(control_files)
            control_count = len(control_files)
            print(f"  Found {control_count} control files")
        
        if experiment.lower() in ["treatment", "both"]:
            # Try both naming conventions for treatment files
            treatment_files = sorted(input_path.glob("treatment_*.json"))
            if len(treatment_files) == 0:
                # Try alternate naming convention
                treatment_files = sorted(input_path.glob("experiment_*.json"))
                if len(treatment_files) > 0:
                    self.vprint("  (Using 'experiment_*' pattern for treatment)")
            files.extend(treatment_files)
            treatment_count = len(treatment_files)
            print(f"  Found {treatment_count} treatment files")
        
        print("")
        
        if not files:
            logger.warning(f"No {experiment} files found in {input_dir}")
            return {}
        
        if self.verbose:
            self._print_header("BATCH CONVERSATION EXTRACTION")
            print(f"Input dir:        {input_dir}")
            print(f"Output dir:       {output_dir}")
            print(f"Experiment type:  {experiment}")
            print(f"Files to process: {len(files)}")
        
        # Store file lists for later separation
        self.control_files = set(str(f) for f in control_files)
        self.treatment_files = set(str(f) for f in treatment_files)
        
        # Process files
        process_func = lambda f: self._process_file(f, output_path, verbose)
        self.results = self._parallel_process(
            files,
            process_func,
            threads,
            verbose,
            "files"
        )
        
        # Generate summary
        self._generate_summary(output_path, experiment)
        
        self._print_header("PROCESSING COMPLETE")
    
    def _process_file(
        self,
        file_path: Path,
        output_dir: Path,
        verbose: bool
    ) -> Optional[Dict[str, Any]]:
        """Extract conversation details from a single file."""
        try:
            if verbose:
                logger.debug(f"Processing: {file_path.name}")
            
            output_file = output_dir / f"{file_path.stem}_conv_details.json"
            
            # Extract using toolkit
            self.toolkit.extract_conversation_details(
                input_file=str(file_path),
                output_file=str(output_file)
            )
            
            if not output_file.exists():
                logger.warning(f"No output for {file_path.name}")
                return None
            
            # Analyze extracted data
            with open(output_file, "r", encoding="utf-8") as f:
                conv_data = json.load(f)
            
            return self._analyze_conversation(
                conv_data,
                file_path.name,
                str(output_file),
                str(file_path)
            )
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path.name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            return None
    
    def _analyze_conversation(
        self,
        conv_data: Dict,
        filename: str,
        output_file: str,
        source_file: str = None
    ) -> Dict[str, Any]:
        """Analyze extracted conversation data."""
        metadata = conv_data.get("metadata", {})
        eval_results = conv_data.get("evaluation_data_results", {})
        turns = eval_results.get("turns", [])
        
        # Calculate hop statistics from actual turn data
        total_hops = 0
        nonempty_hops = 0
        hop_stats_per_turn = []
        
        for turn in turns:
            hops = turn.get("hops", [])
            turn_total_hops = len(hops)
            turn_nonempty_hops = 0
            
            # DEFINITION: "Non-empty hop" in extraction phase means the hop
            # data structure exists, even if it's just a placeholder with no
            # invocations or search results. We count every hop in the array.
            # This is different from the merge phase which requires invocations.
            for hop in hops:
                turn_nonempty_hops += 1
            
            total_hops += turn_total_hops
            nonempty_hops += turn_nonempty_hops
            
            hop_stats_per_turn.append({
                "total": turn_total_hops,
                "nonempty": turn_nonempty_hops
            })
        
        # For multi-turn conversations, determine which turns have hops
        multiturn_pattern = None
        if len(turns) > 1:
            turns_with_hops = [
                i + 1 for i, stats in enumerate(hop_stats_per_turn)
                if stats["nonempty"] > 0
            ]
            
            num_turns_with_hops = len(turns_with_hops)
            if num_turns_with_hops == 0:
                multiturn_pattern = "no_hops"
            elif num_turns_with_hops == 1:
                turn_num = turns_with_hops[0]
                if turn_num == len(turns):
                    multiturn_pattern = "last_turn_only"
                elif turn_num == 1:
                    multiturn_pattern = "first_turn_only"
                else:
                    multiturn_pattern = "middle_turn_only"
            else:
                multiturn_pattern = "multiple_turns"
        
        result = {
            "file": filename,
            "source_file": source_file,
            "conversation_id": metadata.get("conversation_id", ""),
            "query_text": metadata.get("query_text", ""),
            "turn_count": len(turns),
            "total_hops": total_hops,
            "nonempty_hops": nonempty_hops,
            "is_multi_turn": len(turns) > 1,
            "is_multi_hop": nonempty_hops > 1,
            "hop_stats_per_turn": hop_stats_per_turn,
            "multiturn_pattern": multiturn_pattern,
            "output_file": output_file,
        }
        
        # Track multi-turn OR multi-hop conversations for detailed reporting
        if result["is_multi_turn"] or result["is_multi_hop"]:
            self.multi_turn_or_hop_conversations.append(result)
        
        return result
    
    def _generate_summary(self, output_dir: Path, experiment: str = "both") -> Dict[str, Any]:
        """Generate and save summary report."""
        stats = self._calculate_statistics(experiment)
        self._print_statistics(stats, self.input_dir, self.output_dir)
        self._save_reports(stats, output_dir)
        return stats
    
    def _calculate_statistics(self, experiment: str = "both") -> Dict[str, Any]:
        """Calculate processing statistics, separated by experiment type."""
        
        def calc_stats_for_results(results):
            """Helper to calculate stats for a subset of results."""
            total = len(results)
            if total == 0:
                return None
            
            # Count utterances by turn count
            turn_dist = {}
            multi_turn_with_hops = 0
            
            # Multi-turn hop pattern tracking
            multiturn_verification = {
                'total_multi_turn': 0,
                'last_turn_has_hops': 0,
                'first_turn_has_hops': 0,
                'middle_turn_has_hops': 0,
                'multiple_turns_with_hops': 0,
                'pattern_by_turns': {}
            }
            
            for r in results:
                turns = r["turn_count"]
                nonempty_hops = r.get("nonempty_hops", 0)
                
                # Track turn distribution
                turn_dist[turns] = turn_dist.get(turns, 0) + 1
                
                # Count multi-turn utterances with hops
                if turns > 1 and nonempty_hops > 0:
                    multi_turn_with_hops += 1
                
                # Track multi-turn hop patterns
                if turns > 1:
                    multiturn_verification['total_multi_turn'] += 1
                    pattern = r.get("multiturn_pattern")
                    
                    if pattern == "last_turn_only":
                        multiturn_verification['last_turn_has_hops'] += 1
                    elif pattern == "first_turn_only":
                        multiturn_verification['first_turn_has_hops'] += 1
                    elif pattern == "middle_turn_only":
                        multiturn_verification['middle_turn_has_hops'] += 1
                    elif pattern == "multiple_turns":
                        multiturn_verification['multiple_turns_with_hops'] += 1
                    
                    # Track pattern by number of turns
                    turns_key = f"{turns}_turns"
                    if turns_key not in multiturn_verification['pattern_by_turns']:
                        multiturn_verification['pattern_by_turns'][turns_key] = {}
                    
                    if pattern:
                        pattern_counts = multiturn_verification['pattern_by_turns'][turns_key]
                        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            # Calculate totals
            multi_turn = sum(1 for r in results if r.get("turn_count", 0) > 1)
            total_hops = sum(r.get("total_hops", 0) for r in results)
            total_nonempty = sum(r.get("nonempty_hops", 0) for r in results)
            
            # Hop distributions
            total_hop_dist = {}
            nonempty_hop_dist = {}
            for r in results:
                total_hops_count = r.get("total_hops", 0)
                nonempty_hops_count = r.get("nonempty_hops", 0)
                
                total_hop_dist[total_hops_count] = total_hop_dist.get(total_hops_count, 0) + 1
                nonempty_hop_dist[nonempty_hops_count] = nonempty_hop_dist.get(nonempty_hops_count, 0) + 1
            
            return {
                "total_processed": total,
                "multi_turn": multi_turn,
                "single_turn": total - multi_turn,
                "multi_turn_with_hops": multi_turn_with_hops,
                "total_hops": total_hops,
                "total_nonempty_hops": total_nonempty,
                "avg_total_hops": total_hops / total if total > 0 else 0,
                "avg_nonempty_hops": total_nonempty / total if total > 0 else 0,
                "turn_distribution": turn_dist,
                "total_hop_distribution": total_hop_dist,
                "nonempty_hop_distribution": nonempty_hop_dist,
                "multiturn_verification": multiturn_verification,
            }
        
        # Separate results by experiment type
        control_results = [r for r in self.results if r.get("source_file") in self.control_files]
        treatment_results = [r for r in self.results if r.get("source_file") in self.treatment_files]
        
        # Calculate stats for each experiment type
        control_stats = calc_stats_for_results(control_results) if control_results else None
        treatment_stats = calc_stats_for_results(treatment_results) if treatment_results else None
        combined_stats = calc_stats_for_results(self.results)
        
        return {
            "experiment": experiment,
            "errors": len(self.errors),
            "combined": combined_stats,
            "control": control_stats,
            "treatment": treatment_stats,
        }
    
    def _print_statistics(self, stats: Dict[str, Any], input_dir: str = None, output_dir: str = None):
        """Print statistics to console (always displayed regardless of verbose)."""
        
        def print_stats_section(title: str, s: Dict[str, Any]):
            """Helper to print a statistics section."""
            if not s:
                return
            
            print(f"\n{title}:")
            print(f"  Total processed: {s['total_processed']}")
            
            # Utterance breakdown by # of turns
            print(f"  Utterance breakdown by # of turns:")
            print(f"    (Non-empty turn = turn with hop data structures, even if empty placeholders)")
            total = s['total_processed']
            for turns in sorted(s['turn_distribution'].keys()):
                count = s['turn_distribution'][turns]
                pct = (count / total * 100) if total > 0 else 0
                if turns == 0:
                    print(f"    {turns} turns (no data): {count} ({pct:.1f}%)")
                elif turns == 1:
                    print(f"    {turns} turn: {count} ({pct:.1f}%)")
                else:
                    print(f"    {turns} turns (1 non-empty: last turn has hops): {count} ({pct:.1f}%)")
            
            # Hop statistics
            print(f"  Aggregate Hop Statistics:")
            print(f"    Total hops: {s['total_hops']}")
            print(f"    Non-empty hops: {s['total_nonempty_hops']}")
            print(f"    Avg hops per conversation: {s['avg_total_hops']:.2f}")
            print(f"    Avg non-empty hops per conversation: {s['avg_nonempty_hops']:.2f}")
        
        # Always print basic summary header and error count
        print("=" * 80)
        print("SUMMARY REPORT")
        print("=" * 80)
        if input_dir:
            print(f"Input dir:  {input_dir}")
        if output_dir:
            print(f"Output dir: {output_dir}")
        print("-" * 80)
        print(f"Errors: {stats['errors']}")
        print("-" * 80)
        
        # Only print detailed statistics when verbose=True
        if not self.verbose:
            return
        
        # Print combined statistics
        if stats.get('combined'):
            print_stats_section("COMBINED (Control + Treatment)", stats['combined'])
        
        # Print control statistics
        if stats.get('control'):
            print_stats_section("CONTROL", stats['control'])
        
        # Print treatment statistics  
        if stats.get('treatment'):
            print_stats_section("TREATMENT", stats['treatment'])
    
    def _save_reports(self, stats: Dict[str, Any], output_dir: Path):
        """Save detailed reports."""
        # Multi-turn/multi-hop conversations TSV
        if self.multi_turn_or_hop_conversations:
            logger.info("-" * 80)
            logger.info(
                f"Multi-turn/multi-hop: {len(self.multi_turn_or_hop_conversations)}"
            )
            
            tsv_file = output_dir / "multi_turn_conversations.tsv"
            with open(tsv_file, "w", encoding="utf-8") as f:
                f.write(
                    "File\tConversation_ID\tTurn_Count\tTotal_Hops\t"
                    "Nonempty_Hops\tQuery_Text\tOutput_File\n"
                )
                for conv in sorted(
                    self.multi_turn_or_hop_conversations,
                    key=lambda x: (x["turn_count"], x["nonempty_hops"]),
                    reverse=True
                ):
                    f.write(
                        f"{conv['file']}\t{conv['conversation_id']}\t"
                        f"{conv['turn_count']}\t{conv['total_hops']}\t"
                        f"{conv['nonempty_hops']}\t{conv['query_text']}\t"
                        f"{conv['output_file']}\n"
                    )
            
            logger.info(f"Multi-turn report: {tsv_file}")
        
        # Full summary JSON
        summary_file = output_dir / "batch_extraction_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "statistics": stats,
                    "multi_turn_or_hop_conversations": self.multi_turn_or_hop_conversations,
                    "processing_errors": self.errors,
                    "all_results": self.results,
                },
                f,
                indent=2,
                ensure_ascii=False
            )
        
        logger.info(f"Full summary: {summary_file}")
    
    def _print_header(self, title: str):
        """Print formatted section header."""
        logger.info("=" * 80)
        logger.info(title)
        logger.info("=" * 80)


# ==========================================================================
# MergeCiteDCGProcessor - Merge conversations with CiteDCG scores
# ==========================================================================

class MergeCiteDCGProcessor(BaseProcessor):
    """
    Merge conversation details with CiteDCG scores.
    
    IMPORTANT - "Non-empty hop" definition in this phase:
    A hop is considered "non-empty" ONLY if it has invocations (meaning
    a search was actually attempted), not just an empty hop placeholder.
    This is stricter than the extraction phase definition.
    
    Example data patterns:
    - Hop with invocations: {"invocations": [{"queries": [...]}]} → non-empty
    - Hop without invocations: {"invocations": []} → EMPTY (not counted)
    - Multi-turn with no invocations: All retry turns have hop structures
      but no actual search attempts (92% of multi-turn conversations)
    
    See ConversationExtractor for the different definition used there.
    """
    
    def __init__(self):
        super().__init__()
        self.citedcg_control_file = None
        self.citedcg_treatment_file = None
        self.all_hop_scores = []  # Track all hop scores for distribution
        self.all_hop_top_k_scores = []  # Track top-k scores
        self.top_k = 5  # Default value
        self.no_scores_count = 0  # Track files with no scores
        self.utterances_not_in_dcg = 0  # Track utterances not in DCG data
        self.verbose = False  # Verbose flag for output control
    
    def process(
        self,
        conv_dir: str,
        citedcg_control_file: str,
        citedcg_treatment_file: str,
        output_dir: str,
        threads: int = 8,
        verbose: bool = False
    ):
        """
        Merge conversation details with CiteDCG scores.
        
        Args:
            conv_dir: Directory with conversation detail files
            citedcg_control_file: CiteDCG scores for control
            citedcg_treatment_file: CiteDCG scores for treatment
            output_dir: Output directory
            threads: Number of parallel threads
            verbose: Enable verbose logging
        """
        self._setup_logging(verbose)
        self.verbose = verbose  # Store for use in output methods
        
        conv_path = Path(conv_dir)
        output_path = Path(output_dir)
        
        if not self._validate_path(conv_path):
            return
        
        self._validate_path(output_path, create=True)
        
        # Load CiteDCG files (if provided) - now in JSONL format
        citedcg_control_count = 0
        citedcg_treatment_count = 0
        
        if citedcg_control_file:
            control_path = Path(citedcg_control_file)
            if control_path.exists():
                logger.info(f"Loading control CiteDCG: {citedcg_control_file}")
                with open(control_path, "r", encoding="utf-8") as f:
                    citedcg_control_count = sum(1 for line in f if line.strip())
                logger.info(
                    f"  Loaded {citedcg_control_count} control utterances"
                )
                self.citedcg_control_file = str(control_path)
            else:
                logger.error(f"Control CiteDCG file not found: {control_path}")
                return
        
        if citedcg_treatment_file:
            treatment_path = Path(citedcg_treatment_file)
            if treatment_path.exists():
                logger.info(
                    f"Loading treatment CiteDCG: {citedcg_treatment_file}"
                )
                with open(treatment_path, "r", encoding="utf-8") as f:
                    citedcg_treatment_count = sum(1 for line in f if line.strip())
                logger.info(
                    f"  Loaded {citedcg_treatment_count} treatment utterances"
                )
                self.citedcg_treatment_file = str(treatment_path)
            else:
                logger.error(
                    f"Treatment CiteDCG file not found: {treatment_path}"
                )
                return
        
        # Find conversation files
        control_files = sorted(conv_path.glob("control_*_conv_details.json"))
        
        # Try both naming patterns for treatment files
        treatment_files = sorted(
            conv_path.glob("treatment_*_conv_details.json")
        )
        if len(treatment_files) == 0:
            # Try alternate naming convention
            treatment_files = sorted(
                conv_path.glob("experiment_*_conv_details.json")
            )
        
        if self.verbose:
            print(f"Found {len(control_files)} control conversations")
            print(f"Found {len(treatment_files)} treatment conversations")
            print("=" * 80)
        
        # Process control files (only if CiteDCG file provided)
        if self.citedcg_control_file:
            if control_files:
                print(f"Processing {len(control_files)} control conversations...")
                print(f"  → Using CiteDCG: {self.citedcg_control_file}")
                control_output = output_path / "control"
                control_output.mkdir(exist_ok=True)
                
                control_results = self._parallel_process(
                    control_files,
                    lambda f: self._process_file(
                        f, control_output, "control", verbose
                    ),
                    threads,
                    verbose,
                    "control files"
                )
                self.results.extend(control_results)
            else:
                logger.warning("No control conversation files found")
        
        # Process treatment files (only if CiteDCG file provided)
        if self.citedcg_treatment_file:
            if treatment_files:
                print(
                    f"Processing {len(treatment_files)} treatment "
                    "conversations..."
                )
                print(f"  → Using CiteDCG: {self.citedcg_treatment_file}")
                treatment_output = output_path / "treatment"
                treatment_output.mkdir(exist_ok=True)
                
                treatment_results = self._parallel_process(
                    treatment_files,
                    lambda f: self._process_file(
                        f, treatment_output, "treatment", verbose
                    ),
                    threads,
                    verbose,
                    "treatment files"
                )
                self.results.extend(treatment_results)
            else:
                logger.warning("No treatment conversation files found")
        
        # Generate summary
        self._print_summary()
        
        self._print_header("MERGE COMPLETE")
    
    def _process_file(
        self,
        conv_file: Path,
        output_dir: Path,
        experiment_type: str,
        verbose: bool
    ) -> Optional[Dict[str, Any]]:
        """Merge a single conversation file with CiteDCG scores."""
        try:
            if verbose:
                logger.debug(f"Merging: {conv_file.name}")
            
            # Determine output files
            base_name = conv_file.stem.replace("_conv_details", "")
            merged_file = output_dir / f"{base_name}_merged.json"
            stats_file = output_dir / f"{base_name}_stats.json"
            
            # Select correct CiteDCG file path
            citedcg_file = (
                self.citedcg_control_file if experiment_type == "control"
                else self.citedcg_treatment_file
            )
            
            if not citedcg_file:
                logger.error(
                    f"No CiteDCG file for {experiment_type}: {conv_file.name}"
                )
                return None
            
            # Use merge_seval_results to do the actual merging
            merged_data, stats = merge_citedcg_and_calculate_stats(
                conversation_file=str(conv_file),
                citedcg_file=citedcg_file,
                output_file=str(merged_file),
                stats_file=str(stats_file),
                top_k=5  # Default value, not used in merge, only for stats compatibility
            )
            
            # Load merged file to get statistics
            with open(merged_file, "r", encoding="utf-8") as f:
                merged_content = json.load(f)
            
            # Count hops and turns
            total_hops = 0
            nonempty_hops = 0  # Only hops WITH invocations (search attempted)
            max_hops_including_empty = 0  # Max hops across all turns (structure exists)
            max_hops_excluding_empty = 0  # Max hops with invocations (search attempted)
            
            eval_results = merged_content.get("evaluation_data_results", {})
            total_turns = len(eval_results.get("turns", []))
            
            # Count results with scores and track max hops
            results_with_scores = 0
            for turn in eval_results.get("turns", []):
                turn_hops = turn.get("hops", [])
                turn_total_hops = len(turn_hops)
                turn_nonempty_hops = 0
                
                for hop in turn_hops:
                    total_hops += 1
                    invocations = hop.get("invocations", [])
                    # DEFINITION: "Non-empty hop" here means has invocations
                    # (search was attempted), not just hop structure exists
                    if invocations:
                        nonempty_hops += 1
                        turn_nonempty_hops += 1
                        # Count how many results have scores in this hop
                        for inv in invocations:
                            for query in inv.get("queries", []):
                                for result in query.get("results", []):
                                    if result.get("citedcg_score") is not None:
                                        results_with_scores += 1
                
                # Track max hops for this turn
                max_hops_including_empty = max(max_hops_including_empty, turn_total_hops)
                max_hops_excluding_empty = max(max_hops_excluding_empty, turn_nonempty_hops)
            
            # Track if this file has any scores
            has_scores = results_with_scores > 0
            if not has_scores:
                self.no_scores_count += 1
            
            # Extract utterances_not_in_dcg from summary (NEW location)
            summary = eval_results.get("summary", {})
            utterances_not_in_dcg = summary.get("utterances_not_in_dcg", 0)
            queries_no_match = summary.get("queries_no_match", 0)
            
            # Extract multi-turn turn information
            turn_used_for_scoring = summary.get("turn_used_for_scoring", None)
            
            return {
                "file": conv_file.name,
                "experiment": experiment_type,
                "merged_file": str(merged_file),
                "stats_file": str(stats_file),
                "results_with_scores": results_with_scores,
                "total_turns": total_turns,
                "total_hops": total_hops,
                "nonempty_hops": nonempty_hops,
                "max_hops_including_empty": max_hops_including_empty,
                "max_hops_excluding_empty": max_hops_excluding_empty,
                "has_scores": has_scores,
                "utterances_not_in_dcg": utterances_not_in_dcg,
                "queries_no_match": queries_no_match,
                "turn_used_for_scoring": turn_used_for_scoring,
            }
        
        except Exception as e:
            logger.error(f"Error merging {conv_file.name}: {e}")
            self.errors.append({
                "file": conv_file.name,
                "error": str(e)
            })
            return None
    
    def _print_summary(self):
        """Print merge summary."""
        total = len(self.results)
        
        # Accumulate utterances_not_in_dcg from all results
        self.utterances_not_in_dcg = sum(
            r.get("utterances_not_in_dcg", 0) for r in self.results
        )
        
        print("")
        
        if total > 0:
            # Separate results by experiment
            control_results = [r for r in self.results if r.get("experiment") == "control"]
            treatment_results = [r for r in self.results if r.get("experiment") == "treatment"]
            
            # Check if we have both experiments
            has_both = len(control_results) > 0 and len(treatment_results) > 0
            
            if has_both:
                # Print basic summary for each experiment (always shown)
                for exp_name, exp_results in [("CONTROL", control_results), ("TREATMENT", treatment_results)]:
                    exp_total = len(exp_results)
                    exp_no_scores = sum(1 for r in exp_results if not r.get("has_scores", False))
                    exp_not_in_dcg = sum(r.get("utterances_not_in_dcg", 0) for r in exp_results)
                    exp_queries_no_match = sum(r.get("queries_no_match", 0) for r in exp_results)
                    exp_errors = sum(1 for e in self.errors if any(r.get("file") == e.get("file") for r in exp_results))
                    
                    print(f"{exp_name} Experiment Summary:")
                    print("-" * 80)
                    print(f"Total conversations:     {exp_total}")
                    print(f"  With CiteDCG scores:   {exp_total - exp_no_scores}")
                    print(f"  Without scores:        {exp_no_scores}")
                    print(f"  Not in DCG data:       {exp_not_in_dcg}")
                    print(f"  Queries no match:      {exp_queries_no_match}")
                    print(
                        f"  Matching errors:       {exp_errors} "
                        f"(DCG exists but tools/queries/results don't match)"
                    )
                    
                    # Only show detailed statistics when verbose=True
                    if not self.verbose:
                        print("")
                        continue
                    
                    # Hop and Turn Statistics for this experiment
                    exp_results_scored = sum(r.get("results_with_scores", 0) for r in exp_results)
                    exp_total_hops = sum(r.get("total_hops", 0) for r in exp_results)
                    exp_nonempty_hops = sum(r.get("nonempty_hops", 0) for r in exp_results)
                    
                    # Calculate max hops across all utterances
                    exp_max_hops_incl_empty = max((r.get("max_hops_including_empty", 0) for r in exp_results), default=0)
                    exp_max_hops_excl_empty = max((r.get("max_hops_excluding_empty", 0) for r in exp_results), default=0)
                    
                    print("")
                    print("  Hop and Turn Statistics:")
                    print(f"    Search results with scores: {exp_results_scored}")
                    print(f"    Total hops (incl empty):    {exp_total_hops}")
                    print(f"    Non-empty hops:             {exp_nonempty_hops}")
                    print(f"    Max hops per utterance (including empty): {exp_max_hops_incl_empty}")
                    print(f"    Max hops per utterance (excluding empty): {exp_max_hops_excl_empty}")
                    
                    # Turn distribution - track which turn has non-empty hops
                    turn_dist = {}
                    turn_nonempty_dist = {}  # Track non-empty turn for each turn count
                    multi_turn_with_invocations = 0
                    multi_turn_with_scores = 0
                    multi_turn_no_invocations = 0
                    
                    for result in exp_results:
                        turn_count = result.get("total_turns", 0)
                        turn_dist[turn_count] = turn_dist.get(turn_count, 0) + 1
                        
                        # Track which turn has non-empty hops for multi-turn
                        if turn_count > 1:
                            nonempty_turn = result.get("turn_used_for_scoring", None)
                            if nonempty_turn:
                                key = (turn_count, nonempty_turn)
                                if key not in turn_nonempty_dist:
                                    turn_nonempty_dist[key] = 0
                                turn_nonempty_dist[key] += 1
                        
                        if turn_count > 1:
                            nonempty_hops = result.get("nonempty_hops", 0)
                            has_scores = result.get("has_scores", False)
                            
                            if nonempty_hops > 0:
                                multi_turn_with_invocations += 1
                                if has_scores:
                                    multi_turn_with_scores += 1
                            else:
                                multi_turn_no_invocations += 1
                    
                    if turn_dist:
                        print("")
                        print("  Turn Distribution:")
                        for turn_count in sorted(turn_dist.keys()):
                            file_count = turn_dist[turn_count]
                            pct = (
                                (file_count / exp_total * 100)
                                if exp_total > 0 else 0
                            )
                            
                            # Format turn count display
                            if turn_count == 0:
                                turn_label = f"{turn_count} turns"
                            elif turn_count == 1:
                                turn_label = f"{turn_count} turn "
                            else:
                                turn_label = f"{turn_count} turns"
                            
                            # Add non-empty turn info for multi-turn
                            if turn_count > 1:
                                # Find which turn has non-empty hops
                                nonempty_turns = [
                                    nt for (tc, nt), count
                                    in turn_nonempty_dist.items()
                                    if tc == turn_count
                                ]
                                if nonempty_turns:
                                    # Get most common non-empty turn
                                    from collections import Counter
                                    most_common = Counter(
                                        nonempty_turns
                                    ).most_common(1)[0][0]
                                    turn_info = f" (turn {most_common} non-empty)"
                                else:
                                    # No turns with invocations found
                                    turn_info = " (0 non-empty turns)"
                                print(
                                    f"    {turn_label}{turn_info}: "
                                    f"{file_count} conversations ({pct:.1f}%)"
                                )
                            else:
                                print(
                                    f"    {turn_label}: "
                                    f"{file_count} conversations ({pct:.1f}%)"
                                )
                        
                        multi_turn_total = sum(turn_dist.get(k, 0) for k in turn_dist if k > 1)
                        if multi_turn_total > 0:
                            print("")
                            print("    Multi-turn breakdown (retry scenarios):")
                            print(f"      Total: {multi_turn_total} conversations")
                            pct_inv = (multi_turn_with_invocations / multi_turn_total * 100)
                            print(f"      With invocations: {multi_turn_with_invocations} ({pct_inv:.1f}% - search attempted)")
                            print(f"        → With CiteDCG scores: {multi_turn_with_scores}")
                            no_scores = multi_turn_with_invocations - multi_turn_with_scores
                            print(f"        → No scores: {no_scores} (not in DCG or no match)")
                            pct_no_inv = (multi_turn_no_invocations / multi_turn_total * 100)
                            print(f"      No invocations: {multi_turn_no_invocations} ({pct_no_inv:.1f}% - search not attempted)")
            else:
                # Single experiment - use original format
                # Calculate queries_no_match for single experiment
                total_queries_no_match = sum(
                    r.get("queries_no_match", 0) for r in self.results
                )
                
                print("")
                print("Utterance-Based Summary:")
                print("-" * 80)
                print(f"Total conversations processed: {total}")
                print(f"  With CiteDCG scores:     {total - self.no_scores_count}")
                print(f"  Without scores:          {self.no_scores_count}")
                print(f"  Not in DCG data:         {self.utterances_not_in_dcg}")
                print(f"  Queries no match:        {total_queries_no_match}")
                print(
                    f"  Matching errors:         {len(self.errors)} "
                    f"(DCG exists but tools/queries/results don't match)"
                )
                
                # SECTION 2: EXPERIMENT BREAKDOWN
                control = sum(
                    1 for r in self.results if r.get("experiment") == "control"
                )
                treatment = sum(
                    1 for r in self.results
                    if r.get("experiment") == "treatment"
                )
                
                print("")
                print("Experiment Breakdown:")
                print("-" * 80)
                print(f"Control conversations:   {control}")
                print(f"Treatment conversations: {treatment}")
                
                # SECTION 3: HOP AND TURN STATISTICS (for single experiment)
                total_results_scored = sum(
                    r.get("results_with_scores", 0) for r in self.results
                )
                total_hops = sum(
                    r.get("total_hops", 0) for r in self.results
                )
                total_nonempty_hops = sum(
                    r.get("nonempty_hops", 0) for r in self.results
                )
                
                # Calculate max hops across all utterances
                max_hops_incl_empty = max(
                    (r.get("max_hops_including_empty", 0) for r in self.results),
                    default=0
                )
                max_hops_excl_empty = max(
                    (r.get("max_hops_excluding_empty", 0) for r in self.results),
                    default=0
                )
                
                print("")
                print("Hop and Turn Statistics:")
                print("-" * 80)
                print(f"Total search results with scores: {total_results_scored}")
                print(f"Total hops (incl empty): {total_hops}")
                print(f"Non-empty hops:          {total_nonempty_hops}")
                print(f"Max hops per utterance (including empty): {max_hops_incl_empty}")
                print(f"Max hops per utterance (excluding empty): {max_hops_excl_empty}")
                
                # Turn distribution with multi-turn breakdown
                turn_dist = {}
                turn_nonempty_dist = {}  # Track non-empty turn for each turn count
                # NOTE: "nonempty_hops" in merge phase = hops with invocations
                # (not just hop data structures, but actual search attempts)
                multi_turn_with_invocations = 0
                multi_turn_with_scores = 0
                multi_turn_no_invocations = 0
                
                for result in self.results:
                    turn_count = result.get("total_turns", 0)
                    turn_dist[turn_count] = turn_dist.get(turn_count, 0) + 1
                    
                    # Track which turn has non-empty hops for multi-turn
                    if turn_count > 1:
                        nonempty_turn = result.get("turn_used_for_scoring", None)
                        # If not set, assume last turn for multi-turn with hops
                        if nonempty_turn is None and result.get("nonempty_hops", 0) > 0:
                            nonempty_turn = turn_count
                        if nonempty_turn:
                            key = (turn_count, nonempty_turn)
                            if key not in turn_nonempty_dist:
                                turn_nonempty_dist[key] = 0
                            turn_nonempty_dist[key] += 1
                    
                    # Track multi-turn conversations
                    if turn_count > 1:
                        # "nonempty_hops" = hops WITH invocations
                        nonempty_hops = result.get("nonempty_hops", 0)
                        has_scores = result.get("has_scores", False)
                        
                        if nonempty_hops > 0:
                            multi_turn_with_invocations += 1
                            if has_scores:
                                multi_turn_with_scores += 1
                        else:
                            multi_turn_no_invocations += 1
                
                if turn_dist:
                    print("")
                    print("Turn Distribution:")
                    for turn_count in sorted(turn_dist.keys()):
                        file_count = turn_dist[turn_count]
                        pct = (file_count / total * 100) if total > 0 else 0
                        
                        # Format turn count display
                        if turn_count == 0:
                            turn_label = f"{turn_count} turns"
                        elif turn_count == 1:
                            turn_label = f"{turn_count} turn "
                        else:
                            turn_label = f"{turn_count} turns"
                        
                        # Add non-empty turn info for multi-turn
                        if turn_count > 1:
                            # Find which turn has non-empty hops
                            nonempty_turns = [
                                nt for (tc, nt), count
                                in turn_nonempty_dist.items()
                                if tc == turn_count
                            ]
                            if nonempty_turns:
                                # Get most common non-empty turn
                                from collections import Counter
                                most_common = Counter(
                                    nonempty_turns
                                ).most_common(1)[0][0]
                                turn_info = f" (turn {most_common} non-empty)"
                            else:
                                # No turns with invocations found
                                turn_info = " (0 non-empty turns)"
                            print(
                                f"  {turn_label}{turn_info}: "
                                f"{file_count} conversations ({pct:.1f}%)"
                            )
                        else:
                            print(
                                f"  {turn_label}: "
                                f"{file_count} conversations ({pct:.1f}%)"
                            )
                    
                    # Add detailed multi-turn summary
                    multi_turn_total = sum(
                        turn_dist.get(k, 0) for k in turn_dist if k > 1
                    )
                    if multi_turn_total > 0:
                        print("")
                        print("  Multi-turn breakdown:")
                        print(f"    Total: {multi_turn_total} conversations")
                        pct_inv = (
                            multi_turn_with_invocations / multi_turn_total * 100
                        )
                        print(
                            f"    With invocations: {multi_turn_with_invocations}"
                            f" ({pct_inv:.1f}% - search attempted)"
                        )
                        print(f"      → With CiteDCG scores: {multi_turn_with_scores}")
                        no_scores = (
                            multi_turn_with_invocations - multi_turn_with_scores
                        )
                        print(
                            f"      → No scores: {no_scores} "
                            f"(not in DCG or no match)"
                        )
                        pct_no_inv = (
                            multi_turn_no_invocations / multi_turn_total * 100
                        )
                        print(
                            f"    No invocations: {multi_turn_no_invocations} "
                            f"({pct_no_inv:.1f}% - search not attempted)"
                        )
        
        print("=" * 80)
    
    def _print_header(self, title: str):
        """Print formatted section header."""
        print("=" * 80)
        print(title)
        print("=" * 80)


# ==========================================================================
# CLI Functions
# ==========================================================================

def extract_conversations(
    input_dir: str = r"seval_data\130949_scraping_raw_data_output",
    output_dir: str = r"results\conversation_details",
    experiment: str = "both",
    threads: int = 8,
    verbose: bool = False,
    clean: bool = False
):
    """
    Extract conversation details from SEVAL files.
    
    Args:
        input_dir: Directory with SEVAL JSON files
        output_dir: Output directory for extracted details
        experiment: Experiment type: 'control', 'treatment', or 'both' (default: 'both')
        threads: Number of parallel threads (default: 8)
        verbose: Enable verbose logging
        clean: Clean output directory before extraction (default: False)
        
    Example:
        python seval_batch_processor.py extract_conversations
        python seval_batch_processor.py extract_conversations --experiment control
        python seval_batch_processor.py extract_conversations --experiment treatment
        python seval_batch_processor.py extract_conversations --experiment both --threads 16
        python seval_batch_processor.py extract_conversations --clean=True
    """
    extractor = ConversationExtractor()
    return extractor.process(input_dir, output_dir, experiment, threads, verbose, clean)


def merge_citescg_scores(
    conv_dir: str = r"results\conversation_details",
    citedcg_control_file: str = "",
    citedcg_treatment_file: str = "",
    output_dir: str = r"results\merged",
    threads: int = 8,
    verbose: bool = False
):
    """
    Batch merge conversation details with CiteDCG scores.
    
    Matches conversations with CiteDCG scores by:
    1. Experiment type (control vs treatment) from filename
    2. Utterance text (user query)
    3. Search domain and query string
    
    Args:
        conv_dir: Directory with conversation detail files
        citedcg_control_file: Optional CiteDCG scores file for control
        citedcg_treatment_file: Optional CiteDCG scores file for treatment
        output_dir: Output directory for merged results and statistics
        threads: Number of parallel threads
        verbose: Enable verbose logging
        
    Example:
        # Process both control and treatment
        python seval_batch_processor.py merge_citescg_scores \\
            --conv_dir="results/conversation_details" \\
            --citedcg_control_file="results/control_citedcg.json" \\
            --citedcg_treatment_file="results/treatment_citedcg.json"
        
        # Process only control
        python seval_batch_processor.py merge_citescg_scores \\
            --conv_dir="results/conversation_details" \\
            --citedcg_control_file="results/control_citedcg.json"
    """
    processor = MergeCiteDCGProcessor()
    processor._setup_logging(verbose)
    
    if verbose:
        processor._print_header("BATCH CITEDCG MERGE")
        print(f"Conversation dir:      {conv_dir}")
        if citedcg_control_file:
            print(f"Control CiteDCG:       {citedcg_control_file}")
        if citedcg_treatment_file:
            print(f"Treatment CiteDCG:     {citedcg_treatment_file}")
        if not citedcg_control_file and not citedcg_treatment_file:
            logger.error("At least one CiteDCG file must be provided")
            return
        print(f"Output dir:            {output_dir}")
        print(f"Threads:               {threads}")
        print("=" * 80)
    
    # Always clean output directory to avoid mixing old/new results
    output_path = Path(output_dir)
    if output_path.exists():
        merged_files = list(output_path.rglob("*.json"))
        if merged_files:
            if verbose:
                print(f"Output dir contains {len(merged_files)} files, deleting...")
            import shutil
            shutil.rmtree(output_path)
            if verbose:
                print("Done")
                print("")
    
    processor.process(
        conv_dir=conv_dir,
        citedcg_control_file=citedcg_control_file,
        citedcg_treatment_file=citedcg_treatment_file,
        output_dir=output_dir,
        threads=threads,
        verbose=verbose
    )


def process_seval_job_multihop_citedcg(
    job_id: str = "130949",
    raw_data_dir: str = None,
    metrics_dir: str = None,
    output_base_dir: str = "results",
    experiment: str = "control",
    top_k: int = 5,
    threads: int = 8,
    clean: bool = False,
    verbose: bool = False
):
    """
    Complete end-to-end multi-hop CiteDCG processing for a SEVAL job in one command.
    
    Runs all three steps automatically:
    1. Extract CiteDCG scores from metrics
    2. Extract conversation details from raw data
    3. Merge CiteDCG scores with conversation details
    
    Args:
        job_id: SEVAL job ID (e.g., "130949")
        raw_data_dir: Raw data directory (default: seval_data/{job_id}_scraping_raw_data_output)
        metrics_dir: Metrics folder name only (default: {job_id}_metrics, NOT full path)
        output_base_dir: Base output directory (default: "results")
        experiment: Experiment type: 'control', 'treatment', or 'both'
        top_k: Top-k value for CiteDCG calculation
        threads: Number of parallel threads
        clean: If True, re-extract both CiteDCG and conversations. If False, reuse existing files
        verbose: Enable verbose logging
        
    Example:
        # Process job 130949 (control only)
        python seval_batch_processor.py process_seval_job_multihop_citedcg --job_id=130949
        
        # Process both control and treatment with 16 threads
        python seval_batch_processor.py process_seval_job_multihop_citedcg \\
            --job_id=130949 --experiment=both --threads=16
    """
    from get_seval_metrics import extract_per_result_citedcg

    # Set default paths if not provided
    if raw_data_dir is None:
        raw_data_dir = f"seval_data/{job_id}_scraping_raw_data_output"
    if metrics_dir is None:
        metrics_dir = f"{job_id}_metrics"  # Just folder name, not full path
    
    # Create output directories
    citedcg_dir = f"{output_base_dir}/{job_id}_citedcg"
    conv_dir = f"{output_base_dir}/{job_id}_conversation_details"
    merged_dir = f"{output_base_dir}/{job_id}_conversation_w_citedcg_details"
    
    Path(citedcg_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"SEVAL JOB PROCESSING: {job_id}")
    print("=" * 80)
    print(f"Raw data:      {raw_data_dir}")
    print(f"Metrics:       seval_data/{metrics_dir}")
    print(f"Experiment:    {experiment}")
    print(f"Top-k:         {top_k}")
    print(f"Threads:       {threads}")
    print("=" * 80)
    print("")
    
    # Determine which experiments to process
    experiments = []
    if experiment.lower() in ["control", "both"]:
        experiments.append("control")
    if experiment.lower() in ["treatment", "both"]:
        experiments.append("treatment")
    
    citedcg_files = {}
    
    # Step 1: Extract CiteDCG scores for each experiment
    for exp in experiments:
        citedcg_file = f"{citedcg_dir}/{job_id}_citedcg_scores_{exp}.json"
        citedcg_files[exp] = citedcg_file
        
        # Check if file exists and clean=False
        if not clean and Path(citedcg_file).exists():
            print(f"STEP 1/{len(experiments)}: Reusing existing CiteDCG scores ({exp})...")
            print(f"  → File: {citedcg_file}")
            print(f"  → Use --clean=True to re-extract")
        else:
            print(f"STEP 1/{len(experiments)}: Extracting CiteDCG scores ({exp})...")
            count = extract_per_result_citedcg(
                metrics_folder=metrics_dir,
                experiment=exp,
                output_file=citedcg_file
            )
            print(f"  → Extracted {count} utterances")
            print(f"  → Saved to: {citedcg_file}")
        print("")
    
    # Step 2: Extract conversation details
    # Check if output directory exists and has files
    conv_dir_path = Path(conv_dir)
    has_existing_convs = (
        conv_dir_path.exists() and 
        len(list(conv_dir_path.glob('*.json'))) > 0
    )
    
    if not clean and has_existing_convs:
        print(f"STEP 2: Reusing existing conversation details...")
        print(f"  → Directory: {conv_dir}")
        print(f"  → Files: {len(list(conv_dir_path.glob('*.json')))}")
        print(f"  → Use --clean=True to re-extract")
    else:
        print(f"STEP 2: Extracting conversation details...")
        extract_conversations(
            input_dir=raw_data_dir,
            output_dir=conv_dir,
            experiment=experiment,
            threads=threads,
            verbose=verbose,
            clean=clean  # Pass clean parameter
        )
    print("")
    
    # Step 3: Merge CiteDCG with conversations
    print(f"STEP 3: Merging CiteDCG scores with conversations...")
    merge_citescg_scores(
        conv_dir=conv_dir,
        citedcg_control_file=citedcg_files.get("control"),
        citedcg_treatment_file=citedcg_files.get("treatment"),
        output_dir=merged_dir,
        threads=threads,
        verbose=verbose
    )
    print("")
    
    print("=" * 80)
    print("✓ SEVAL JOB PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Merged results: {merged_dir}")
    print("=" * 80)


def process_seval_job_with_statistics_plots(
    job_id: str,
    experiment: str = "control",
    top_k_list: str = "1,3,5",
    threads: int = 8,
    raw_data_dir: str = None,
    metrics_dir: str = None,
    output_base_dir: str = "results",
    clean: bool = False,
    verbose: bool = False
):
    """
    Process a SEVAL job: extract data, merge with CiteDCG, calculate statistics
    for multiple top-k values, and generate plots.
    
    New Architecture (optimized for multiple top-k values):
    1. Extract CiteDCG scores from metrics (done once, reused for all k)
    2. Extract conversation details (done once, reused for all k)
    3. Merge conversations with CiteDCG scores (done once, no statistics)
    4. Calculate statistics for each top-k value from merged data
    5. Generate visualization plots comparing the different top-k values
    
    
    Efficiency: Merges conversation data with CiteDCG scores ONCE, then calculates
    statistics for different top-k values by reading the same merged files.
    This is much faster than re-merging for each k value.
    
    Args:
        job_id: SEVAL job ID (e.g., "130949")
        experiment: Which experiment to process ("control", "treatment", or "both")
        top_k_list: Comma-separated list of top-k values (e.g., "1,3,5")
        threads: Number of parallel threads for processing (default: 8)
        raw_data_dir: Raw SEVAL data directory 
            (default: seval_data/{job_id}_scraping_raw_data_output)
        metrics_dir: Metrics data directory (default: {job_id}_metrics)
        output_base_dir: Base directory for outputs (default: "results")
        clean: Force complete regeneration (default: False)
            When False: Reuses existing CiteDCG scores, conversation details,
                       and merged files if they exist
            When True: Deletes and regenerates everything from scratch
            Note: Plots folder is ALWAYS cleaned regardless of this setting
        verbose: Enable verbose output (default: False)
    
    Example:
        # Process with default top-k values [1, 3, 5]
        python seval_batch_processor.py process_seval_job_with_statistics_plots \\
            --job_id=130949
        
        # Process with custom top-k values
        python seval_batch_processor.py process_seval_job_with_statistics_plots \\
            --job_id=130949 \\
            --top_k_list="1,2,3,5,10" \\
            --threads=16
        
        # Force complete regeneration
        python seval_batch_processor.py process_seval_job_with_statistics_plots \\
            --job_id=130949 \\
            --clean=True
    
    Note:
        With --clean=False (default): Reuses existing files (CiteDCG scores,
        conversation details, merged data) to save time.
        
        With --clean=True: Deletes and regenerates all intermediate files
        from scratch.
    """
    import sys

    from get_seval_metrics import extract_per_result_citedcg

    # Parse top-k list - handle both string and tuple/list from Fire
    try:
        if isinstance(top_k_list, (list, tuple)):
            # Fire parsed as multiple arguments
            k_values = [int(k) for k in top_k_list]
        elif isinstance(top_k_list, str):
            # String format "1,3,5"
            k_values = [int(k.strip()) for k in top_k_list.split(",")]
        else:
            # Single integer
            k_values = [int(top_k_list)]
        
        k_values = sorted(set(k_values))  # Remove duplicates and sort
    except (ValueError, AttributeError) as e:
        print(f"Error: Invalid top_k_list format: {top_k_list}")
        print(f"Expected comma-separated integers like '1,3,5' or list [1,3,5]")
        print(f"Error details: {e}")
        sys.exit(1)
    
    if len(k_values) < 1:
        print("Error: At least one top-k value is required")
        sys.exit(1)
    
    # Set default paths if not provided
    if raw_data_dir is None:
        raw_data_dir = f"seval_data/{job_id}_scraping_raw_data_output"
    if metrics_dir is None:
        metrics_dir = f"{job_id}_metrics"
    
    # Create base output directory
    Path(output_base_dir).mkdir(parents=True, exist_ok=True)
    
    # Helper function for verbose-controlled printing
    def vprint(*args, **kwargs):
        """Print only when verbose=True (for detailed logs)"""
        if verbose:
            print(*args, **kwargs)
    
    # Always show basic job info
    if verbose:
        print("Processing SEVAL data...")
        print("=" * 80)
        print(f"SEVAL JOB PROCESSING WITH MULTIPLE TOP-K VALUES: {job_id}")
        print("=" * 80)
        print(f"Raw data:      {raw_data_dir}")
        print(f"Metrics:       seval_data/{metrics_dir}")
        print(f"Experiment:    {experiment}")
        print(f"Top-k values:  {k_values}")
        print(f"Threads:       {threads}")
        print(f"Clean existing: {clean}")
        print("=" * 80)
    print("")
    
    # Determine which experiments to process
    experiments = []
    if experiment.lower() in ["control", "both"]:
        experiments.append("control")
    if experiment.lower() in ["treatment", "both"]:
        experiments.append("treatment")
    
    if not experiments:
        print(f"Error: Invalid experiment value: '{experiment}'")
        print("Must be 'control', 'treatment', or 'both'")
        sys.exit(1)
    
    # Define output directories (new architecture: single merged folder)
    existing_citedcg_dir = f"{output_base_dir}/{job_id}_citedcg"
    existing_conv_dir = f"{output_base_dir}/{job_id}_conversation_details"
    merged_dir = (
        f"{output_base_dir}/{job_id}_conversation_w_citedcg_details"
    )
    
    # Clean existing directories if requested
    processor = BaseProcessor()
    if clean:
        vprint("FORCING COMPLETE REGENERATION (--clean=True):")
        vprint("  Deleting all intermediate files...")
        vprint("")
        
        # Clean all intermediate directories
        processor._clean_directory(
            Path(existing_citedcg_dir), "CiteDCG scores"
        )
        processor._clean_directory(
            Path(existing_conv_dir), "conversation details"
        )
        processor._clean_directory(
            Path(merged_dir), "merged conversations with CiteDCG"
        )
        vprint("")
        
        can_reuse_citedcg = False
        can_reuse_conv = False
        can_reuse_merged = False
    else:
        # Check what can be reused - must have ALL required experiments
        can_reuse_citedcg = Path(existing_citedcg_dir).exists()
        
        # Check if conversation directory exists AND contains files for ALL experiments
        conv_path = Path(existing_conv_dir)
        if conv_path.exists():
            conv_files = list(conv_path.glob("*_conv_details.json"))
            if len(conv_files) > 0:
                # Check if we have files for all required experiments
                has_control = any("control_" in f.name for f in conv_files)
                # Treatment files can be named "treatment_" or "experiment_"
                has_treatment = any(
                    "treatment_" in f.name or "experiment_" in f.name
                    for f in conv_files
                )
                
                if "control" in experiments and not has_control:
                    can_reuse_conv = False
                elif "treatment" in experiments and not has_treatment:
                    can_reuse_conv = False
                else:
                    can_reuse_conv = True
            else:
                can_reuse_conv = False
        else:
            can_reuse_conv = False
        
        # Check if merged directory exists AND contains files for ALL experiments
        merged_path = Path(merged_dir)
        if merged_path.exists():
            merged_files = list(merged_path.rglob("*.json"))
            if len(merged_files) > 0:
                # Check if we have files for all required experiments
                has_control = any("control_" in f.name for f in merged_files)
                # Treatment files can be named "treatment_" or "experiment_"
                has_treatment = any(
                    "treatment_" in f.name or "experiment_" in f.name
                    for f in merged_files
                )
                
                if "control" in experiments and not has_control:
                    can_reuse_merged = False
                elif "treatment" in experiments and not has_treatment:
                    can_reuse_merged = False
                else:
                    can_reuse_merged = True
            else:
                can_reuse_merged = False
        else:
            can_reuse_merged = False
        
        if can_reuse_citedcg or can_reuse_conv or can_reuse_merged:
            vprint("REUSING EXISTING RESULTS:")
            if can_reuse_citedcg:
                vprint(f"  ✓ CiteDCG scores: {existing_citedcg_dir}")
            if can_reuse_conv:
                control_conv = len(
                    list(conv_path.glob("control_*_conv_details.json"))
                )
                # Count treatment files (both naming patterns)
                treatment_conv = len(
                    list(conv_path.glob("treatment_*_conv_details.json"))
                )
                if treatment_conv == 0:
                    treatment_conv = len(
                        list(conv_path.glob("experiment_*_conv_details.json"))
                    )
                
                exp_info = []
                if control_conv > 0:
                    exp_info.append(f"control: {control_conv}")
                if treatment_conv > 0:
                    exp_info.append(f"treatment: {treatment_conv}")
                vprint(
                    f"  ✓ Conversation files ({', '.join(exp_info)}): "
                    f"{existing_conv_dir}"
                )
            if can_reuse_merged:
                control_merged = len(
                    [f for f in merged_files if "control_" in f.name]
                )
                # Count treatment files (both naming patterns)
                treatment_merged = len(
                    [f for f in merged_files 
                     if "treatment_" in f.name or "experiment_" in f.name]
                )
                exp_info = []
                if control_merged > 0:
                    exp_info.append(f"control: {control_merged}")
                if treatment_merged > 0:
                    exp_info.append(f"treatment: {treatment_merged}")
                vprint(
                    f"  ✓ Merged files ({', '.join(exp_info)}): "
                    f"{merged_dir}"
                )
            vprint("")
    
    # STEP 1: Extract CiteDCG scores ONCE (same for all top-k values)
    citedcg_files = {}
    
    if can_reuse_citedcg:
        vprint("=" * 80)
        vprint("STEP 1: REUSING EXISTING CITEDCG SCORES")
        vprint("=" * 80)
        for exp in experiments:
            citedcg_file = (
                f"{existing_citedcg_dir}/{job_id}_citedcg_scores_{exp}.json"
            )
            if Path(citedcg_file).exists():
                # Count utterances in CiteDCG file
                with open(citedcg_file, 'r', encoding='utf-8') as f:
                    utterance_count = sum(1 for line in f if line.strip())
                citedcg_files[exp] = citedcg_file
                vprint(f"  ✓ {exp}: {citedcg_file} ({utterance_count} utterances)")
            else:
                vprint(f"  ✗ {exp}: Not found, will extract")
        vprint("")
    
    # Extract CiteDCG if not reusing or if files missing
    if not can_reuse_citedcg or len(citedcg_files) < len(experiments):
        print("=" * 80)
        print("STEP 1: EXTRACTING CITEDCG SCORES ...")
        print("=" * 80)
        citedcg_dir = Path(existing_citedcg_dir)
        citedcg_dir.mkdir(parents=True, exist_ok=True)
        
        for exp in experiments:
            if exp in citedcg_files:
                vprint(f"  Skipping {exp} (already exists)")
                continue
            
            citedcg_file = f"{citedcg_dir}/{job_id}_citedcg_scores_{exp}.json"
            
            count = extract_per_result_citedcg(
                metrics_folder=metrics_dir,
                experiment=exp,
                output_file=citedcg_file,
                verbose=verbose
            )
            
            citedcg_files[exp] = citedcg_file
        print("")
    
    # Validate that we have all required CiteDCG files
    if len(citedcg_files) < len(experiments):
        print("")
        print("=" * 80)
        print("✗ ERROR: Missing CiteDCG files")
        print("=" * 80)
        print(f"Expected {len(experiments)} experiment(s): {experiments}")
        print(f"Found {len(citedcg_files)} file(s): {list(citedcg_files.keys())}")
        print("")
        print("Please run this command first to extract CiteDCG scores:")
        print(f"  python seval_batch_processor.py process_seval_job_multihop_citedcg \\")
        print(f"    --job_id={job_id} --experiment={experiment}")
        print("=" * 80)
        sys.exit(1)
    
    # STEP 2: Extract conversation details ONCE (same for all top-k values)
    conv_dir = existing_conv_dir
    
    if can_reuse_conv:
        # Count conversation files
        conv_file_count = len(list(Path(existing_conv_dir).glob("*_conv_details.json")))
        vprint("=" * 80)
        vprint("STEP 2: REUSING EXISTING CONVERSATION DETAILS")
        vprint("=" * 80)
        vprint(f"  Location: {existing_conv_dir}")
        vprint(f"  Files: {conv_file_count} conversation files")
        vprint("")
    else:
        print("=" * 80)
        print("STEP 2: EXTRACTING CONVERSATION DETAILS ...")
        print("=" * 80)
        vprint(f"  Input:  {raw_data_dir}")
        vprint(f"  Output: {conv_dir}")
        vprint(f"  Experiments: {', '.join(experiments)}")
        vprint(f"  Threads: {threads}")
        vprint("")
        
        extract_conversations(
            input_dir=raw_data_dir,
            output_dir=conv_dir,
            experiment=experiment,
            threads=threads,
            verbose=verbose
        )
        print("")
    
    # Validate that conversation files exist
    conv_path = Path(conv_dir)
    if not conv_path.exists():
        print("")
        print("=" * 80)
        print("✗ ERROR: Conversation directory not found")
        print("=" * 80)
        print(f"Expected directory: {conv_dir}")
        print("")
        print("Please run this command first to extract conversations:")
        print(f"  python seval_batch_processor.py process_seval_job_multihop_citedcg \\")
        print(f"    --job_id={job_id} --experiment={experiment}")
        print("=" * 80)
        sys.exit(1)
    
    conv_files = list(conv_path.glob("*_conv_details.json"))
    if len(conv_files) == 0:
        print("")
        print("=" * 80)
        print("✗ ERROR: No conversation files found")
        print("=" * 80)
        print(f"Directory exists but is empty: {conv_dir}")
        print("")
        print("Please run this command first to extract conversations:")
        print(f"  python seval_batch_processor.py process_seval_job_multihop_citedcg \\")
        print(f"    --job_id={job_id} --experiment={experiment}")
        print("=" * 80)
        sys.exit(1)
    
    # STEP 3: Merge CiteDCG with conversations ONCE (no statistics)
    if can_reuse_merged:
        # Count merged files
        merged_file_count = len(list(Path(merged_dir).rglob("*_merged.json")))
        vprint("=" * 80)
        vprint("STEP 3: REUSING EXISTING MERGED DATA")
        vprint("=" * 80)
        vprint(f"  Location: {merged_dir}")
        vprint(f"  Files: {merged_file_count} merged files")
        vprint("")
    else:
        print("=" * 80)
        print("STEP 3: MERGING CITEDCG WITH CONVERSATIONS (Extract CiteDCG scores for search results) ...")
        print("=" * 80)
        
        merge_citescg_scores(
            conv_dir=conv_dir,
            citedcg_control_file=citedcg_files.get("control"),
            citedcg_treatment_file=citedcg_files.get("treatment"),
            output_dir=merged_dir,
            threads=threads,
            verbose=verbose
        )
        vprint(f"✓ Merge complete: {merged_dir}")
        print("")
    
    # Validate merged files exist
    merged_path = Path(merged_dir)
    if not merged_path.exists():
        print("")
        print("=" * 80)
        print("✗ ERROR: Merged directory not found")
        print("=" * 80)
        print(f"Expected: {merged_dir}")
        print("=" * 80)
        sys.exit(1)
    
    merged_files = list(merged_path.rglob("*_merged.json"))
    if len(merged_files) == 0:
        print("")
        print("=" * 80)
        print("✗ ERROR: No merged files found")
        print("=" * 80)
        print(f"Directory exists but is empty: {merged_dir}")
        print("=" * 80)
        sys.exit(1)
    
    # STEP 4-1: Build/update per-utterance details with hop-level scores
    print("=" * 80)
    print("STEP 4-1: BUILDING PER-UTTERANCE DETAILS")
    print("=" * 80)
    
    from merge_seval_results import build_utterance_details_with_top_k

    # Create utterance details directory
    utterance_details_dir = Path(f"{output_base_dir}/{job_id}_utterance_hop_citedcg_scores")
    utterance_details_dir.mkdir(parents=True, exist_ok=True)
    
    vprint(f"  Experiment: {experiment}")
    vprint(f"  Reading from: {merged_dir}")
    vprint(f"  Output to: {utterance_details_dir}")
    vprint(f"  Top-k values: {k_values}")
    vprint(f"  Merged files reused: {can_reuse_merged}")
    vprint("")
    
    # Build utterance details for each experiment
    utterance_details_files = {}  # {experiment: filepath}
    # Track which files were reused {experiment: bool}
    details_files_reused = {}
    
    for exp in experiments:
        vprint(f"Processing {exp.upper()}...")
        
        exp_merged_dir = Path(merged_dir) / exp
        if not exp_merged_dir.exists():
            vprint(f"    ✗ No merged files found for {exp}")
            continue
        
        # Define output file for this experiment in the dedicated folder
        details_file = (
            utterance_details_dir /
            f"{job_id}_{exp}_utterance_details.json"
        )
        
        # Check if we can reuse existing file
        can_reuse = False
        if details_file.exists() and can_reuse_merged:
            # Load existing file to check k-values
            try:
                with open(details_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                metadata = existing_data.get("metadata", {})
                existing_k = set(metadata.get("k_values_calculated", []))
                requested_k = set(k_values)
                
                if requested_k <= existing_k:
                    # All requested k-values already calculated
                    can_reuse = True
                    utterance_details_files[exp] = str(details_file)
                    details_files_reused[exp] = True
                    vprint(f"    \u2713 Reused existing file: "
                          f"{details_file.name}")
                elif requested_k > existing_k:
                    # Need to add new k-values
                    new_k = sorted(requested_k - existing_k)
                    vprint(f"    \u27A4 Will update with new k-values: "
                          f"{new_k}")
            except Exception as e:
                logger.warning(f"Could not load existing file for {exp}: {e}")
                vprint("    \u27A4 Will regenerate (existing file unreadable)")
        
        if not can_reuse:
            # Build or update the file
            existing_file = (
                str(details_file) if details_file.exists() else None
            )
            
            try:
                details = build_utterance_details_with_top_k(
                    merged_dir=str(exp_merged_dir),
                    top_k_list=k_values,
                    output_file=str(details_file),
                    experiment=exp,
                    existing_file=existing_file
                )
                
                utterance_details_files[exp] = str(details_file)
                details_files_reused[exp] = False
                
                # Determine what action was taken
                metadata = details.get("metadata", {})
                k_calculated = metadata.get("k_values_calculated", [])
                
                if existing_file:
                    new_k = sorted(set(k_values) - set(k_calculated))
                    if new_k:
                        vprint(f"    \u2713 Updated with new k-values "
                              f"{new_k}: {details_file.name}")
                    else:
                        vprint(f"    \u2713 Regenerated: "
                              f"{details_file.name}")
                else:
                    print(f"    ✓ Generated new file: {details_file.name}")
                
            except Exception as e:
                msg = f"Failed to build utterance details for {exp}: {e}"
                logger.error(msg)
                print(f"    ✗ Error: {e}")
                details_files_reused[exp] = False
    
    vprint("")
    if utterance_details_files:
        count = len(utterance_details_files)
        vprint(f"✓ Utterance details built: {count} file(s)")
    else:
        vprint("⚠ No utterance details were created")
    print("")
    
    # STEP 4-2: Find paired utterances (only if both experiments processed)
    paired_utterances_file = None
    if len(experiments) == 2 and len(utterance_details_files) == 2:
        print("=" * 80)
        print("STEP 4-2: FINDING PAIRED UTTERANCES WITH SCORES")
        print("=" * 80)
        
        from merge_seval_results import find_paired_utterances_with_scores
        
        control_file = utterance_details_files.get("control")
        treatment_file = utterance_details_files.get("treatment")
        
        if control_file and treatment_file:
            paired_file = (
                utterance_details_dir / f"{job_id}_paired_utterances.json"
            )
            
            vprint(f"  Control: {Path(control_file).name}")
            vprint(f"  Treatment: {Path(treatment_file).name}")
            vprint(f"  Output: {paired_file.name}")
            vprint("")
            
            # Check if we can reuse existing paired file
            can_reuse_paired = False
            both_details_reused = (
                details_files_reused.get("control", False) and
                details_files_reused.get("treatment", False)
            )
            
            if paired_file.exists() and both_details_reused:
                # Both input files were reused, so paired file is still valid
                can_reuse_paired = True
                vprint("  ✓ Reused existing paired file")
                vprint("")
                
                # Load the file to show summary
                try:
                    with open(paired_file, 'r', encoding='utf-8') as f:
                        paired_data = json.load(f)
                    paired_utterances_file = str(paired_file)
                except Exception as e:
                    logger.warning(f"Could not load paired file: {e}")
                    can_reuse_paired = False
            
            if not can_reuse_paired:
                try:
                    paired_data = find_paired_utterances_with_scores(
                        control_details_file=control_file,
                        treatment_details_file=treatment_file,
                        output_file=str(paired_file)
                    )
                    paired_utterances_file = str(paired_file)
                    print(" ✓ Generated paired utterances file")
                    vprint("")
                except Exception as e:
                    logger.error(f"Failed to find paired utterances: {e}")
                    print(f"  ✗ Error: {e}")
                    paired_data = None
            
            # Print detailed summary if we have data
            if paired_data:
                metadata = paired_data.get("metadata", {})
                total = metadata.get("total_utterances", 0)
                paired = metadata.get("paired_with_scores", 0)
                control_only = metadata.get("control_only_with_scores", 0)
                treatment_only = metadata.get(
                    "treatment_only_with_scores", 0
                )
                no_scores = metadata.get("no_scores_in_either", 0)
                
                pct = 100.0 * paired / max(1, total)
                vprint("")
                vprint("  Results:")
                print(f"    Total unique queries: {total}")
                print(f"    ✓ Paired (scores in both): {paired} "
                      f"({pct:.1f}%)")
                print(f"    • Control only: {control_only}")
                print(f"    • Treatment only: {treatment_only}")
                print(f"    • No scores: {no_scores}")
                print("")
    
    # Step 5: Generate plots with plot-specific statistics
    # Always clean plot output folder first
    plots_dir = Path(f"{output_base_dir}/{job_id}_statistics_plots")
    
    if len(k_values) >= 2:
        # Ensure plots directory exists
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 80)
        print("STEP 5: GENERATING PLOTS AND PLOT-SPECIFIC STATISTICS")
        print("=" * 80)
        
        if plots_dir.exists():
            old_files = (
                list(plots_dir.rglob("*.png")) +
                list(plots_dir.rglob("*.jpg")) +
                list(plots_dir.rglob("*.json"))
            )
            if old_files:
                vprint(f"  Deleting {len(old_files)} old files from {plots_dir}")
                processor._clean_directory(plots_dir, "plots", silent=True)
        
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        vprint(f"  Experiment: {experiment}")
        vprint(f"  Top-k values: {k_values}")
        vprint(f"  Output directory: {plots_dir}")
        vprint("")
        
        try:
            from merge_seval_results import \
                generate_plot_statistics_from_utterance_details

            # Generate plot-specific statistics for each experiment and k-value
            stats_files_by_exp = {}  # {experiment: {k: filepath}}
            
            for exp in experiments:
                if exp not in utterance_details_files:
                    continue
                    
                vprint(f"  Generating plot statistics for {exp.upper()}...")
                stats_files_by_exp[exp] = {}
                
                for k in k_values:
                    # Generate plot statistics in plots folder
                    stats_file = plots_dir / f"{job_id}_{exp}_plot_stats_k{k}.json"
                    
                    try:
                        stats = generate_plot_statistics_from_utterance_details(
                            utterance_details_file=utterance_details_files[exp],
                            k_value=k,
                            output_json=str(stats_file)
                        )
                        stats_files_by_exp[exp][k] = str(stats_file)
                        vprint(f"    ✓ k={k}: {stats_file.name}")
                    except Exception as e:
                        logger.error(f"Failed to generate plot stats for {exp} k={k}: {e}")
                        vprint(f"    ✗ k={k}: Error: {e}")
                vprint("")
            
            # Generate comparison plots from statistics files
            if len(experiments) == 1:
                # Single experiment - use original 1x2 layout with all k-values
                generate_statistics_plots(
                    stats_files=stats_files_by_exp[experiments[0]],
                    output_dir=plots_dir,
                    job_id=job_id,
                    experiment=experiments[0]
                )
            else:
                # Both experiments - use 2x2 layout for comparison with all k-values
                generate_comparison_plots(
                    stats_files_control=stats_files_by_exp["control"],
                    stats_files_treatment=stats_files_by_exp["treatment"],
                    output_dir=plots_dir,
                    job_id=job_id
                )
                
                # Generate paired utterances comparison plot
                if paired_utterances_file:
                    vprint("")
                    vprint("  Generating paired utterances plot...")
                    try:
                        generate_paired_utterances_plot(
                            paired_utterances_file=paired_utterances_file,
                            output_dir=plots_dir,
                            job_id=job_id,
                            k_values=k_values
                        )
                    except Exception as e:
                        logger.error(f"Failed to generate paired utterances plot: {e}")
                        vprint(f"  ✗ Error: {e}")
            
            vprint(f"✓ Plots generated in: {plots_dir}")
        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")
            print(f"✗ Error generating plots: {e}")
            import traceback
            traceback.print_exc()
        print("")
    else:
        vprint("")
        vprint("⚠ Note: Plotting requires at least 2 different top-k values")
        vprint(f"  You provided: {k_values}")
        vprint("  Plots will not be generated")
        vprint("")
    
    # Final summary
    print("")
    print("=" * 80)
    print("✓ COMPLETE: SEVAL JOB PROCESSING WITH STATISTICS")
    print("=" * 80)
    vprint(f"Job ID: {job_id}")
    vprint(f"Experiment: {experiment}")
    vprint(f"Top-k values: {k_values}")
    vprint("")
    
    vprint("Generated files:")
    
    # Utterance details files (per experiment in dedicated folder)
    if utterance_details_files:
        vprint(f"  Utterance details folder ({utterance_details_dir}/):")
        file_count = len(utterance_details_files)
        if paired_utterances_file:
            file_count += 1  # Include paired file
        vprint(f"    {file_count} file(s)")
        for exp, filepath in utterance_details_files.items():
            filename = Path(filepath).name
            vprint(f"    - {filename}")
        if paired_utterances_file:
            filename = Path(paired_utterances_file).name
            vprint(f"    - {filename} (paired utterances)")
    
    # Plot-specific statistics and plots (in same folder)
    if len(k_values) >= 2 and plots_dir.exists():
        stats_count = len(list(plots_dir.glob("*_plot_stats_*.json")))
        plot_count = len(list(plots_dir.glob("*.png")))
        
        if stats_count > 0 or plot_count > 0:
            vprint(f"  Plots folder ({plots_dir}/):")
            if stats_count > 0:
                vprint(f"    Plot statistics: {stats_count} files")
            if plot_count > 0:
                vprint(f"    Plot images: {plot_count} files")
    
    print("=" * 80)


# ==========================================================================
# CLI Entry Point
# ==========================================================================

if __name__ == "__main__":
    # Fire automatically exposes all module-level functions as CLI commands
    # To add new commands, simply define new functions above
    fire.Fire({
        'extract_conversations': extract_conversations,
        'merge_citescg_scores': merge_citescg_scores,
        'process_seval_job_multihop_citedcg': process_seval_job_multihop_citedcg,
        'process_seval_job_with_statistics_plots': process_seval_job_with_statistics_plots,
    })
