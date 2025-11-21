#!/usr/bin/env python3
"""
SEVAL Batch Processor - Modular Architecture

Provides batch processing utilities for SEVAL data analysis with
extensible modular design.

Available Commands:
    extract_conversations           - Extract conversation details from SEVAL raw data files
    merge_citescg_scores            - Merge conversations with CiteDCG scores
    process_seval_job_multihop_citedcg - Complete end-to-end multi-hop CiteDCG processing

Usage Examples:
    # Complete multi-hop CiteDCG processing (recommended - runs all 3 steps)
    python seval_batch_processor.py process_seval_job_multihop_citedcg --job_id=130949
    
    # Extract conversations only
    python seval_batch_processor.py extract_conversations \
        --input_dir "seval_data/130949_scraping_raw_data_output" \
        --output_dir "results/conversation_details" \
        --experiment control --threads 8

    # Merge with CiteDCG scores only
    python seval_batch_processor.py merge_citescg_scores \
        --conv_dir "results/conversation_details" \
        --citedcg_control_file "results/130949_citedcg_scores_control.json" \
        --output_dir "results/merged" --top_k 5

Three-Step Workflow:
    1. Extract CiteDCG scores from metrics (via get_seval_metrics.py)
    2. Extract conversation details from raw SEVAL data
    3. Merge CiteDCG scores with conversation details
    
    The process_seval_job_multihop_citedcg command automates all three steps.
"""

import json
import logging
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
from merge_seval_results import merge_citedcg_and_calculate_stats
from seval_analysis_toolkit import SEVALAnalysisToolkit

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
    """Extract conversation details from SEVAL files."""
    
    def __init__(self):
        super().__init__()
        self.toolkit = SEVALAnalysisToolkit()
        self.multi_turn_or_hop_conversations: List[Dict] = []
        self.input_dir = None
        self.output_dir = None
    
    def process(
        self,
        input_dir: str,
        output_dir: str,
        experiment: str = "both",
        threads: int = 8,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Extract conversation details from SEVAL files.
        
        Args:
            input_dir: Directory with SEVAL JSON files
            output_dir: Directory for extracted conversation details
            experiment: Experiment type: 'control', 'treatment', or 'both' (default: 'both')
            threads: Number of parallel threads
            verbose: Enable verbose logging
            
        Returns:
            Summary statistics
        """
        self._setup_logging(verbose)
        
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not self._validate_path(input_path):
            return {}
        
        self._validate_path(output_path, create=True)
        
        # Find files based on experiment type
        files = []
        if experiment.lower() in ["control", "both"]:
            control_files = sorted(input_path.glob("control_*.json"))
            files.extend(control_files)
            if verbose:
                logger.info(f"Found {len(control_files)} control files")
        
        if experiment.lower() in ["treatment", "both"]:
            treatment_files = sorted(input_path.glob("treatment_*.json"))
            files.extend(treatment_files)
            if verbose:
                logger.info(f"Found {len(treatment_files)} treatment files")
        
        if not files:
            logger.warning(f"No {experiment} files found in {input_dir}")
            return {}
        
        self._print_header("BATCH CONVERSATION EXTRACTION")
        print(f"Input dir:        {input_dir}")
        print(f"Output dir:       {output_dir}")
        print(f"Experiment type:  {experiment}")
        print(f"Files to process: {len(files)}")
        
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
        self._generate_summary(output_path)
        
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
                str(output_file)
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
        output_file: str
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
            
            for hop in hops:
                invocations = hop.get("invocations", [])
                if invocations:  # Hop is non-empty if it has invocations
                    turn_nonempty_hops += 1
            
            total_hops += turn_total_hops
            nonempty_hops += turn_nonempty_hops
            
            hop_stats_per_turn.append({
                "total": turn_total_hops,
                "nonempty": turn_nonempty_hops
            })
        
        result = {
            "file": filename,
            "conversation_id": metadata.get("conversation_id", ""),
            "query_text": metadata.get("query_text", ""),
            "turn_count": len(turns),
            "total_hops": total_hops,
            "nonempty_hops": nonempty_hops,
            "is_multi_turn": len(turns) > 1,
            "is_multi_hop": nonempty_hops > 1,
            "hop_stats_per_turn": hop_stats_per_turn,
            "output_file": output_file,
        }
        
        # Track multi-turn OR multi-hop conversations for detailed reporting
        if result["is_multi_turn"] or result["is_multi_hop"]:
            self.multi_turn_or_hop_conversations.append(result)
        
        return result
    
    def _generate_summary(self, output_dir: Path) -> Dict[str, Any]:
        """Generate and save summary report."""
        stats = self._calculate_statistics()
        self._print_statistics(stats, self.input_dir, self.output_dir)
        self._save_reports(stats, output_dir)
        return stats
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate processing statistics."""
        total = len(self.results)
        # Count conversations with multiple turns (turn_count > 1)
        multi_turn = sum(1 for r in self.results if r.get("turn_count", 0) > 1)
        # Count conversations with multiple non-empty hops (nonempty_hops > 1)
        multi_hop = sum(1 for r in self.results if r.get("is_multi_hop"))
        
        total_hops = sum(r.get("total_hops", 0) for r in self.results)
        total_nonempty = sum(r.get("nonempty_hops", 0) for r in self.results)
        
        # Distributions
        turn_dist = {}
        total_hop_dist = {}  # Distribution of total_hops (including empty)
        nonempty_hop_dist = {}  # Distribution of nonempty_hops only
        for r in self.results:
            turns = r["turn_count"]
            total_hops_count = r.get("total_hops", 0)
            nonempty_hops_count = r.get("nonempty_hops", 0)
            
            turn_dist[turns] = turn_dist.get(turns, 0) + 1
            total_hop_dist[total_hops_count] = total_hop_dist.get(total_hops_count, 0) + 1
            nonempty_hop_dist[nonempty_hops_count] = nonempty_hop_dist.get(nonempty_hops_count, 0) + 1
        
        return {
            "total_processed": total,
            "multi_turn": multi_turn,
            "single_turn": total - multi_turn,
            "multi_hop_excl_empty": multi_hop,
            "errors": len(self.errors),
            "total_hops": total_hops,
            "total_nonempty_hops": total_nonempty,
            "avg_total_hops": total_hops / total if total > 0 else 0,
            "avg_nonempty_hops": total_nonempty / total if total > 0 else 0,
            "turn_distribution": turn_dist,
            "total_hop_distribution": total_hop_dist,
            "nonempty_hop_distribution": nonempty_hop_dist,
        }
    
    def _print_statistics(self, stats: Dict[str, Any], input_dir: str = None, output_dir: str = None):
        """Print statistics to console (always displayed regardless of verbose)."""
        print("=" * 80)
        print("SUMMARY REPORT")
        print("=" * 80)
        if input_dir:
            print(f"Input dir:  {input_dir}")
        if output_dir:
            print(f"Output dir: {output_dir}")
        if input_dir or output_dir:
            print("-" * 80)
        print(f"Total processed:             {stats['total_processed']}")
        print(f"Single-turn:                 {stats['single_turn']}")
        print(f"Multi-turn:                  {stats['multi_turn']}")
        print(f"Multi-hop (excl empty hops): {stats['multi_hop_excl_empty']}")
        print(f"Errors:                      {stats['errors']}")
        print("-" * 80)
        print("Hop Statistics:")
        print(f"  Total hops (incl empty): {stats['total_hops']}")
        print(f"  Non-empty hops:          {stats['total_nonempty_hops']}")
        print(f"  Avg total hops/conv:     {stats['avg_total_hops']:.2f}")
        print(f"  Avg nonempty hops/conv:  {stats['avg_nonempty_hops']:.2f}")
        print("-" * 80)
        
        # Turn distribution
        print("Turn Distribution:")
        for turns in sorted(stats['turn_distribution'].keys()):
            count = stats['turn_distribution'][turns]
            pct = count / stats['total_processed'] * 100
            print(f"  {turns} turn(s): {count} ({pct:.1f}%)")
        
        print("-" * 80)
        
        # Total hop distribution (including empty hops)
        print("Total Hop Distribution (including empty):")
        for hops in sorted(stats['total_hop_distribution'].keys()):
            count = stats['total_hop_distribution'][hops]
            pct = count / stats['total_processed'] * 100
            print(f"  {hops} hop(s): {count} ({pct:.1f}%)")
        
        print("-" * 80)
        
        # Nonempty hop distribution
        print("Non-empty Hop Distribution:")
        for hops in sorted(stats['nonempty_hop_distribution'].keys()):
            count = stats['nonempty_hop_distribution'][hops]
            pct = count / stats['total_processed'] * 100
            print(f"  {hops} hop(s): {count} ({pct:.1f}%)")
    
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
    """Merge conversation details with CiteDCG scores."""
    
    def __init__(self):
        super().__init__()
        self.citedcg_control_file = None
        self.citedcg_treatment_file = None
        self.all_hop_scores = []  # Track all hop scores for distribution
        self.all_hop_top_k_scores = []  # Track top-k scores
        self.top_k = 5  # Default value
        self.no_scores_count = 0  # Track files with no scores
    
    def process(
        self,
        conv_dir: str,
        citedcg_control_file: str,
        citedcg_treatment_file: str,
        output_dir: str,
        threads: int = 8,
        top_k: int = 5,
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
            top_k: Top-k value for score calculation
            verbose: Enable verbose logging
        """
        self._setup_logging(verbose)
        
        self.top_k = top_k  # Store for use in summary
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
        treatment_files = sorted(
            conv_path.glob("treatment_*_conv_details.json")
        )
        
        print(f"Found {len(control_files)} control conversations")
        print(f"Found {len(treatment_files)} treatment conversations")
        print("=" * 80)
        
        # Process control files (only if CiteDCG file provided)
        if self.citedcg_control_file:
            if control_files:
                print(f"Processing {len(control_files)} control conversations...")
                control_output = output_path / "control"
                control_output.mkdir(exist_ok=True)
                
                control_results = self._parallel_process(
                    control_files,
                    lambda f: self._process_file(
                        f, control_output, "control", top_k, verbose
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
                treatment_output = output_path / "treatment"
                treatment_output.mkdir(exist_ok=True)
                
                treatment_results = self._parallel_process(
                    treatment_files,
                    lambda f: self._process_file(
                        f, treatment_output, "treatment", top_k, verbose
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
        top_k: int,
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
                top_k=top_k
            )
            
            # Calculate hop-level statistics
            # We need to load the merged file to get accurate hop counts and scores
            with open(merged_file, "r", encoding="utf-8") as f:
                merged_content = json.load(f)
            
            total_hops = 0
            nonempty_hops = 0
            hop_scores = []
            hop_top_k_scores = []
            
            # Count and extract scores from merged data
            eval_results = merged_content.get("evaluation_data_results", {})
            total_turns = len(eval_results.get("turns", []))
            for turn in eval_results.get("turns", []):
                for hop in turn.get("hops", []):
                    total_hops += 1
                    invocations = hop.get("invocations", [])
                    if invocations:
                        nonempty_hops += 1
                        
                        # Collect individual hop scores (not turn averages)
                        hop_avg = hop.get("avg_citedcg_score")
                        hop_top_k_avg = hop.get(f"avg_top_{top_k}_citedcg_score")
                        
                        if hop_avg is not None:
                            hop_scores.append(hop_avg)
                            self.all_hop_scores.append(hop_avg)
                        if hop_top_k_avg is not None:
                            hop_top_k_scores.append(hop_top_k_avg)
                            self.all_hop_top_k_scores.append(hop_top_k_avg)
            
            avg_hop_score = (
                sum(hop_scores) / len(hop_scores) if hop_scores else 0.0
            )
            avg_hop_top_k_score = (
                sum(hop_top_k_scores) / len(hop_top_k_scores)
                if hop_top_k_scores else 0.0
            )
            
            # Track if this file has no scores
            has_scores = len(hop_scores) > 0
            if not has_scores:
                self.no_scores_count += 1
            
            return {
                "file": conv_file.name,
                "experiment": experiment_type,
                "merged_file": str(merged_file),
                "stats_file": str(stats_file),
                "total_results": stats.get("overall_statistics", {}).get(
                    "results_with_citedcg", 0
                ),
                "total_turns": total_turns,
                "total_hops": total_hops,
                "nonempty_hops": nonempty_hops,
                "hop_scores": hop_scores,
                "hop_top_k_scores": hop_top_k_scores,
                "avg_hop_score": avg_hop_score,
                "avg_hop_top_k_score": avg_hop_top_k_score,
                "has_scores": has_scores,
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
        
        print("")
        print("=" * 80)
        print("MERGE SUMMARY")
        print("="* 80)
        print(f"Total processed: {total}")
        print(f"With scores:     {total - self.no_scores_count}")
        print(f"No scores:       {self.no_scores_count}")
        print(f"Errors:          {len(self.errors)}")
        
        if total > 0:
            control = sum(
                1 for r in self.results if r.get("experiment") == "control"
            )
            treatment = sum(
                1 for r in self.results
                if r.get("experiment") == "treatment"
            )
            
            print(f"Control:         {control}")
            print(f"Treatment:       {treatment}")
            
            total_results = sum(
                r.get("total_results", 0) for r in self.results
            )
            total_hops = sum(
                r.get("total_hops", 0) for r in self.results
            )
            total_nonempty_hops = sum(
                r.get("nonempty_hops", 0) for r in self.results
            )
            
            print(f"Total scored results: {total_results}")
            print(f"Total hops (incl empty): {total_hops}")
            print(f"Non-empty hops:       {total_nonempty_hops}")
            
            # Turn distribution across files
            turn_dist = {}
            for result in self.results:
                turn_count = result.get("total_turns", 0)
                turn_dist[turn_count] = turn_dist.get(turn_count, 0) + 1
            
            if turn_dist:
                print("Turn distribution:")
                for turn_count in sorted(turn_dist.keys()):
                    file_count = turn_dist[turn_count]
                    print(f"  {turn_count} turns: {file_count} files")
            
            # Calculate hop score statistics (only from nonempty hops)
            if self.all_hop_scores:
                avg_hop_score = (
                    sum(self.all_hop_scores) / len(self.all_hop_scores)
                )
                avg_hop_top_k_score = (
                    sum(self.all_hop_top_k_scores) / len(self.all_hop_top_k_scores)
                    if self.all_hop_top_k_scores else 0.0
                )
                
                # Build hop score distribution
                hop_score_dist = {}
                for score in self.all_hop_scores:
                    rounded_score = round(score, 1)
                    hop_score_dist[rounded_score] = (
                        hop_score_dist.get(rounded_score, 0) + 1
                    )
                
                # Build top-k hop score distribution
                hop_top_k_score_dist = {}
                for score in self.all_hop_top_k_scores:
                    rounded_score = round(score, 1)
                    hop_top_k_score_dist[rounded_score] = (
                        hop_top_k_score_dist.get(rounded_score, 0) + 1
                    )
                
                print(f"Avg hop CiteDCG:      {avg_hop_score:.4f}")
                if self.all_hop_top_k_scores:
                    print(f"Avg hop top-{self.top_k} CiteDCG: {avg_hop_top_k_score:.4f}")
                print("-" * 80)
                print(f"Hop-Average Score Distribution ({len(self.all_hop_scores)} non-empty hops):")
                for score in sorted(hop_score_dist.keys()):
                    count = hop_score_dist[score]
                    pct = count / len(self.all_hop_scores) * 100
                    print(f"  {score:.1f}: {count} hops ({pct:.1f}%)")
                
                if self.all_hop_top_k_scores:
                    print(f"\nHop-Average Top-{self.top_k} Score Distribution ({len(self.all_hop_top_k_scores)} non-empty hops):")
                    for score in sorted(hop_top_k_score_dist.keys()):
                        count = hop_top_k_score_dist[score]
                        pct = count / len(self.all_hop_top_k_scores) * 100
                        print(f"  {score:.1f}: {count} hops ({pct:.1f}%)")
                print("-" * 80)
                
                # File-based statistics
                print("File-based Hop Statistics:")
                
                # Group by total/nonempty hop counts
                hop_count_dist = {}
                for result in self.results:
                    total_h = result.get("total_hops", 0)
                    nonempty_h = result.get("nonempty_hops", 0)
                    key = f"{total_h} total, {nonempty_h} non-empty"
                    if key not in hop_count_dist:
                        hop_count_dist[key] = {
                            "count": 0,
                            "hop_scores": [],
                            "hop_top_k_scores": []
                        }
                    hop_count_dist[key]["count"] += 1
                    hop_count_dist[key]["hop_scores"].extend(
                        result.get("hop_scores", [])
                    )
                    hop_count_dist[key]["hop_top_k_scores"].extend(
                        result.get("hop_top_k_scores", [])
                    )
                
                for hop_config in sorted(hop_count_dist.keys()):
                    data = hop_count_dist[hop_config]
                    file_count = data["count"]
                    scores = data["hop_scores"]
                    top_k_scores = data["hop_top_k_scores"]
                    avg_score = (
                        sum(scores) / len(scores) if scores else 0.0
                    )
                    avg_top_k_score = (
                        sum(top_k_scores) / len(top_k_scores)
                        if top_k_scores else 0.0
                    )
                    print(
                        f"  {hop_config} hops: {file_count} files "
                        f"(avg: {avg_score:.4f}, top-{self.top_k}: {avg_top_k_score:.4f})"
                    )
            else:
                print("No hop scores available")
    
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
    verbose: bool = False
):
    """
    Extract conversation details from SEVAL files.
    
    Args:
        input_dir: Directory with SEVAL JSON files
        output_dir: Output directory for extracted details
        experiment: Experiment type: 'control', 'treatment', or 'both' (default: 'both')
        threads: Number of parallel threads (default: 8)
        verbose: Enable verbose logging
        
    Example:
        python seval_batch_processor.py extract_conversations
        python seval_batch_processor.py extract_conversations --experiment control
        python seval_batch_processor.py extract_conversations --experiment treatment
        python seval_batch_processor.py extract_conversations --experiment both --threads 16
    """
    extractor = ConversationExtractor()
    return extractor.process(input_dir, output_dir, experiment, threads, verbose)


def merge_citescg_scores(
    conv_dir: str = r"results\conversation_details",
    citedcg_control_file: str = "",
    citedcg_treatment_file: str = "",
    output_dir: str = r"results\merged",
    threads: int = 8,
    top_k: int = 5,
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
        top_k: Top-k value for score calculation
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
    print(f"Top-k:                 {top_k}")
    print(f"Threads:               {threads}")
    print("=" * 80)
    
    processor.process(
        conv_dir=conv_dir,
        citedcg_control_file=citedcg_control_file,
        citedcg_treatment_file=citedcg_treatment_file,
        output_dir=output_dir,
        threads=threads,
        top_k=top_k,
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
    merged_dir = f"{output_base_dir}/{job_id}_merged"
    
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
        print(f"STEP 1/{len(experiments)}: Extracting CiteDCG scores ({exp})...")
        citedcg_file = f"{citedcg_dir}/{job_id}_citedcg_scores_{exp}.json"
        
        extract_per_result_citedcg(
            metrics_folder=metrics_dir,  # Pass just folder name, not full path
            experiment=exp,
            output_file=citedcg_file
        )
        
        citedcg_files[exp] = citedcg_file
        print(f"✓ CiteDCG scores saved to: {citedcg_file}")
        print("")
    
    # Step 2: Extract conversation details
    print(f"STEP 2: Extracting conversation details...")
    extract_conversations(
        input_dir=raw_data_dir,
        output_dir=conv_dir,
        experiment=experiment,
        threads=threads,
        verbose=verbose
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
        top_k=top_k,
        verbose=verbose
    )
    print("")
    
    print("=" * 80)
    print("✓ SEVAL JOB PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Merged results: {merged_dir}")
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
    })
