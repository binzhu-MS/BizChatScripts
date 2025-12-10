#!/usr/bin/env python3
"""
Plot SEVAL metrics for CiteDCG analysis.

This script generates four types of visualizations:
1. Hop distribution across conversations
2. Average and top-k CiteDCG score curves per utterance
3. Score changes between adjacent hops per utterance
4. Average scores across hops for each utterance

Key Features:
- Flexible support for any number of top-k values (not limited to k=1,3,5)
- Validates directory consistency (same files across all k values)
- Sparse data handling (only plots existing transitions)
- Comprehensive sanity checks with diagnostic output

Requirements:
- All top-k directories must have the same conversation files
- Currently focuses on single-turn conversations with >1 non-empty hop
- Directory names must follow pattern: *_k<N> (e.g., job_merged_k1)

Usage:
    # Via batch processor (recommended)
    python seval_batch_processor.py process_seval_job_with_plots \\
        --job_id=132951 --top_k_list=1,3,5 --generate_plots=True
    
    # Direct invocation
    python plot_seval_metrics.py plot_seval_metrics \\
        --merged_dirs='["results/132951_merged_k1", "results/132951_merged_k3"]' \\
        --experiment="control"
"""

import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SEvalMetricsPlotter:
    """Plot CiteDCG metrics from merged SEVAL results."""
    
    def __init__(
        self,
        merged_dirs: Dict[int, str],
        output_dir: str = None
    ):
        """
        Initialize the plotter.
        
        Args:
            merged_dirs: Dictionary mapping k values to merged result directories
                        e.g., {1: "results/job_merged_k1", 3: "results/job_merged_k3"}
            output_dir: Directory to save plots (defaults to parent of first dir + /plots)
        
        Raises:
            ValueError: If any directory doesn't exist or is empty
        """
        self.merged_dirs = {}
        
        # Validate and convert directories
        for k, dir_path in merged_dirs.items():
            dir_path = Path(dir_path)
            if not dir_path.exists():
                raise ValueError(f"Directory for k={k} does not exist: {dir_path}")
            if not any(dir_path.iterdir()):
                raise ValueError(f"Directory for k={k} is empty: {dir_path}")
            self.merged_dirs[k] = dir_path
        
        if not self.merged_dirs:
            raise ValueError("No merged directories provided")
        
        # Use first directory for output
        first_dir = next(iter(self.merged_dirs.values()))
        base_dir = first_dir.parent
        self.output_dir = Path(output_dir) if output_dir else base_dir / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage: {conversation_id: {k: stats_data}}
        self.conversations = {}
        self.single_turn_convs = {}
        self.plotted_convs = {}  # Filtered conversations for plotting
        
        # Store sorted list of k values for plotting
        self.k_values = sorted(self.merged_dirs.keys())
    
    def _validate_directory_consistency(self, experiment: str = "control"):
        """
        Validate that all top-k directories have the same files.
        
        Args:
            experiment: Experiment type (control or treatment)
        
        Returns:
            bool: True if validation passed, False otherwise
        """
        print("\n" + "="*70)
        print("SANITY CHECK: Directory Consistency")
        print("="*70)
        
        # Collect file information from each directory
        files_by_k = {}
        for k, merged_dir in self.merged_dirs.items():
            exp_dir = merged_dir / experiment
            if not exp_dir.exists():
                print(f"\n✗ FAILED: Experiment directory not found: {exp_dir}")
                print("="*70 + "\n")
                return False
            
            stats_files = set(f.name for f in exp_dir.glob("*_stats.json"))
            files_by_k[k] = stats_files
            print(f"  @{k}: {len(stats_files)} files in {exp_dir}")
        
        # Compare file counts
        file_counts = [len(files) for files in files_by_k.values()]
        if len(set(file_counts)) > 1:
            print("\n✗ FAILED: Directories have different number of files")
            for k, files in files_by_k.items():
                print(f"    @{k}: {len(files)} files")
            print("  This indicates incomplete processing.")
            print("="*70 + "\n")
            return False
        
        # Compare file names across all k values
        k_values = sorted(files_by_k.keys())
        reference_k = k_values[0]
        reference_files = files_by_k[reference_k]
        
        all_match = True
        for k in k_values[1:]:
            current_files = files_by_k[k]
            
            # Files in reference but not in current
            missing = reference_files - current_files
            # Files in current but not in reference
            extra = current_files - reference_files
            
            if missing or extra:
                all_match = False
                print(f"\n✗ FAILED: File mismatch between @{reference_k} and @{k}")
                if missing:
                    print(f"    Files in @{reference_k} but not @{k}:")
                    for f in sorted(missing)[:5]:  # Show first 5
                        print(f"      - {f}")
                    if len(missing) > 5:
                        print(f"      ... and {len(missing)-5} more")
                if extra:
                    print(f"    Files in @{k} but not @{reference_k}:")
                    for f in sorted(extra)[:5]:  # Show first 5
                        print(f"      - {f}")
                    if len(extra) > 5:
                        print(f"      ... and {len(extra)-5} more")
        
        if all_match:
            print(f"\n✓ PASSED: All {len(k_values)} directories have identical file names")
            print(f"  Total files per directory: {len(reference_files)}")
            print("="*70 + "\n")
            return True
        else:
            print("\n  This indicates inconsistent processing.")
            print("="*70 + "\n")
            return False
        
    def load_conversations(self, experiment: str = "control"):
        """
        Load all conversation stats files from all top-k runs.
        
        Args:
            experiment: Experiment type (control or treatment)
        
        Raises:
            ValueError: If directory validation fails
        """
        # First, validate that all directories have the same files
        if not self._validate_directory_consistency(experiment):
            raise ValueError(
                "Directory consistency validation failed. "
                "Cannot proceed with loading conversations."
            )
        
        # Load from each top-k directory
        for k, merged_dir in self.merged_dirs.items():
            exp_dir = merged_dir / experiment
            if not exp_dir.exists():
                logger.warning(f"Experiment directory not found: {exp_dir}")
                continue
            
            stats_files = list(exp_dir.glob("*_stats.json"))
            logger.info(
                f"Found {len(stats_files)} stats files in {exp_dir} (top-{k})"
            )
            
            for stats_file in stats_files:
                try:
                    with open(stats_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # Get conversation ID
                    conv_id = data['conversation_metadata']['conversation_id']
                    
                    # Initialize conversation entry if first time
                    if conv_id not in self.conversations:
                        self.conversations[conv_id] = {}
                    
                    # Store data for this top-k
                    self.conversations[conv_id][k] = data
                    
                except Exception as e:
                    logger.error(f"Error loading {stats_file}: {e}")
        
        # Filter single-turn conversations
        for conv_id, k_data in self.conversations.items():
            # Use any k to check turn count (should be same across all)
            first_k = next(iter(k_data.values()))
            if first_k['overall_statistics']['total_turns'] == 1:
                self.single_turn_convs[conv_id] = k_data
        
        logger.info(
            f"Loaded {len(self.conversations)} total conversations, "
            f"{len(self.single_turn_convs)} single-turn"
        )
        
        # Filter for plotting: single-turn with >1 non-empty hop
        self._filter_conversations_for_plotting()
    
    def _filter_conversations_for_plotting(self):
        """
        Filter conversations that meet plotting requirements.
        
        Requirements:
        - Single-turn conversations only
        - More than 1 non-empty hop
        - Have data for all required top-k values
        
        This ensures all plots use the same set of conversations.
        """
        for conv_id, k_data in self.single_turn_convs.items():
            # Check if all required top-k data exists
            if not all(k in k_data for k in self.merged_dirs.keys()):
                logger.warning(
                    f"Skipping {conv_id[:8]}: missing top-k data"
                )
                continue
            
            # Use any k to count non-empty hops (should be same across all)
            data = next(iter(k_data.values()))
            
            # Count non-empty hops
            nonempty_hops = 0
            for turn in data['turn_statistics']:
                for hop in turn['hops']:
                    if hop['results_with_scores'] > 0:
                        nonempty_hops += 1
            
            # Only include if >1 non-empty hop
            if nonempty_hops > 1:
                self.plotted_convs[conv_id] = k_data
        
        logger.info(
            f"Filtered for plotting: {len(self.plotted_convs)} conversations "
            f"(single-turn with >1 non-empty hop)"
        )
    
    def plot_hop_distribution(self):
        """
        Plot 1: Hop distribution across conversations.
        
        Shows:
        - Total hops distribution (including empty hops)
        - Non-empty hops distribution
        
        Uses data from any top-k run (hop counts are the same).
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Collect hop counts (use data from any k, they're all the same)
        total_hops = []
        nonempty_hops = []
        
        for conv_id, k_data in self.plotted_convs.items():
            # Use first available k
            data = next(iter(k_data.values()))
            stats = data['overall_statistics']
            total_hops.append(stats['total_hops'])
            
            # Count non-empty hops
            nonempty = 0
            for turn in data['turn_statistics']:
                for hop in turn['hops']:
                    if hop['results_with_scores'] > 0:
                        nonempty += 1
            nonempty_hops.append(nonempty)
        
        # Plot total hops distribution
        if total_hops:
            hop_counts = defaultdict(int)
            for h in total_hops:
                hop_counts[h] += 1
            
            hops = sorted(hop_counts.keys())
            counts = [hop_counts[h] for h in hops]
            
            ax1.bar(hops, counts, color='steelblue', alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Number of Total Hops', fontsize=12)
            ax1.set_ylabel('Number of Conversations', fontsize=12)
            ax1.set_title('Total Hop Distribution (including empty hops)', fontsize=14, fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)
            
            # Add count labels on bars
            for hop, count in zip(hops, counts):
                ax1.text(hop, count, str(count), ha='center', va='bottom', fontsize=10)
        
        # Plot non-empty hops distribution
        if nonempty_hops:
            hop_counts = defaultdict(int)
            for h in nonempty_hops:
                hop_counts[h] += 1
            
            hops = sorted(hop_counts.keys())
            counts = [hop_counts[h] for h in hops]
            
            ax2.bar(hops, counts, color='coral', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Number of Non-Empty Hops', fontsize=12)
            ax2.set_ylabel('Number of Conversations', fontsize=12)
            ax2.set_title('Non-Empty Hop Distribution', fontsize=14, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            
            # Add count labels on bars
            for hop, count in zip(hops, counts):
                ax2.text(hop, count, str(count), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        output_file = self.output_dir / "hop_distribution.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved hop distribution plot to {output_file}")
        plt.close()
    
    def plot_score_curves(self):
        """
        Plot 2: CiteDCG score curves per hop.
        
        Shows average and top-k scores for each utterance across hops.
        Creates subplots for average + each k value.
        X-axis: Utterance index
        Y-axis: CiteDCG Score
        Different colored bars/lines represent different hops.
        """
        # Determine number of subplots needed (avg + each k value)
        num_plots = 1 + len(self.k_values)  # average + top-k scores
        nrows = (num_plots + 1) // 2  # 2 columns
        ncols = 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 6*nrows))
        axes = axes.flatten() if num_plots > 1 else [axes]
        
        num_convs = len(self.plotted_convs)
        conv_ids = list(self.plotted_convs.keys())
        x_positions = np.arange(num_convs)
        
        # Collect data for all utterances
        # avg_by_hop: hop_num -> [scores for each utterance]
        # scores_by_k_by_hop: {k: {hop_num: [scores]}}
        avg_by_hop = defaultdict(list)
        scores_by_k_by_hop = {k: defaultdict(list) for k in self.k_values}
        max_hops = 0
        
        # Use the highest k value for average scores
        highest_k = max(self.k_values)
        
        for conv_id in conv_ids:
            k_data = self.plotted_convs[conv_id]
            
            # Check if all required k data exists
            if not all(k in k_data for k in self.k_values):
                logger.warning(
                    f"Missing top-k data for conversation {conv_id[:8]}"
                )
                continue
            
            # Collect average scores (from highest k)
            data_highest = k_data[highest_k]
            turn_highest = data_highest['turn_statistics'][0]
            for hop in turn_highest['hops']:
                if hop['results_with_scores'] > 0:
                    hop_num = hop['hop_number']
                    max_hops = max(max_hops, hop_num)
                    avg_by_hop[hop_num].append(hop.get('avg_citedcg_score', 0))
            
            # Collect scores for each k value
            for k in self.k_values:
                data_k = k_data[k]
                turn_k = data_k['turn_statistics'][0]
                for hop in turn_k['hops']:
                    if hop['results_with_scores'] > 0:
                        hop_num = hop['hop_number']
                        score_key = f'avg_top_{k}_citedcg_score'
                        scores_by_k_by_hop[k][hop_num].append(
                            hop.get(score_key, 0)
                        )
        
        # Create bar plots grouped by hop
        bar_width = 0.8 / max_hops if max_hops > 0 else 0.8
        colors_hop = plt.cm.Set3(np.linspace(0, 1, max_hops))
        conv_labels = [cid[:8] for cid in conv_ids]
        
        # Plot average scores
        ax = axes[0]
        for hop_num in sorted(avg_by_hop.keys()):
            scores = avg_by_hop[hop_num]
            offset = (hop_num - 1) * bar_width - (max_hops - 1) * bar_width / 2
            ax.bar(
                x_positions[:len(scores)] + offset, scores,
                width=bar_width, label=f'Hop {hop_num}',
                color=colors_hop[hop_num - 1], alpha=0.8, edgecolor='black'
            )
        ax.set_xlabel('Utterance', fontsize=12)
        ax.set_ylabel('CiteDCG Score', fontsize=12)
        ax.set_title('Average CiteDCG Score', fontsize=14, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(conv_labels, rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        
        # Plot scores for each k value
        for idx, k in enumerate(self.k_values):
            ax = axes[idx + 1]
            for hop_num in sorted(scores_by_k_by_hop[k].keys()):
                scores = scores_by_k_by_hop[k][hop_num]
                offset = (hop_num - 1) * bar_width - (max_hops - 1) * bar_width / 2
                ax.bar(
                    x_positions[:len(scores)] + offset, scores,
                    width=bar_width, label=f'Hop {hop_num}',
                    color=colors_hop[hop_num - 1], alpha=0.8, edgecolor='black'
                )
            ax.set_xlabel('Utterance', fontsize=12)
            ax.set_ylabel('CiteDCG Score', fontsize=12)
            ax.set_title(f'Top-{k} CiteDCG Score', fontsize=14, fontweight='bold')
            ax.set_xticks(x_positions)
            ax.set_xticklabels(conv_labels, rotation=45, ha='right', fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            ax.legend(fontsize=8, loc='best')
        
        # Hide any unused subplots
        for idx in range(num_plots, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        output_file = self.output_dir / "score_curves_per_utterance.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved score curves plot to {output_file}")
        plt.close()
    
    def plot_adjacent_hop_changes(self):
        """
        Plot 3: Score changes between adjacent hops.
        
        Shows the delta in top-k scores between consecutive non-empty hops.
        Creates one subplot per k value.
        X-axis: Utterance
        Y-axis: Score change (delta) between hops
        Different hop transitions shown with different markers and colors.
        """
        num_k = len(self.k_values)
        fig, axes = plt.subplots(num_k, 1, figsize=(16, 5*num_k))
        axes = [axes] if num_k == 1 else axes
        
        num_convs = len(self.plotted_convs)
        conv_ids = list(self.plotted_convs.keys())
        x_positions = np.arange(num_convs)
        
        # First pass: collect all possible transitions and conversation deltas
        conv_deltas = {}  # {conv_id: {transition: {k: delta}}}
        all_transitions = set()
        
        for conv_id in conv_ids:
            k_data = self.plotted_convs[conv_id]
            
            # Check if all required k data exists
            if not all(k in k_data for k in self.k_values):
                conv_deltas[conv_id] = {}
                continue
            
            # Extract scores for each k value
            scores_by_k = {}
            hop_nums = None
            
            for k in self.k_values:
                data_k = k_data[k]
                turn_k = data_k['turn_statistics'][0]
                
                k_scores = []
                k_hop_nums = []
                
                for hop in turn_k['hops']:
                    if hop['results_with_scores'] > 0:
                        k_hop_nums.append(hop['hop_number'])
                        score_key = f'avg_top_{k}_citedcg_score'
                        k_scores.append(hop.get(score_key, 0))
                
                scores_by_k[k] = k_scores
                if hop_nums is None:
                    hop_nums = k_hop_nums
            
            if hop_nums is None or len(hop_nums) < 2:
                # No deltas for single hop or no hops
                conv_deltas[conv_id] = {}
                continue
            
            # Calculate deltas between adjacent non-empty hops
            # Use sequential position (1st, 2nd, 3rd non-empty hop) not hop numbers
            conv_transitions = {}
            for i in range(len(hop_nums)-1):
                # Transition between i-th and (i+1)-th non-empty hop
                transition = f"{i+1}\u2192{i+2}"
                all_transitions.add(transition)
                
                # Calculate delta for each k
                transition_deltas = {}
                for k in self.k_values:
                    if len(scores_by_k[k]) > i+1:
                        transition_deltas[k] = scores_by_k[k][i+1] - scores_by_k[k][i]
                
                conv_transitions[transition] = transition_deltas
            
            conv_deltas[conv_id] = conv_transitions
        
        # Second pass: build arrays for each k value
        deltas_by_k_by_transition = {
            k: defaultdict(list) for k in self.k_values
        }
        # Track x positions for each transition (only for conversations that have it)
        x_positions_by_transition = defaultdict(list)
        
        for idx, conv_id in enumerate(conv_ids):
            conv_transitions = conv_deltas.get(conv_id, {})
            for transition in sorted(all_transitions):
                if transition in conv_transitions:
                    for k in self.k_values:
                        if k in conv_transitions[transition]:
                            deltas_by_k_by_transition[k][transition].append(
                                conv_transitions[transition][k]
                            )
                    # Track x position once per transition
                    if transition not in x_positions_by_transition or \
                       idx not in x_positions_by_transition[transition]:
                        x_positions_by_transition[transition].append(idx)
                # If conversation doesn't have this transition, don't add anything
        
        # Create dot plots with different markers and colors for transitions
        transitions = sorted(all_transitions)
        
        # Define distinct markers and colors for different transitions
        markers = ['o', 'x', '^', 'D', 'v', 's', 'p', '*', 'h', '<', '>']
        colors_list = [
            'red', 'blue', 'green', 'black', 'purple',
            'darkred', 'darkblue', 'darkgreen', 'brown', 'navy'
        ]
        
        x_positions = np.arange(num_convs)
        conv_labels = [cid[:8] for cid in conv_ids]
        
        # Plot deltas for each k value
        for k_idx, k in enumerate(self.k_values):
            ax = axes[k_idx]
            for trans_idx, transition in enumerate(transitions):
                deltas = deltas_by_k_by_transition[k][transition]
                x_pos = x_positions_by_transition[transition]
                marker = markers[trans_idx % len(markers)]
                color = colors_list[trans_idx % len(colors_list)]
                # Only add edgecolor for filled markers
                if marker in ['x', '+', '|', '_']:
                    ax.scatter(
                        x_pos, deltas,
                        marker=marker, s=80, label=transition,
                        color=color, alpha=0.7, linewidth=1.5
                    )
                else:
                    ax.scatter(
                        x_pos, deltas,
                        marker=marker, s=80, label=transition,
                        color=color, alpha=0.7, edgecolor='black',
                        linewidth=0.8
                    )
        
        # Sanity check: count dots and distribution of non-empty hops
        print("\n" + "="*70)
        print("SANITY CHECK: Adjacent Hop Changes Plot")
        print("="*70)
        
        # Count total dots plotted for each k
        print("Total data points plotted:")
        total_dots_by_k = {}
        for k in self.k_values:
            total_dots = sum(
                len(deltas_by_k_by_transition[k][t]) for t in transitions
            )
            total_dots_by_k[k] = total_dots
            print(f"  @{k}: {total_dots} dots")
        
        # Count dots per transition (use first k value for counting)
        print("\nDots per transition:")
        first_k = self.k_values[0]
        for transition in transitions:
            count = len(deltas_by_k_by_transition[first_k][transition])
            print(f"  {transition}: {count} utterances")
        
        # Distribution of non-empty hops (use first k value)
        hop_count_distribution = defaultdict(int)
        for conv_id in conv_ids:
            k_data = self.plotted_convs[conv_id]
            data_first_k = k_data.get(first_k)
            if data_first_k:
                turn_first_k = data_first_k['turn_statistics'][0]
                num_nonempty_hops = sum(
                    1 for hop in turn_first_k['hops']
                    if hop['results_with_scores'] > 0
                )
                hop_count_distribution[num_nonempty_hops] += 1
        
        print("\nDistribution of non-empty hops:")
        for num_hops in sorted(hop_count_distribution.keys()):
            count = hop_count_distribution[num_hops]
            print(f"  {num_hops} non-empty hops: {count} utterances")
        
        # Expected total transitions
        expected_transitions = sum(
            max(0, num_hops - 1) * count
            for num_hops, count in hop_count_distribution.items()
        )
        print(f"\nExpected total transitions: {expected_transitions}")
        # Use first k value for comparison
        first_k = self.k_values[0]
        print(f"Actual total dots plotted (@{first_k}): {total_dots_by_k[first_k]}")
        if expected_transitions == total_dots_by_k[first_k]:
            print("✓ Sanity check PASSED: Expected = Actual")
        else:
            print(
                f"✗ Sanity check FAILED: "
                f"Expected ({expected_transitions}) != Actual ({total_dots_by_k[first_k]})"
            )
        print("="*70 + "\n")
        
        # Configure subplots
        for k_idx, k in enumerate(self.k_values):
            ax = axes[k_idx]
            title = f'Top-{k} Score Change Between Adjacent Non-Empty Hops'
            ax.set_xlabel('Utterance', fontsize=12)
            ax.set_ylabel('Score Delta', fontsize=12)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_xticks(x_positions)
            ax.set_xticklabels(
                conv_labels, rotation=45, ha='right', fontsize=8
            )
            ax.axhline(
                y=0, color='black', linestyle='--', linewidth=1, alpha=0.5
            )
            ax.grid(axis='y', alpha=0.3)
            ax.legend(
                title='Non-empty hop transition',
                fontsize=8, loc='best', title_fontsize=8
            )
        
        plt.tight_layout()
        output_file = self.output_dir / "adjacent_hop_score_changes.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved adjacent hop changes plot to {output_file}")
        plt.close()
    
    def plot_average_scores_across_hops(self):
        """
        Plot 4: Average @k scores across all hops for each utterance.
        
        For each utterance, calculates the average score across all its hops
        for each k value, then displays in separate subplots.
        X-axis: Utterance
        Y-axis: Average CiteDCG Score
        """
        num_k = len(self.k_values)
        fig, axes = plt.subplots(num_k, 1, figsize=(16, 5*num_k))
        axes = [axes] if num_k == 1 else axes
        
        num_convs = len(self.plotted_convs)
        conv_ids = list(self.plotted_convs.keys())
        x_positions = np.arange(num_convs)
        
        # Collect average scores and standard deviations for each utterance and each k
        avg_scores_by_k = {k: [] for k in self.k_values}
        std_scores_by_k = {k: [] for k in self.k_values}
        
        for conv_id in conv_ids:
            k_data = self.plotted_convs[conv_id]
            
            # Check if all required k data exists
            if not all(k in k_data for k in self.k_values):
                for k in self.k_values:
                    avg_scores_by_k[k].append(0)
                    std_scores_by_k[k].append(0)
                continue
            
            # Calculate average and std dev for each k value
            for k in self.k_values:
                data_k = k_data[k]
                turn_k = data_k['turn_statistics'][0]
                
                k_scores = []
                for hop in turn_k['hops']:
                    if hop['results_with_scores'] > 0:
                        score_key = f'avg_top_{k}_citedcg_score'
                        k_scores.append(hop.get(score_key, 0))
                
                # Calculate average and standard deviation across hops
                if k_scores:
                    avg_scores_by_k[k].append(np.mean(k_scores))
                    std_scores_by_k[k].append(np.std(k_scores))
                else:
                    avg_scores_by_k[k].append(0)
                    std_scores_by_k[k].append(0)
        
        # Create bar charts with error bars in separate subplots
        conv_labels = [cid[:8] for cid in conv_ids]
        colors = ['steelblue', 'coral', 'mediumseagreen', 'gold', 'orchid', 'tomato']
        
        for idx, k in enumerate(self.k_values):
            ax = axes[idx]
            color = colors[idx % len(colors)]
            
            ax.bar(
                x_positions, avg_scores_by_k[k],
                yerr=std_scores_by_k[k],  # Add error bars for standard deviation
                color=color, alpha=0.8, edgecolor='black',
                capsize=3, error_kw={'elinewidth': 1, 'ecolor': 'black', 'alpha': 0.7}
            )
            ax.set_xlabel('Utterance', fontsize=12)
            ax.set_ylabel(f'Average @{k} Score ± Std Dev', fontsize=12)
            ax.set_title(f'Average @{k} Across Hops (with Standard Deviation)', fontsize=14, fontweight='bold')
            ax.set_xticks(x_positions)
            ax.set_xticklabels(conv_labels, rotation=45, ha='right', fontsize=8)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / "average_scores_across_hops.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved average scores plot to {output_file}")
        plt.close()
    
    def print_plotting_statistics(self):
        """Print statistics about which utterances are included in plots."""
        total_convs = len(self.conversations)
        single_turn_convs = len(self.single_turn_convs)
        multi_turn_convs = total_convs - single_turn_convs
        plotted_convs = len(self.plotted_convs)
        excluded_single_hop = single_turn_convs - plotted_convs
        
        print("")
        print("=" * 80)
        print("PLOTTING STATISTICS")
        print("=" * 80)
        print(f"Total conversations loaded:          {total_convs}")
        print(f"  Single-turn conversations:         {single_turn_convs}")
        print(f"  Multi-turn conversations:          {multi_turn_convs}")
        print("")
        print("Single-turn conversation analysis:")
        print(f"  Included in plots (>1 non-empty):  {plotted_convs}")
        print(f"  Excluded (≤1 non-empty hop):       {excluded_single_hop}")
        print("")
        print("Exclusion criteria:")
        print("  - Multi-turn conversations excluded (current implementation)")
        print("  - Single-turn with ≤1 non-empty hop excluded")
        print("")
        print(f"Total utterances plotted:            {plotted_convs}")
        print("=" * 80)
        print("")
    
    def generate_all_plots(self, experiment: str = "control"):
        """
        Generate all plots for an experiment.
        
        Args:
            experiment: Experiment type (control or treatment)
        """
        logger.info(f"Generating plots for experiment: {experiment}")
        
        # Load data
        self.load_conversations(experiment=experiment)
        
        if not self.plotted_convs:
            logger.warning(
                "No conversations meet plotting requirements. Skipping plots."
            )
            return
        
        # Print statistics
        self.print_plotting_statistics()
        
        # Generate plots
        try:
            logger.info("Plot 1: Hop distribution...")
            self.plot_hop_distribution()
        except Exception as e:
            logger.error(f"Error generating hop distribution plot: {e}")
        
        try:
            logger.info("Plot 2: Score curves per utterance...")
            self.plot_score_curves()
        except Exception as e:
            logger.error(f"Error generating score curves plot: {e}")
        
        try:
            logger.info("Plot 3: Adjacent hop score changes...")
            self.plot_adjacent_hop_changes()
        except Exception as e:
            logger.error(f"Error generating adjacent hop changes plot: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            logger.info("Plot 4: Average scores across hops...")
            self.plot_average_scores_across_hops()
        except Exception as e:
            logger.error(f"Error generating average scores plot: {e}")
            import traceback
            traceback.print_exc()
        
        logger.info(f"All plots saved to {self.output_dir}")


def plot_seval_metrics(
    merged_dirs: List[str],
    experiment: str = "control",
    output_dir: str = None
):
    """
    Generate all SEVAL metric plots from merged results directories.
    
    Args:
        merged_dirs: List of merged result directory paths.
                    Directory names should end with '_k<N>' where N is the top-k value.
                    e.g., ["results/job_merged_k1", "results/job_merged_k3", "results/job_merged_k5"]
                    Supports any number of k values and any k values (not limited to 1,3,5).
        experiment: Experiment type (control or treatment)
        output_dir: Directory to save plots (optional)
    
    Example:
        # Run batch processor with multiple top_k values
        # --top_k_list=1,3,5 creates results/JOB_merged_k1, k3, k5
        
        python plot_seval_metrics.py plot \\
            --merged_dirs='["results/130949_merged_k1", "results/130949_merged_k3", "results/130949_merged_k5"]' \\
            --experiment="control"
        
        # Or use custom k values like 1,2,10
        python plot_seval_metrics.py plot \\
            --merged_dirs='["results/job_merged_k1", "results/job_merged_k2", "results/job_merged_k10"]' \\
            --experiment="control"
    """
    # Parse k values from directory names
    k_to_dir = {}
    for dir_path in merged_dirs:
        dir_name = Path(dir_path).name
        # Extract k value from directory name (e.g., "job_merged_k3" -> 3)
        if '_k' in dir_name:
            try:
                k_value = int(dir_name.split('_k')[-1])
                k_to_dir[k_value] = dir_path
            except ValueError:
                logger.warning(
                    f"Could not parse k value from directory name: {dir_name}"
                )
        else:
            logger.warning(
                f"Directory name doesn't follow '_k<N>' pattern: {dir_name}"
            )
    
    if not k_to_dir:
        raise ValueError(
            "Could not parse any k values from directory names. "
            "Directories should end with '_k<N>' (e.g., 'job_merged_k1')"
        )
    
    logger.info(f"Parsed k values: {sorted(k_to_dir.keys())}")
    
    plotter = SEvalMetricsPlotter(
        merged_dirs=k_to_dir,
        output_dir=output_dir
    )
    plotter.generate_all_plots(experiment=experiment)


def generate_plot_statistics_from_utterance_details(
    utterance_details_file: str,
    k_value: int,
    output_json: str = None
) -> dict:
    """
    Generate aggregated statistics from utterance details for plotting.
    
    This reads the per-utterance details and generates hop-level statistics
    organized by hop sequence, compatible with the plotting functions.
    
    Args:
        utterance_details_file: Path to utterance details JSON file
        k_value: Top-k value to extract statistics for
        output_json: Optional path to save statistics JSON (if None, only returns dict)
    
    Returns:
        dict: Statistics organized with single_hop and multi_hop containing hop sequences as keys
    """
    import json
    from collections import defaultdict
    from pathlib import Path
    
    with open(utterance_details_file, 'r', encoding='utf-8') as f:
        details = json.load(f)
    
    utterances = details.get("utterances", {})
    k_str = str(k_value)
    
    # Initialize statistics (same format as old code)
    stats = {
        "source_file": utterance_details_file,
        "top_k": k_value,
        "total_utterances": len(utterances),
        "utterances_with_scores": 0,
        "utterances_without_any_scores": 0,  # Utterances with NO scores anywhere
        "per_hop": defaultdict(lambda: {
            "all_scores": [],
            "top_k_scores": [],
            "total_utterances": 0,
            "utterances_with_scores": 0,
            "utterances_without_scores": 0,
            "utterances_with_scores_elsewhere": 0  # Has scores at other hops but not this one
        }),
        "per_hop_sequence": defaultdict(lambda: {
            "all_scores": [],
            "top_k_scores": [],
            "utterances_with_scores": 0
        }),
        "single_hop": defaultdict(lambda: {
            "all_scores": [],
            "top_k_scores": [],
            "utterances_count": 0
        }),
        "multi_hop": defaultdict(lambda: {
            "all_scores": [],
            "top_k_scores": [],
            "utterances_count": 0
        })
    }
    
    # Process each utterance
    for utterance_id, utterance_data in utterances.items():
        hops = utterance_data.get("hops", {})
        
        if not hops:
            # Utterance has no hops data at all - count as no scores anywhere
            stats["utterances_without_any_scores"] += 1
            continue
        
        utterance_has_scores = False
        non_empty_hops = []
        hops_with_scores_set = set()  # Track which hop indices have scores for this utterance
        all_hop_indices = []  # Track all hop indices for this utterance
        
        # Process each hop
        for hop_idx_str, hop_data in hops.items():
            hop_idx = int(hop_idx_str)
            all_hop_indices.append(hop_idx)
            k_data = hop_data.get(k_str, {})
            
            if not k_data:
                continue
            
            is_empty = k_data.get("is_empty", True)
            hop_seq = k_data.get("hop_sequence")
            
            # Handle both dict and plain value formats
            avg_all_raw = k_data.get("avg_all_scores")
            avg_topk_raw = k_data.get("avg_topk_scores")
            
            avg_all = avg_all_raw.get("mean") if isinstance(avg_all_raw, dict) else avg_all_raw
            avg_topk = avg_topk_raw.get("mean") if isinstance(avg_topk_raw, dict) else avg_topk_raw
            
            # Per-hop index (includes empty hops)
            stats["per_hop"][hop_idx]["total_utterances"] += 1
            
            if not is_empty and avg_all is not None:
                utterance_has_scores = True
                hops_with_scores_set.add(hop_idx)
                non_empty_hops.append((hop_seq, avg_all, avg_topk))
                
                stats["per_hop"][hop_idx]["all_scores"].append(avg_all)
                stats["per_hop"][hop_idx]["top_k_scores"].append(avg_topk)
                stats["per_hop"][hop_idx]["utterances_with_scores"] += 1
                
                # Per-hop sequence (only non-empty)
                if hop_seq is not None:
                    stats["per_hop_sequence"][hop_seq]["all_scores"].append(avg_all)
                    stats["per_hop_sequence"][hop_seq]["top_k_scores"].append(avg_topk)
                    stats["per_hop_sequence"][hop_seq]["utterances_with_scores"] += 1
            else:
                stats["per_hop"][hop_idx]["utterances_without_scores"] += 1
        
        if utterance_has_scores:
            stats["utterances_with_scores"] += 1
            # For hops where this utterance doesn't have scores but has scores elsewhere,
            # increment the "utterances_with_scores_elsewhere" counter
            for hop_idx in all_hop_indices:
                if hop_idx not in hops_with_scores_set:
                    stats["per_hop"][hop_idx]["utterances_with_scores_elsewhere"] += 1
        else:
            # Utterance has no scores at any hop
            stats["utterances_without_any_scores"] += 1
        
        # Single-hop vs multi-hop
        if non_empty_hops:
            if len(non_empty_hops) == 1:
                # Single-hop
                hop_seq, avg_all, avg_topk = non_empty_hops[0]
                stats["single_hop"][hop_seq]["all_scores"].append(avg_all)
                stats["single_hop"][hop_seq]["top_k_scores"].append(avg_topk)
                stats["single_hop"][hop_seq]["utterances_count"] += 1
            else:
                # Multi-hop
                for hop_seq, avg_all, avg_topk in non_empty_hops:
                    stats["multi_hop"][hop_seq]["all_scores"].append(avg_all)
                    stats["multi_hop"][hop_seq]["top_k_scores"].append(avg_topk)
                    # Note: utterances_count will be calculated from len(all_scores) later
    
    # Calculate averages, std, and counts for each category
    import math
    
    def calc_std(values):
        """Calculate standard deviation"""
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
    
    for hop_idx, hop_data in stats["per_hop"].items():
        if hop_data["all_scores"]:
            hop_data["avg_all_scores"] = sum(hop_data["all_scores"]) / len(hop_data["all_scores"])
            hop_data["std_all_scores"] = calc_std(hop_data["all_scores"])
            hop_data["avg_topk_scores"] = sum(hop_data["top_k_scores"]) / len(hop_data["top_k_scores"])
            hop_data["std_topk_scores"] = calc_std(hop_data["top_k_scores"])
        else:
            hop_data["avg_all_scores"] = None
            hop_data["std_all_scores"] = None
            hop_data["avg_topk_scores"] = None
            hop_data["std_topk_scores"] = None
    
    for hop_seq, hop_data in stats["per_hop_sequence"].items():
        if hop_data["all_scores"]:
            hop_data["avg_all_scores"] = sum(hop_data["all_scores"]) / len(hop_data["all_scores"])
            hop_data["std_all_scores"] = calc_std(hop_data["all_scores"])
            hop_data["avg_topk_scores"] = sum(hop_data["top_k_scores"]) / len(hop_data["top_k_scores"])
            hop_data["std_topk_scores"] = calc_std(hop_data["top_k_scores"])
        else:
            hop_data["avg_all_scores"] = None
            hop_data["std_all_scores"] = None
            hop_data["avg_topk_scores"] = None
            hop_data["std_topk_scores"] = None
    
    for hop_seq, hop_data in stats["single_hop"].items():
        if hop_data["all_scores"]:
            hop_data["avg_all_scores"] = sum(hop_data["all_scores"]) / len(hop_data["all_scores"])
            hop_data["std_all_scores"] = calc_std(hop_data["all_scores"])
            hop_data["avg_topk_scores"] = sum(hop_data["top_k_scores"]) / len(hop_data["top_k_scores"])
            hop_data["std_topk_scores"] = calc_std(hop_data["top_k_scores"])
        # utterances_count is already correctly incremented
    
    for hop_seq, hop_data in stats["multi_hop"].items():
        if hop_data["all_scores"]:
            hop_data["avg_all_scores"] = sum(hop_data["all_scores"]) / len(hop_data["all_scores"])
            hop_data["std_all_scores"] = calc_std(hop_data["all_scores"])
            hop_data["avg_topk_scores"] = sum(hop_data["top_k_scores"]) / len(hop_data["top_k_scores"])
            hop_data["std_topk_scores"] = calc_std(hop_data["top_k_scores"])
            # Set utterances_count to the number of scores at this hop
            hop_data["utterances_count"] = len(hop_data["all_scores"])
        else:
            # No scores means no utterances reached this hop
            hop_data["utterances_count"] = 0
    
    # Convert defaultdicts to regular dicts for JSON serialization
    stats["per_hop"] = dict(stats["per_hop"])
    stats["per_hop_sequence"] = dict(stats["per_hop_sequence"])
    stats["single_hop"] = dict(stats["single_hop"])
    stats["multi_hop"] = dict(stats["multi_hop"])
    
    # Save to file if output_json is provided
    if output_json:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    
    return stats


def generate_statistics_plots(
    stats_files: Dict[int, str],
    output_dir: Path,
    job_id: str,
    experiment: str = "control"
):
    """
    Generate hop progression plots from statistics files.
    Shows all k-values together: All Results, Top-1, Top-3, Top-5
    
    Creates 2 plots:
    1. By hop index (includes empty hops)
    2. By hop sequence (non-empty hops only)
    
    Args:
        stats_files: Dictionary mapping k values to statistics file paths
        output_dir: Directory to save plots
        job_id: SEVAL job ID for plot titles
        experiment: Experiment type (control/treatment) for labeling
    """
    import json

    import matplotlib.pyplot as plt
    import numpy as np

    # Load all statistics
    stats_data = {}
    for k, stats_file in sorted(stats_files.items()):
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats_data[k] = json.load(f)
    
    k_values = sorted(stats_data.keys())
    
    # ========== PLOT 1: Hop Index (includes empty hops) ==========
    hop_key = "per_hop"
    all_hop_numbers = set()
    for k in k_values:
        per_hop = stats_data[k].get(hop_key, {})
        all_hop_numbers.update(int(h) for h in per_hop.keys())
    
    if all_hop_numbers:
        hop_indices = sorted(all_hop_numbers)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot "All Results" line (using k_values[0] as it's same for all k)
        per_hop = stats_data[k_values[0]].get(hop_key, {})
        hop_avgs = [per_hop.get(str(i), {}).get("avg_all_scores")
                    if per_hop.get(str(i), {}).get("avg_all_scores") is not None else np.nan
                    for i in hop_indices]
        hop_stds = [per_hop.get(str(i), {}).get("std_all_scores", 0)
                    if per_hop.get(str(i), {}).get("std_all_scores") is not None else 0
                    for i in hop_indices]
        
        ax.errorbar(hop_indices, hop_avgs, yerr=hop_stds, marker='o', linewidth=3,
                    color='steelblue', label='All Results', markersize=8, capsize=5, capthick=2)
        
        # Plot Top-k lines for each k value
        for k in k_values:
            per_hop = stats_data[k].get(hop_key, {})
            hop_topk_avgs = [per_hop.get(str(i), {}).get("avg_topk_scores")
                            if per_hop.get(str(i), {}).get("avg_topk_scores") is not None else np.nan
                            for i in hop_indices]
            hop_topk_stds = [per_hop.get(str(i), {}).get("std_topk_scores", 0)
                            if per_hop.get(str(i), {}).get("std_topk_scores") is not None else 0
                            for i in hop_indices]
            ax.errorbar(hop_indices, hop_topk_avgs, yerr=hop_topk_stds, marker='s', linewidth=2.5,
                       linestyle='--', label=f'Top-{k}', markersize=7, capsize=4, capthick=1.5)
        
        ax.set_xlabel('Hop Index', fontsize=12)
        ax.set_ylabel('Utterance-Average CiteDCG Score', fontsize=12)
        ax.set_title(f'{experiment.upper()} - CiteDCG by Hop Index (Job: {job_id})', fontsize=13, fontweight='bold')
        ax.set_xticks(hop_indices)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = output_dir / f"{job_id}_{experiment}_by_hop_index.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {plot_file.name}")
    
    # ========== PLOT 2: Hop Sequence (non-empty hops only) ==========
    hop_key = "per_hop_sequence"
    all_sequences = set()
    for k in k_values:
        per_hop_seq = stats_data[k].get(hop_key, {})
        all_sequences.update(per_hop_seq.keys())
    
    if all_sequences:
        # Sort by length first, then by numeric hop values
        def sort_key(seq):
            hops = seq.split('->')
            return (len(hops), [int(h) for h in hops])
        sequences = sorted(all_sequences, key=sort_key)
        x_pos = np.arange(len(sequences))
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot "All Results" line (using k_values[0] as it's same for all k)
        per_hop_seq = stats_data[k_values[0]].get(hop_key, {})
        seq_avgs = [per_hop_seq.get(seq, {}).get("avg_all_scores")
                   if per_hop_seq.get(seq, {}).get("avg_all_scores") is not None else np.nan
                   for seq in sequences]
        seq_stds = [per_hop_seq.get(seq, {}).get("std_all_scores", 0)
                   if per_hop_seq.get(seq, {}).get("std_all_scores") is not None else 0
                   for seq in sequences]
        
        ax.errorbar(x_pos, seq_avgs, yerr=seq_stds, marker='o', linewidth=3,
                    color='steelblue', label='All Results', markersize=8, capsize=5, capthick=2)
        
        # Plot Top-k lines for each k value
        for k in k_values:
            per_hop_seq = stats_data[k].get(hop_key, {})
            seq_topk_avgs = [per_hop_seq.get(seq, {}).get("avg_topk_scores")
                            if per_hop_seq.get(seq, {}).get("avg_topk_scores") is not None else np.nan
                            for seq in sequences]
            seq_topk_stds = [per_hop_seq.get(seq, {}).get("std_topk_scores", 0)
                            if per_hop_seq.get(seq, {}).get("std_topk_scores") is not None else 0
                            for seq in sequences]
            ax.errorbar(x_pos, seq_topk_avgs, yerr=seq_topk_stds, marker='s', linewidth=2.5,
                    linestyle='--', label=f'Top-{k}', markersize=7)
        
        ax.set_xlabel('Hop Sequence', fontsize=12)
        ax.set_ylabel('Utterance-Average CiteDCG Score', fontsize=12)
        ax.set_title(f'{experiment.upper()} - CiteDCG by Hop Sequence (Job: {job_id})', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sequences, rotation=45, ha='right')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = output_dir / f"{job_id}_{experiment}_by_hop_sequence.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {plot_file.name}")
    
    # ========== PLOT 3: Single-Hop vs Multi-Hop Analysis ==========
    single_hop_data = stats_data[k_values[0]].get("single_hop", {})
    multi_hop_data = stats_data[k_values[0]].get("multi_hop", {})
    
    if single_hop_data or multi_hop_data:
        from matplotlib.lines import Line2D
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        # Define marker styles for different metric types
        metric_styles = {
            'all': {'marker': 'o', 'size': 12, 'linestyle': '-'},
            1: {'marker': '^', 'size': 10, 'linestyle': '--'},
            3: {'marker': 's', 'size': 9, 'linestyle': '-.'},
            5: {'marker': 'D', 'size': 8, 'linestyle': ':'}
        }
        
        single_hop_color = 'green'
        multi_hop_color = 'blue'
        
        # LEFT: CiteDCG scores for both single-hop and multi-hop
        max_hop = 1
        if multi_hop_data:
            hop_nums = [int(h) for h in multi_hop_data.keys() if h.isdigit()]
            if hop_nums:
                max_hop = max(max_hop, max(hop_nums))
        
        # Plot single-hop (only at hop 1) - "All Results" is k-independent
        if single_hop_data:
            hop_data = single_hop_data.get("1", {})
            avg_all = hop_data.get("avg_all_scores")
            
            if avg_all is not None:
                style = metric_styles['all']
                ax1.plot([1], [avg_all], marker=style['marker'], linewidth=0,
                        color=single_hop_color, label='All Results',
                        markersize=style['size'], zorder=5)
        
        # Plot multi-hop (progression across hops) - "All Results" is k-independent
        multi_hop_numbers = []
        if multi_hop_data:
            multi_hop_numbers = sorted([int(h) for h in multi_hop_data.keys() if h.isdigit()])
            
            if multi_hop_numbers:
                hop_avgs = []
                for hop_num in multi_hop_numbers:
                    hop_data = multi_hop_data.get(str(hop_num), {})
                    avg_all = hop_data.get("avg_all_scores")
                    hop_avgs.append(avg_all if avg_all is not None else np.nan)
                
                style = metric_styles['all']
                ax1.plot(multi_hop_numbers, hop_avgs, marker=style['marker'],
                        linewidth=3, color=multi_hop_color, label='All Results',
                        markersize=style['size'], linestyle=style['linestyle'],
                        zorder=5)
        
        # Plot top-k for each k value for single-hop
        if single_hop_data:
            for k in k_values:
                style = metric_styles.get(k, {'marker': 'x', 'size': 8, 'linestyle': '-'})
                single_hop = stats_data[k].get("single_hop", {})
                hop_data = single_hop.get("1", {})
                avg_topk = hop_data.get("avg_topk_scores")
                
                if avg_topk is not None:
                    ax1.plot([1], [avg_topk], marker=style['marker'],
                            linewidth=0, color=single_hop_color,
                            label=f'Top-{k}',
                            markersize=style['size'], alpha=0.8, zorder=4)
        
        # Plot top-k for each k value for multi-hop
        if multi_hop_data and multi_hop_numbers:
            for k in k_values:
                style = metric_styles.get(k, {'marker': 'x', 'size': 8, 'linestyle': '-'})
                multi_hop = stats_data[k].get("multi_hop", {})
                hop_topk_avgs = []
                for hop_num in multi_hop_numbers:
                    hop_data = multi_hop.get(str(hop_num), {})
                    avg_topk = hop_data.get("avg_topk_scores")
                    hop_topk_avgs.append(
                        avg_topk if avg_topk is not None else np.nan
                    )
                
                ax1.plot(multi_hop_numbers, hop_topk_avgs,
                        marker=style['marker'], linewidth=2.5,
                        linestyle=style['linestyle'], color=multi_hop_color,
                        label=f'Top-{k}',
                        markersize=style['size'], alpha=0.8, zorder=4)
        
        ax1.set_xlabel('Hop Sequence (only non-empty hops)', fontsize=13)
        ax1.set_ylabel('Utterance-Average CiteDCG Score', fontsize=13)
        ax1.set_title('CiteDCG Scores by Hop\n(only hops with scores)',
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(range(1, max_hop + 1))
        
        # Custom legend with two columns: Single-Hop | Multi-Hop
        single_hop_elements = []
        multi_hop_elements = []
        
        # Single-Hop column items
        single_hop_elements.append(Line2D([0], [0], marker='', color='none',
                                         linestyle='', label='Single-Hop'))
        
        style = metric_styles['all']
        single_hop_elements.append(Line2D([0], [0], marker=style['marker'],
                                         color=single_hop_color,
                                         markersize=style['size'],
                                         linestyle=style['linestyle'],
                                         label='All Results'))
        
        for k in k_values:
            style = metric_styles.get(k, {'marker': 'x', 'size': 8,
                                         'linestyle': '-'})
            single_hop_elements.append(Line2D([0], [0], marker=style['marker'],
                                             color=single_hop_color,
                                             markersize=style['size'],
                                             linestyle=style['linestyle'],
                                             label=f'Top-{k}'))
        
        # Multi-Hop column items
        multi_hop_elements.append(Line2D([0], [0], marker='', color='none',
                                        linestyle='', label='Multi-Hop'))
        
        style = metric_styles['all']
        multi_hop_elements.append(Line2D([0], [0], marker=style['marker'],
                                        color=multi_hop_color,
                                        markersize=style['size'],
                                        linestyle=style['linestyle'],
                                        label='All Results'))
        
        for k in k_values:
            style = metric_styles.get(k, {'marker': 'x', 'size': 8,
                                         'linestyle': '-'})
            multi_hop_elements.append(Line2D([0], [0], marker=style['marker'],
                                            color=multi_hop_color,
                                            markersize=style['size'],
                                            linestyle=style['linestyle'],
                                            label=f'Top-{k}'))
        
        # Combine elements for two-column layout
        legend_elements = single_hop_elements + multi_hop_elements
        leg = ax1.legend(handles=legend_elements, fontsize=10, loc='best',
                        ncol=2, columnspacing=2.5, handlelength=2.5,
                        handletextpad=0.5)
        
        # Bold the column headers
        texts = leg.get_texts()
        num_items_per_col = len(legend_elements) // 2
        texts[0].set_weight('bold')
        texts[num_items_per_col].set_weight('bold')
        
        ax1.grid(True, alpha=0.3)
        
        # RIGHT: Utterance counts
        hop_positions = list(range(1, max_hop + 1))
        single_hop_counts = []
        multi_hop_counts = []
        
        for hop_num in hop_positions:
            # Single-hop: only has count at hop 1
            if hop_num == 1 and single_hop_data:
                sh_data = single_hop_data.get("1", {})
                single_count = sh_data.get("utterances_count", 0)
            else:
                single_count = 0
            single_hop_counts.append(single_count)
            
            # Multi-hop: has counts at each hop sequence
            if multi_hop_data:
                mh_data = multi_hop_data.get(str(hop_num), {})
                multi_count = mh_data.get("utterances_count", 0)
            else:
                multi_count = 0
            multi_hop_counts.append(multi_count)
        
        ax2.plot(hop_positions, single_hop_counts, marker='o', linewidth=3,
                color='green', label='Single-Hop Utterances', markersize=8)
        ax2.plot(hop_positions, multi_hop_counts, marker='s', linewidth=3,
                color='blue', label='Multi-Hop Utterances', markersize=8,
                linestyle='--')
        
        ax2.set_xlabel('Hop Sequence (only non-empty hops)', fontsize=13)
        ax2.set_ylabel('Number of Utterances', fontsize=13)
        ax2.set_title('Utterance Counts by Hop\n(only hops with scores)',
                     fontsize=14, fontweight='bold')
        ax2.set_xticks(hop_positions)
        ax2.legend(fontsize=11, loc='best')
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(
            f'Single-Hop vs Multi-Hop Utterance Analysis\n'
            f'Job: {job_id} | Experiment: {experiment.upper()}',
            fontsize=15, fontweight='bold', y=0.98
        )
        
        plt.tight_layout()
        plot_file = output_dir / f"{job_id}_{experiment}_single_vs_multi_hop.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {plot_file.name}")


def generate_comparison_plots(
    stats_files_control: Dict[int, str],
    stats_files_treatment: Dict[int, str],
    output_dir: Path,
    job_id: str
):
    """
    Generate 2x2 comparison plots for control vs treatment experiments.
    
    Top row: Control experiment (CiteDCG scores | Utterance counts)
    Bottom row: Treatment experiment (CiteDCG scores | Utterance counts)
    
    Shows all k-values together: All Results, Top-1, Top-3, Top-5
    
    Args:
        stats_files_control: Dict mapping k values to control stats file paths
        stats_files_treatment: Dict mapping k values to treatment stats file paths
        output_dir: Directory to save plots
        job_id: SEVAL job ID for plot titles
    """
    import json

    import matplotlib.pyplot as plt
    import numpy as np
    
    print("  Generating 2x2 comparison plots...")
    
    # Validate inputs
    if not stats_files_control:
        print("  ⚠ Warning: No control statistics files provided")
        return
    if not stats_files_treatment:
        print("  ⚠ Warning: No treatment statistics files provided")
        return
    
    # Load statistics for both experiments
    stats_control = {}
    stats_treatment = {}
    
    for k, stats_file in sorted(stats_files_control.items()):
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats_control[k] = json.load(f)
    
    for k, stats_file in sorted(stats_files_treatment.items()):
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats_treatment[k] = json.load(f)
    
    k_values = sorted(stats_control.keys())
    
    if not k_values:
        print("  ⚠ Warning: No statistics data available for comparison plots")
        return
    
    # ========== PLOT 1: Hop Index (includes empty hops) - 3 rows x 2 cols layout ==========
    hop_key = "per_hop"
    
    # Determine max hop index across both experiments
    max_hop_index_control = max(
        int(h) for k_data in stats_control.values()
        for h in k_data.get(hop_key, {}).keys()
    ) if any(stats_control[k].get(hop_key) for k in k_values) else 0
    
    max_hop_index_treatment = max(
        int(h) for k_data in stats_treatment.values()
        for h in k_data.get(hop_key, {}).keys()
    ) if any(stats_treatment[k].get(hop_key) for k in k_values) else 0
    
    max_hop_index = max(max_hop_index_control, max_hop_index_treatment)
    
    if max_hop_index > 0:
        # Include all hop indices that exist in either experiment's statistics
        # (regardless of whether they have scores or not)
        hop_indices_with_data = []
        for i in range(1, max_hop_index + 1):
            # Check if this hop index exists in either experiment's statistics
            control_has_hop = str(i) in stats_control[k_values[0]].get(hop_key, {})
            treatment_has_hop = str(i) in stats_treatment[k_values[0]].get(hop_key, {})
            if control_has_hop or treatment_has_hop:
                hop_indices_with_data.append(i)
        
        if not hop_indices_with_data:
            print("  ⚠ Warning: No hop indices found in statistics")
            return
        
        hop_indices = hop_indices_with_data
        
        # Create 3 rows x 2 cols: Row 1-2 for score metrics, Row 3 for utterance counts
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        
        # ===== ROW 1: "All Results" =====
        # Row 1, Col 1: Control - All Results
        ax = axes[0, 0]
        per_hop = stats_control[k_values[0]].get(hop_key, {})
        hop_avgs = [per_hop.get(str(i), {}).get("avg_all_scores")
                    if per_hop.get(str(i), {}).get("avg_all_scores") is not None else np.nan
                    for i in hop_indices]
        hop_stds = [per_hop.get(str(i), {}).get("std_all_scores", 0)
                    if per_hop.get(str(i), {}).get("std_all_scores") is not None else 0
                    for i in hop_indices]
        
        ax.errorbar(hop_indices, hop_avgs, yerr=hop_stds, marker='o', linewidth=3,
                    color='steelblue', label='Control', markersize=8, capsize=5, capthick=2)
        
        # Add treatment data to same plot
        per_hop_treatment = stats_treatment[k_values[0]].get(hop_key, {})
        hop_avgs_treatment = [per_hop_treatment.get(str(i), {}).get("avg_all_scores")
                             if per_hop_treatment.get(str(i), {}).get("avg_all_scores") is not None else np.nan
                             for i in hop_indices]
        hop_stds_treatment = [per_hop_treatment.get(str(i), {}).get("std_all_scores", 0)
                             if per_hop_treatment.get(str(i), {}).get("std_all_scores") is not None else 0
                             for i in hop_indices]
        
        ax.errorbar(hop_indices, hop_avgs_treatment, yerr=hop_stds_treatment, marker='s', linewidth=3,
                    color='coral', label='Treatment', markersize=8, capsize=5, capthick=2, linestyle='--')
        
        ax.set_xlabel('Hop Index', fontsize=12)
        ax.set_ylabel('Utterance-Average CiteDCG Score', fontsize=12)
        ax.set_title('All Results - Hop Index', fontsize=13, fontweight='bold')
        ax.set_xticks(hop_indices)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Row 1, Col 2: Top-1
        ax = axes[0, 1]
        k = k_values[0] if len(k_values) > 0 else 1
        
        per_hop_control_k = stats_control[k].get(hop_key, {})
        hop_topk_avgs_control = [per_hop_control_k.get(str(i), {}).get("avg_topk_scores")
                                if per_hop_control_k.get(str(i), {}).get("avg_topk_scores") is not None else np.nan
                                for i in hop_indices]
        hop_topk_stds_control = [per_hop_control_k.get(str(i), {}).get("std_topk_scores", 0)
                                if per_hop_control_k.get(str(i), {}).get("std_topk_scores") is not None else 0
                                for i in hop_indices]
        
        ax.errorbar(hop_indices, hop_topk_avgs_control, yerr=hop_topk_stds_control, marker='o', linewidth=3,
                   color='steelblue', label='Control', markersize=8, capsize=5, capthick=2)
        
        per_hop_treatment_k = stats_treatment[k].get(hop_key, {})
        hop_topk_avgs_treatment = [per_hop_treatment_k.get(str(i), {}).get("avg_topk_scores")
                                  if per_hop_treatment_k.get(str(i), {}).get("avg_topk_scores") is not None else np.nan
                                  for i in hop_indices]
        hop_topk_stds_treatment = [per_hop_treatment_k.get(str(i), {}).get("std_topk_scores", 0)
                                  if per_hop_treatment_k.get(str(i), {}).get("std_topk_scores") is not None else 0
                                  for i in hop_indices]
        
        ax.errorbar(hop_indices, hop_topk_avgs_treatment, yerr=hop_topk_stds_treatment, marker='s', linewidth=3,
                   color='coral', label='Treatment', markersize=8, capsize=5, capthick=2, linestyle='--')
        
        ax.set_xlabel('Hop Index', fontsize=12)
        ax.set_ylabel('Utterance-Average CiteDCG Score', fontsize=12)
        ax.set_title(f'Top-{k} - Hop Index', fontsize=13, fontweight='bold')
        ax.set_xticks(hop_indices)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        # ===== ROW 2: Top-3 and Top-5 =====
        # Row 2, Col 1: Top-3
        ax = axes[1, 0]
        k = k_values[1] if len(k_values) > 1 else 3
        
        per_hop_control_k = stats_control[k].get(hop_key, {})
        hop_topk_avgs_control = [per_hop_control_k.get(str(i), {}).get("avg_topk_scores")
                                if per_hop_control_k.get(str(i), {}).get("avg_topk_scores") is not None else np.nan
                                for i in hop_indices]
        hop_topk_stds_control = [per_hop_control_k.get(str(i), {}).get("std_topk_scores", 0)
                                if per_hop_control_k.get(str(i), {}).get("std_topk_scores") is not None else 0
                                for i in hop_indices]
        
        ax.errorbar(hop_indices, hop_topk_avgs_control, yerr=hop_topk_stds_control, marker='o', linewidth=3,
                   color='steelblue', label='Control', markersize=8, capsize=5, capthick=2)
        
        per_hop_treatment_k = stats_treatment[k].get(hop_key, {})
        hop_topk_avgs_treatment = [per_hop_treatment_k.get(str(i), {}).get("avg_topk_scores")
                                  if per_hop_treatment_k.get(str(i), {}).get("avg_topk_scores") is not None else np.nan
                                  for i in hop_indices]
        hop_topk_stds_treatment = [per_hop_treatment_k.get(str(i), {}).get("std_topk_scores", 0)
                                  if per_hop_treatment_k.get(str(i), {}).get("std_topk_scores") is not None else 0
                                  for i in hop_indices]
        
        ax.errorbar(hop_indices, hop_topk_avgs_treatment, yerr=hop_topk_stds_treatment, marker='s', linewidth=3,
                   color='coral', label='Treatment', markersize=8, capsize=5, capthick=2, linestyle='--')
        
        ax.set_xlabel('Hop Index', fontsize=12)
        ax.set_ylabel('Utterance-Average CiteDCG Score', fontsize=12)
        ax.set_title(f'Top-{k} - Hop Index', fontsize=13, fontweight='bold')
        ax.set_xticks(hop_indices)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Row 2, Col 2: Top-5
        ax = axes[1, 1]
        k = k_values[2] if len(k_values) > 2 else 5
        
        per_hop_control_k = stats_control[k].get(hop_key, {})
        hop_topk_avgs_control = [per_hop_control_k.get(str(i), {}).get("avg_topk_scores")
                                if per_hop_control_k.get(str(i), {}).get("avg_topk_scores") is not None else np.nan
                                for i in hop_indices]
        hop_topk_stds_control = [per_hop_control_k.get(str(i), {}).get("std_topk_scores", 0)
                                if per_hop_control_k.get(str(i), {}).get("std_topk_scores") is not None else 0
                                for i in hop_indices]
        
        ax.errorbar(hop_indices, hop_topk_avgs_control, yerr=hop_topk_stds_control, marker='o', linewidth=3,
                   color='steelblue', label='Control', markersize=8, capsize=5, capthick=2)
        
        per_hop_treatment_k = stats_treatment[k].get(hop_key, {})
        hop_topk_avgs_treatment = [per_hop_treatment_k.get(str(i), {}).get("avg_topk_scores")
                                  if per_hop_treatment_k.get(str(i), {}).get("avg_topk_scores") is not None else np.nan
                                  for i in hop_indices]
        hop_topk_stds_treatment = [per_hop_treatment_k.get(str(i), {}).get("std_topk_scores", 0)
                                  if per_hop_treatment_k.get(str(i), {}).get("std_topk_scores") is not None else 0
                                  for i in hop_indices]
        
        ax.errorbar(hop_indices, hop_topk_avgs_treatment, yerr=hop_topk_stds_treatment, marker='s', linewidth=3,
                   color='coral', label='Treatment', markersize=8, capsize=5, capthick=2, linestyle='--')
        
        ax.set_xlabel('Hop Index', fontsize=12)
        ax.set_ylabel('Utterance-Average CiteDCG Score', fontsize=12)
        ax.set_title(f'Top-{k} - Hop Index', fontsize=13, fontweight='bold')
        ax.set_xticks(hop_indices)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        # ===== ROW 3: Utterance Counts (3 curves) =====
        # 🟢 With scores at this hop
        # 🟡 Has scores elsewhere but not at this hop  
        # 🔴 No scores anywhere (straight horizontal line)
        
        # Row 3, Col 1: Control utterance counts
        ax = axes[2, 0]
        per_hop = stats_control[k_values[0]].get(hop_key, {})
        total_with_any_scores = stats_control[k_values[0]].get("utterances_with_scores", 0)
        no_scores_anywhere = stats_control[k_values[0]].get("utterances_without_any_scores", 0)
        
        utterances_with_scores = [per_hop.get(str(i), {}).get("utterances_with_scores", 0)
                                  for i in hop_indices]
        # Calculate "scores elsewhere" as: total with any scores - with scores at this hop
        utterances_with_scores_elsewhere = [total_with_any_scores - c for c in utterances_with_scores]
        
        # Filter out zero counts for plotting (don't show markers for 0)
        with_scores_filtered = [(h, c) for h, c in zip(hop_indices, utterances_with_scores) if c > 0]
        scores_elsewhere_filtered = [(h, c) for h, c in zip(hop_indices, utterances_with_scores_elsewhere) if c > 0]
        
        if with_scores_filtered:
            hops_with, counts_with = zip(*with_scores_filtered)
            ax.plot(hops_with, counts_with, marker='o', linewidth=3,
                    color='green', label='With Scores at This Hop', markersize=8)
        if scores_elsewhere_filtered:
            hops_elsewhere, counts_elsewhere = zip(*scores_elsewhere_filtered)
            ax.plot(hops_elsewhere, counts_elsewhere, marker='^', linewidth=3,
                    color='orange', label='Scores Elsewhere (Not Here)', markersize=8, linestyle='--')
        # No scores anywhere - constant horizontal line
        if no_scores_anywhere > 0:
            ax.axhline(y=no_scores_anywhere, color='red', linestyle=':', linewidth=3,
                       label=f'No Scores Anywhere ({no_scores_anywhere})')
        
        ax.set_xlabel('Hop Index', fontsize=12)
        ax.set_ylabel('Number of Utterances', fontsize=12)
        ax.set_title('CONTROL - Utterance Counts', fontsize=13, fontweight='bold')
        ax.set_xticks(hop_indices)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Row 3, Col 2: Treatment utterance counts
        ax = axes[2, 1]
        per_hop = stats_treatment[k_values[0]].get(hop_key, {})
        total_with_any_scores = stats_treatment[k_values[0]].get("utterances_with_scores", 0)
        no_scores_anywhere = stats_treatment[k_values[0]].get("utterances_without_any_scores", 0)
        
        utterances_with_scores = [per_hop.get(str(i), {}).get("utterances_with_scores", 0)
                                  for i in hop_indices]
        # Calculate "scores elsewhere" as: total with any scores - with scores at this hop
        utterances_with_scores_elsewhere = [total_with_any_scores - c for c in utterances_with_scores]
        
        # Filter out zero counts for plotting (don't show markers for 0)
        with_scores_filtered = [(h, c) for h, c in zip(hop_indices, utterances_with_scores) if c > 0]
        scores_elsewhere_filtered = [(h, c) for h, c in zip(hop_indices, utterances_with_scores_elsewhere) if c > 0]
        
        if with_scores_filtered:
            hops_with, counts_with = zip(*with_scores_filtered)
            ax.plot(hops_with, counts_with, marker='o', linewidth=3,
                    color='green', label='With Scores at This Hop', markersize=8)
        if scores_elsewhere_filtered:
            hops_elsewhere, counts_elsewhere = zip(*scores_elsewhere_filtered)
            ax.plot(hops_elsewhere, counts_elsewhere, marker='^', linewidth=3,
                    color='orange', label='Scores Elsewhere (Not Here)', markersize=8, linestyle='--')
        # No scores anywhere - constant horizontal line
        if no_scores_anywhere > 0:
            ax.axhline(y=no_scores_anywhere, color='red', linestyle=':', linewidth=3,
                       label=f'No Scores Anywhere ({no_scores_anywhere})')
        
        ax.set_xlabel('Hop Index', fontsize=12)
        ax.set_ylabel('Number of Utterances', fontsize=12)
        ax.set_title('TREATMENT - Utterance Counts', fontsize=13, fontweight='bold')
        ax.set_xticks(hop_indices)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        fig.suptitle(
            f'Control vs Treatment: Hop Index Analysis (including empty hops)\n'
            f'Job: {job_id}',
            fontsize=16, fontweight='bold', y=0.995
        )
        
        plt.tight_layout()
        plot_file = output_dir / f"{job_id}_comparison_by_hop_index.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {plot_file.name}")
    
    # ========== PLOT 2: Hop Sequence (non-empty hops only) - 3 rows x 2 cols layout ==========
    hop_key = "per_hop_sequence"
    
    # Collect all sequences from both experiments
    all_sequences = set()
    for k in k_values:
        all_sequences.update(stats_control[k].get(hop_key, {}).keys())
        all_sequences.update(stats_treatment[k].get(hop_key, {}).keys())
    
    if all_sequences:
        # Sort by length first, then by numeric hop values
        def sort_key(seq):
            hops = seq.split('->')
            return (len(hops), [int(h) for h in hops])
        sequences = sorted(all_sequences, key=sort_key)
        
        fig, axes = plt.subplots(3, 2, figsize=(22, 20))
        x_pos = np.arange(len(sequences))
        
        # ===== ROW 1: "All Results" =====
        # Row 1, Col 1: All Results
        ax = axes[0, 0]
        per_hop_seq_control = stats_control[k_values[0]].get(hop_key, {})
        seq_avgs_control = [per_hop_seq_control.get(seq, {}).get("avg_all_scores")
                           if per_hop_seq_control.get(seq, {}).get("avg_all_scores") is not None else np.nan
                           for seq in sequences]
        seq_stds_control = [per_hop_seq_control.get(seq, {}).get("std_all_scores", 0)
                           if per_hop_seq_control.get(seq, {}).get("std_all_scores") is not None else 0
                           for seq in sequences]
        
        ax.errorbar(x_pos, seq_avgs_control, yerr=seq_stds_control, marker='o', linewidth=3,
                    color='steelblue', label='Control', markersize=8, capsize=5, capthick=2)
        
        per_hop_seq_treatment = stats_treatment[k_values[0]].get(hop_key, {})
        seq_avgs_treatment = [per_hop_seq_treatment.get(seq, {}).get("avg_all_scores")
                             if per_hop_seq_treatment.get(seq, {}).get("avg_all_scores") is not None else np.nan
                             for seq in sequences]
        seq_stds_treatment = [per_hop_seq_treatment.get(seq, {}).get("std_all_scores", 0)
                             if per_hop_seq_treatment.get(seq, {}).get("std_all_scores") is not None else 0
                             for seq in sequences]
        
        ax.errorbar(x_pos, seq_avgs_treatment, yerr=seq_stds_treatment, marker='s', linewidth=3,
                    color='coral', label='Treatment', markersize=8, capsize=5, capthick=2, linestyle='--')
        
        ax.set_xlabel('Hop Sequence', fontsize=12)
        ax.set_ylabel('Utterance-Average CiteDCG Score', fontsize=12)
        ax.set_title('All Results - Hop Sequence', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sequences, rotation=45, ha='right')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Row 1, Col 2: Top-1
        ax = axes[0, 1]
        k = k_values[0] if len(k_values) > 0 else 1
        
        per_hop_seq_control_k = stats_control[k].get(hop_key, {})
        seq_topk_avgs_control = [per_hop_seq_control_k.get(seq, {}).get("avg_topk_scores")
                                if per_hop_seq_control_k.get(seq, {}).get("avg_topk_scores") is not None else np.nan
                                for seq in sequences]
        seq_topk_stds_control = [per_hop_seq_control_k.get(seq, {}).get("std_topk_scores", 0)
                                if per_hop_seq_control_k.get(seq, {}).get("std_topk_scores") is not None else 0
                                for seq in sequences]
        
        ax.errorbar(x_pos, seq_topk_avgs_control, yerr=seq_topk_stds_control, marker='o', linewidth=3,
                   color='steelblue', label='Control', markersize=8, capsize=5, capthick=2)
        
        per_hop_seq_treatment_k = stats_treatment[k].get(hop_key, {})
        seq_topk_avgs_treatment = [per_hop_seq_treatment_k.get(seq, {}).get("avg_topk_scores")
                                  if per_hop_seq_treatment_k.get(seq, {}).get("avg_topk_scores") is not None else np.nan
                                  for seq in sequences]
        seq_topk_stds_treatment = [per_hop_seq_treatment_k.get(seq, {}).get("std_topk_scores", 0)
                                  if per_hop_seq_treatment_k.get(seq, {}).get("std_topk_scores") is not None else 0
                                  for seq in sequences]
        
        ax.errorbar(x_pos, seq_topk_avgs_treatment, yerr=seq_topk_stds_treatment, marker='s', linewidth=3,
                   color='coral', label='Treatment', markersize=8, capsize=5, capthick=2, linestyle='--')
        
        ax.set_xlabel('Hop Sequence', fontsize=12)
        ax.set_ylabel('Utterance-Average CiteDCG Score', fontsize=12)
        ax.set_title(f'Top-{k} - Hop Sequence', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sequences, rotation=45, ha='right')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        # ===== ROW 2: Top-3 and Top-5 =====
        # Row 2, Col 1: Top-3
        ax = axes[1, 0]
        k = k_values[1] if len(k_values) > 1 else 3
        
        per_hop_seq_control_k = stats_control[k].get(hop_key, {})
        seq_topk_avgs_control = [per_hop_seq_control_k.get(seq, {}).get("avg_topk_scores")
                                if per_hop_seq_control_k.get(seq, {}).get("avg_topk_scores") is not None else np.nan
                                for seq in sequences]
        seq_topk_stds_control = [per_hop_seq_control_k.get(seq, {}).get("std_topk_scores", 0)
                                if per_hop_seq_control_k.get(seq, {}).get("std_topk_scores") is not None else 0
                                for seq in sequences]
        
        ax.errorbar(x_pos, seq_topk_avgs_control, yerr=seq_topk_stds_control, marker='o', linewidth=3,
                   color='steelblue', label='Control', markersize=8, capsize=5, capthick=2)
        
        per_hop_seq_treatment_k = stats_treatment[k].get(hop_key, {})
        seq_topk_avgs_treatment = [per_hop_seq_treatment_k.get(seq, {}).get("avg_topk_scores")
                                  if per_hop_seq_treatment_k.get(seq, {}).get("avg_topk_scores") is not None else np.nan
                                  for seq in sequences]
        seq_topk_stds_treatment = [per_hop_seq_treatment_k.get(seq, {}).get("std_topk_scores", 0)
                                  if per_hop_seq_treatment_k.get(seq, {}).get("std_topk_scores") is not None else 0
                                  for seq in sequences]
        
        ax.errorbar(x_pos, seq_topk_avgs_treatment, yerr=seq_topk_stds_treatment, marker='s', linewidth=3,
                   color='coral', label='Treatment', markersize=8, capsize=5, capthick=2, linestyle='--')
        
        ax.set_xlabel('Hop Sequence', fontsize=12)
        ax.set_ylabel('Utterance-Average CiteDCG Score', fontsize=12)
        ax.set_title(f'Top-{k} - Hop Sequence', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sequences, rotation=45, ha='right')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Row 2, Col 2: Top-5
        ax = axes[1, 1]
        k = k_values[2] if len(k_values) > 2 else 5
        
        per_hop_seq_control_k = stats_control[k].get(hop_key, {})
        seq_topk_avgs_control = [per_hop_seq_control_k.get(seq, {}).get("avg_topk_scores")
                                if per_hop_seq_control_k.get(seq, {}).get("avg_topk_scores") is not None else np.nan
                                for seq in sequences]
        seq_topk_stds_control = [per_hop_seq_control_k.get(seq, {}).get("std_topk_scores", 0)
                                if per_hop_seq_control_k.get(seq, {}).get("std_topk_scores") is not None else 0
                                for seq in sequences]
        
        ax.errorbar(x_pos, seq_topk_avgs_control, yerr=seq_topk_stds_control, marker='o', linewidth=3,
                   color='steelblue', label='Control', markersize=8, capsize=5, capthick=2)
        
        per_hop_seq_treatment_k = stats_treatment[k].get(hop_key, {})
        seq_topk_avgs_treatment = [per_hop_seq_treatment_k.get(seq, {}).get("avg_topk_scores")
                                  if per_hop_seq_treatment_k.get(seq, {}).get("avg_topk_scores") is not None else np.nan
                                  for seq in sequences]
        seq_topk_stds_treatment = [per_hop_seq_treatment_k.get(seq, {}).get("std_topk_scores", 0)
                                  if per_hop_seq_treatment_k.get(seq, {}).get("std_topk_scores") is not None else 0
                                  for seq in sequences]
        
        ax.errorbar(x_pos, seq_topk_avgs_treatment, yerr=seq_topk_stds_treatment, marker='s', linewidth=3,
                   color='coral', label='Treatment', markersize=8, capsize=5, capthick=2, linestyle='--')
        
        ax.set_xlabel('Hop Sequence', fontsize=12)
        ax.set_ylabel('Utterance-Average CiteDCG Score', fontsize=12)
        ax.set_title(f'Top-{k} - Hop Sequence', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sequences, rotation=45, ha='right')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        # ===== ROW 3: Utterance Counts =====
        # Row 3, Col 1: Control utterance counts
        ax = axes[2, 0]
        per_hop_seq = stats_control[k_values[0]].get(hop_key, {})
        utterance_counts = [per_hop_seq.get(seq, {}).get("utterances_with_scores", 0)
                           for seq in sequences]
        
        # Filter to only plot non-zero counts
        x_pos_with_count = [x_pos[idx] for idx, count in enumerate(utterance_counts) if count > 0]
        counts_filtered = [count for count in utterance_counts if count > 0]
        sequences_filtered = [sequences[idx] for idx, count in enumerate(utterance_counts) if count > 0]
        
        if x_pos_with_count:
            ax.plot(x_pos_with_count, counts_filtered, marker='o', linewidth=3,
                    color='steelblue', label='Utterance Count', markersize=8)
        
        ax.set_xlabel('Hop Sequence', fontsize=12)
        ax.set_ylabel('Number of Utterances', fontsize=12)
        ax.set_title('CONTROL - Utterance Counts', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sequences, rotation=45, ha='right')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Row 3, Col 2: Treatment utterance counts
        ax = axes[2, 1]
        per_hop_seq = stats_treatment[k_values[0]].get(hop_key, {})
        utterance_counts = [per_hop_seq.get(seq, {}).get("utterances_with_scores", 0)
                           for seq in sequences]
        
        # Filter to only plot non-zero counts
        x_pos_with_count = [x_pos[idx] for idx, count in enumerate(utterance_counts) if count > 0]
        counts_filtered = [count for count in utterance_counts if count > 0]
        sequences_filtered = [sequences[idx] for idx, count in enumerate(utterance_counts) if count > 0]
        
        if x_pos_with_count:
            ax.plot(x_pos_with_count, counts_filtered, marker='o', linewidth=3,
                    color='steelblue', label='Utterance Count', markersize=8)
        
        ax.set_xlabel('Hop Sequence', fontsize=12)
        ax.set_ylabel('Number of Utterances', fontsize=12)
        ax.set_title('TREATMENT - Utterance Counts', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sequences, rotation=45, ha='right')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        fig.suptitle(
            f'Control vs Treatment: Hop Sequence Analysis (excluding empty hops)\n'
            f'Job: {job_id}',
            fontsize=16, fontweight='bold', y=0.995
        )
        
        plt.tight_layout()
        plot_file = output_dir / f"{job_id}_comparison_by_hop_sequence.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {plot_file.name}")
    
    # ========== PLOT 3: Single-Hop vs Multi-Hop - 3 rows x 2 cols layout ==========
    single_hop_control = stats_control[k_values[0]].get("single_hop", {})
    multi_hop_control = stats_control[k_values[0]].get("multi_hop", {})
    single_hop_treatment = stats_treatment[k_values[0]].get("single_hop", {})
    multi_hop_treatment = stats_treatment[k_values[0]].get("multi_hop", {})
    
    if any([single_hop_control, multi_hop_control, single_hop_treatment, multi_hop_treatment]):
        from matplotlib.lines import Line2D
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        
        # Define distinct colors and markers for better visibility
        # Control: Red/Gold tones (very different), Treatment: Blue/Black tones
        # Single-hop: Circles/triangles, Multi-hop: Diamonds/squares
        ctrl_single_color = 'red'
        ctrl_single_marker = 'o'
        ctrl_multi_color = 'gold'
        ctrl_multi_marker = 'D'
        
        trt_single_color = 'dodgerblue'
        trt_single_marker = '^'
        trt_multi_color = 'black'
        trt_multi_marker = 's'
        
        # Determine max hop across both experiments
        max_hop = 1
        if multi_hop_control:
            hop_nums = [int(h) for h in multi_hop_control.keys() if h.isdigit()]
            if hop_nums:
                max_hop = max(max_hop, max(hop_nums))
        if multi_hop_treatment:
            hop_nums = [int(h) for h in multi_hop_treatment.keys() if h.isdigit()]
            if hop_nums:
                max_hop = max(max_hop, max(hop_nums))
        
        hop_positions = list(range(1, max_hop + 1))
        
        # Get multi-hop numbers for each experiment
        multi_hop_numbers_control = []
        if multi_hop_control:
            multi_hop_numbers_control = sorted([int(h) for h in multi_hop_control.keys() if h.isdigit()])
        
        multi_hop_numbers_treatment = []
        if multi_hop_treatment:
            multi_hop_numbers_treatment = sorted([int(h) for h in multi_hop_treatment.keys() if h.isdigit()])
        
        # ===== ROW 1: All Results and Top-1 =====
        # Row 1, Col 1: All Results - Control vs Treatment
        ax = axes[0, 0]
        
        # Control: Single-hop at hop 1
        if single_hop_control:
            hop_data = single_hop_control.get("1", {})
            avg_all = hop_data.get("avg_all_scores")
            std_all = hop_data.get("std_all_scores", 0)
            if avg_all is not None:
                ax.errorbar([1], [avg_all], yerr=[std_all], marker=ctrl_single_marker,
                           color=ctrl_single_color, label='Control Single-Hop', markersize=10,
                           capsize=5, capthick=2, linewidth=0)
        
        # Control: Multi-hop progression
        if multi_hop_control and multi_hop_numbers_control:
            hop_avgs = [multi_hop_control.get(str(hop_num), {}).get("avg_all_scores")
                       if multi_hop_control.get(str(hop_num), {}).get("avg_all_scores") is not None else np.nan
                       for hop_num in multi_hop_numbers_control]
            hop_stds = [multi_hop_control.get(str(hop_num), {}).get("std_all_scores", 0)
                       if multi_hop_control.get(str(hop_num), {}).get("std_all_scores") is not None else 0
                       for hop_num in multi_hop_numbers_control]
            ax.errorbar(multi_hop_numbers_control, hop_avgs, yerr=hop_stds, marker=ctrl_multi_marker,
                       linewidth=3, color=ctrl_multi_color, label='Control Multi-Hop',
                       markersize=8, linestyle='--', capsize=5, capthick=2)
        
        # Treatment: Single-hop at hop 1
        if single_hop_treatment:
            hop_data = single_hop_treatment.get("1", {})
            avg_all = hop_data.get("avg_all_scores")
            std_all = hop_data.get("std_all_scores", 0)
            if avg_all is not None:
                ax.errorbar([1], [avg_all], yerr=[std_all], marker=trt_single_marker,
                           color=trt_single_color, label='Treatment Single-Hop', markersize=10,
                           capsize=5, capthick=2, linewidth=0)
        
        # Treatment: Multi-hop progression
        if multi_hop_treatment and multi_hop_numbers_treatment:
            hop_avgs = [multi_hop_treatment.get(str(hop_num), {}).get("avg_all_scores")
                       if multi_hop_treatment.get(str(hop_num), {}).get("avg_all_scores") is not None else np.nan
                       for hop_num in multi_hop_numbers_treatment]
            hop_stds = [multi_hop_treatment.get(str(hop_num), {}).get("std_all_scores", 0)
                       if multi_hop_treatment.get(str(hop_num), {}).get("std_all_scores") is not None else 0
                       for hop_num in multi_hop_numbers_treatment]
            ax.errorbar(multi_hop_numbers_treatment, hop_avgs, yerr=hop_stds, marker=trt_multi_marker,
                       linewidth=3, color=trt_multi_color, label='Treatment Multi-Hop',
                       markersize=8, linestyle='--', capsize=5, capthick=2)
        
        ax.set_xlabel('Hop Sequence', fontsize=12)
        ax.set_ylabel('Utterance-Average CiteDCG Score', fontsize=12)
        ax.set_title('All Results - Control vs Treatment', fontsize=13, fontweight='bold')
        ax.set_xticks(hop_positions)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Row 1, Col 2: Top-1 - Control vs Treatment
        ax = axes[0, 1]
        k = k_values[0] if len(k_values) > 0 else 1
        
        single_hop_control_k = stats_control[k].get("single_hop", {})
        multi_hop_control_k = stats_control[k].get("multi_hop", {})
        single_hop_treatment_k = stats_treatment[k].get("single_hop", {})
        multi_hop_treatment_k = stats_treatment[k].get("multi_hop", {})
        
        # Control: Single-hop at hop 1
        if single_hop_control_k:
            hop_data = single_hop_control_k.get("1", {})
            avg_topk = hop_data.get("avg_topk_scores")
            std_topk = hop_data.get("std_topk_scores", 0)
            if avg_topk is not None:
                ax.errorbar([1], [avg_topk], yerr=[std_topk], marker=ctrl_single_marker,
                           color=ctrl_single_color, label='Control Single-Hop', markersize=10,
                           capsize=5, capthick=2, linewidth=0)
        
        # Control: Multi-hop progression
        if multi_hop_control_k and multi_hop_numbers_control:
            hop_topk_avgs = [multi_hop_control_k.get(str(hop_num), {}).get("avg_topk_scores")
                            if multi_hop_control_k.get(str(hop_num), {}).get("avg_topk_scores") is not None else np.nan
                            for hop_num in multi_hop_numbers_control]
            hop_topk_stds = [multi_hop_control_k.get(str(hop_num), {}).get("std_topk_scores", 0)
                            if multi_hop_control_k.get(str(hop_num), {}).get("std_topk_scores") is not None else 0
                            for hop_num in multi_hop_numbers_control]
            ax.errorbar(multi_hop_numbers_control, hop_topk_avgs, yerr=hop_topk_stds, marker=ctrl_multi_marker,
                       linewidth=3, linestyle='--', color=ctrl_multi_color, label='Control Multi-Hop',
                       markersize=8, capsize=5, capthick=2)
        
        # Treatment: Single-hop at hop 1
        if single_hop_treatment_k:
            hop_data = single_hop_treatment_k.get("1", {})
            avg_topk = hop_data.get("avg_topk_scores")
            std_topk = hop_data.get("std_topk_scores", 0)
            if avg_topk is not None:
                ax.errorbar([1], [avg_topk], yerr=[std_topk], marker=trt_single_marker,
                           color=trt_single_color, label='Treatment Single-Hop', markersize=10,
                           capsize=5, capthick=2, linewidth=0)
        
        # Treatment: Multi-hop progression
        if multi_hop_treatment_k and multi_hop_numbers_treatment:
            hop_topk_avgs = [multi_hop_treatment_k.get(str(hop_num), {}).get("avg_topk_scores")
                            if multi_hop_treatment_k.get(str(hop_num), {}).get("avg_topk_scores") is not None else np.nan
                            for hop_num in multi_hop_numbers_treatment]
            hop_topk_stds = [multi_hop_treatment_k.get(str(hop_num), {}).get("std_topk_scores", 0)
                            if multi_hop_treatment_k.get(str(hop_num), {}).get("std_topk_scores") is not None else 0
                            for hop_num in multi_hop_numbers_treatment]
            ax.errorbar(multi_hop_numbers_treatment, hop_topk_avgs, yerr=hop_topk_stds, marker=trt_multi_marker,
                       linewidth=3, linestyle='--', color=trt_multi_color, label='Treatment Multi-Hop',
                       markersize=8, capsize=5, capthick=2)
        
        ax.set_xlabel('Hop Sequence', fontsize=12)
        ax.set_ylabel('Utterance-Average CiteDCG Score', fontsize=12)
        ax.set_title(f'Top-{k} - Control vs Treatment', fontsize=13, fontweight='bold')
        ax.set_xticks(hop_positions)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        # ===== ROW 2: Top-3 and Top-5 =====
        # Row 2, Col 1: Top-3 - Control vs Treatment
        if len(k_values) > 1:
            ax = axes[1, 0]
            k = k_values[1]
            
            single_hop_control_k3 = stats_control[k].get("single_hop", {})
            multi_hop_control_k3 = stats_control[k].get("multi_hop", {})
            single_hop_treatment_k3 = stats_treatment[k].get("single_hop", {})
            multi_hop_treatment_k3 = stats_treatment[k].get("multi_hop", {})
            
            # Control: Single-hop at hop 1
            if single_hop_control_k3:
                hop_data = single_hop_control_k3.get("1", {})
                avg_topk = hop_data.get("avg_topk_scores")
                std_topk = hop_data.get("std_topk_scores", 0)
                if avg_topk is not None:
                    ax.errorbar([1], [avg_topk], yerr=[std_topk], marker=ctrl_single_marker,
                               color=ctrl_single_color, label='Control Single-Hop', markersize=10,
                               capsize=5, capthick=2, linewidth=0)
            
            # Control: Multi-hop progression
            if multi_hop_control_k3 and multi_hop_numbers_control:
                hop_topk_avgs = [multi_hop_control_k3.get(str(hop_num), {}).get("avg_topk_scores")
                                if multi_hop_control_k3.get(str(hop_num), {}).get("avg_topk_scores") is not None else np.nan
                                for hop_num in multi_hop_numbers_control]
                hop_topk_stds = [multi_hop_control_k3.get(str(hop_num), {}).get("std_topk_scores", 0)
                                if multi_hop_control_k3.get(str(hop_num), {}).get("std_topk_scores") is not None else 0
                                for hop_num in multi_hop_numbers_control]
                ax.errorbar(multi_hop_numbers_control, hop_topk_avgs, yerr=hop_topk_stds, marker=ctrl_multi_marker,
                           linewidth=3, linestyle='--', color=ctrl_multi_color, label='Control Multi-Hop',
                           markersize=8, capsize=5, capthick=2)
            
            # Treatment: Single-hop at hop 1
            if single_hop_treatment_k3:
                hop_data = single_hop_treatment_k3.get("1", {})
                avg_topk = hop_data.get("avg_topk_scores")
                std_topk = hop_data.get("std_topk_scores", 0)
                if avg_topk is not None:
                    ax.errorbar([1], [avg_topk], yerr=[std_topk], marker=trt_single_marker,
                               color=trt_single_color, label='Treatment Single-Hop', markersize=10,
                               capsize=5, capthick=2, linewidth=0)
            
            # Treatment: Multi-hop progression
            if multi_hop_treatment_k3 and multi_hop_numbers_treatment:
                hop_topk_avgs = [multi_hop_treatment_k3.get(str(hop_num), {}).get("avg_topk_scores")
                                if multi_hop_treatment_k3.get(str(hop_num), {}).get("avg_topk_scores") is not None else np.nan
                                for hop_num in multi_hop_numbers_treatment]
                hop_topk_stds = [multi_hop_treatment_k3.get(str(hop_num), {}).get("std_topk_scores", 0)
                                if multi_hop_treatment_k3.get(str(hop_num), {}).get("std_topk_scores") is not None else 0
                                for hop_num in multi_hop_numbers_treatment]
                ax.errorbar(multi_hop_numbers_treatment, hop_topk_avgs, yerr=hop_topk_stds, marker=trt_multi_marker,
                           linewidth=3, linestyle='--', color=trt_multi_color, label='Treatment Multi-Hop',
                           markersize=8, capsize=5, capthick=2)
            
            ax.set_xlabel('Hop Sequence', fontsize=12)
            ax.set_ylabel('Utterance-Average CiteDCG Score', fontsize=12)
            ax.set_title(f'Top-{k} - Control vs Treatment', fontsize=13, fontweight='bold')
            ax.set_xticks(hop_positions)
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, alpha=0.3)
        
        # Row 2, Col 2: Top-5 - Control vs Treatment
        if len(k_values) > 2:
            ax = axes[1, 1]
            k = k_values[2]
            
            single_hop_control_k5 = stats_control[k].get("single_hop", {})
            multi_hop_control_k5 = stats_control[k].get("multi_hop", {})
            single_hop_treatment_k5 = stats_treatment[k].get("single_hop", {})
            multi_hop_treatment_k5 = stats_treatment[k].get("multi_hop", {})
            
            # Control: Single-hop at hop 1
            if single_hop_control_k5:
                hop_data = single_hop_control_k5.get("1", {})
                avg_topk = hop_data.get("avg_topk_scores")
                std_topk = hop_data.get("std_topk_scores", 0)
                if avg_topk is not None:
                    ax.errorbar([1], [avg_topk], yerr=[std_topk], marker=ctrl_single_marker,
                               color=ctrl_single_color, label='Control Single-Hop', markersize=10,
                               capsize=5, capthick=2, linewidth=0)
            
            # Control: Multi-hop progression
            if multi_hop_control_k5 and multi_hop_numbers_control:
                hop_topk_avgs = [multi_hop_control_k5.get(str(hop_num), {}).get("avg_topk_scores")
                                if multi_hop_control_k5.get(str(hop_num), {}).get("avg_topk_scores") is not None else np.nan
                                for hop_num in multi_hop_numbers_control]
                hop_topk_stds = [multi_hop_control_k5.get(str(hop_num), {}).get("std_topk_scores", 0)
                                if multi_hop_control_k5.get(str(hop_num), {}).get("std_topk_scores") is not None else 0
                                for hop_num in multi_hop_numbers_control]
                ax.errorbar(multi_hop_numbers_control, hop_topk_avgs, yerr=hop_topk_stds, marker=ctrl_multi_marker,
                           linewidth=3, linestyle='--', color=ctrl_multi_color, label='Control Multi-Hop',
                           markersize=8, capsize=5, capthick=2)
            
            # Treatment: Single-hop at hop 1
            if single_hop_treatment_k5:
                hop_data = single_hop_treatment_k5.get("1", {})
                avg_topk = hop_data.get("avg_topk_scores")
                std_topk = hop_data.get("std_topk_scores", 0)
                if avg_topk is not None:
                    ax.errorbar([1], [avg_topk], yerr=[std_topk], marker=trt_single_marker,
                               color=trt_single_color, label='Treatment Single-Hop', markersize=10,
                               capsize=5, capthick=2, linewidth=0)
            
            # Treatment: Multi-hop progression
            if multi_hop_treatment_k5 and multi_hop_numbers_treatment:
                hop_topk_avgs = [multi_hop_treatment_k5.get(str(hop_num), {}).get("avg_topk_scores")
                                if multi_hop_treatment_k5.get(str(hop_num), {}).get("avg_topk_scores") is not None else np.nan
                                for hop_num in multi_hop_numbers_treatment]
                hop_topk_stds = [multi_hop_treatment_k5.get(str(hop_num), {}).get("std_topk_scores", 0)
                                if multi_hop_treatment_k5.get(str(hop_num), {}).get("std_topk_scores") is not None else 0
                                for hop_num in multi_hop_numbers_treatment]
                ax.errorbar(multi_hop_numbers_treatment, hop_topk_avgs, yerr=hop_topk_stds, marker=trt_multi_marker,
                           linewidth=3, linestyle='--', color=trt_multi_color, label='Treatment Multi-Hop',
                           markersize=8, capsize=5, capthick=2)
            
            ax.set_xlabel('Hop Sequence', fontsize=12)
            ax.set_ylabel('Utterance-Average CiteDCG Score', fontsize=12)
            ax.set_title(f'Top-{k} - Control vs Treatment', fontsize=13, fontweight='bold')
            ax.set_xticks(hop_positions)
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, alpha=0.3)
        
        # ===== ROW 3: Utterance Counts =====
        # Row 3, Col 1: Control utterance counts
        ax = axes[2, 0]
        single_hop_counts = []
        multi_hop_counts = []
        for hop_num in hop_positions:
            if hop_num == 1 and single_hop_control:
                sh_data = single_hop_control.get("1", {})
                single_count = sh_data.get("utterances_count", 0)
            else:
                single_count = 0
            single_hop_counts.append(single_count)
            
            if multi_hop_control:
                mh_data = multi_hop_control.get(str(hop_num), {})
                multi_count = mh_data.get("utterances_count", 0)
            else:
                multi_count = 0
            multi_hop_counts.append(multi_count)
        
        # Filter to plot only non-zero counts
        hop_single_data = [(hop_positions[idx], single_hop_counts[idx]) 
                           for idx in range(len(hop_positions)) if single_hop_counts[idx] > 0]
        hop_multi_data = [(hop_positions[idx], multi_hop_counts[idx]) 
                          for idx in range(len(hop_positions)) if multi_hop_counts[idx] > 0]
        
        if hop_single_data:
            hops_s, counts_s = zip(*hop_single_data)
            ax.plot(hops_s, counts_s, marker='o', linewidth=3,
                    color='red', label='Single-Hop', markersize=8)
        if hop_multi_data:
            hops_m, counts_m = zip(*hop_multi_data)
            ax.plot(hops_m, counts_m, marker='s', linewidth=3,
                    color='blue', label='Multi-Hop', markersize=8, linestyle='--')
        
        ax.set_xlabel('Hop Sequence', fontsize=12)
        ax.set_ylabel('Number of Utterances', fontsize=12)
        ax.set_title('CONTROL - Utterance Counts', fontsize=13, fontweight='bold')
        ax.set_xticks(hop_positions)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Row 3, Col 2: Treatment utterance counts
        ax = axes[2, 1]
        single_hop_counts = []
        multi_hop_counts = []
        for hop_num in hop_positions:
            if hop_num == 1 and single_hop_treatment:
                sh_data = single_hop_treatment.get("1", {})
                single_count = sh_data.get("utterances_count", 0)
            else:
                single_count = 0
            single_hop_counts.append(single_count)
            
            if multi_hop_treatment:
                mh_data = multi_hop_treatment.get(str(hop_num), {})
                multi_count = mh_data.get("utterances_count", 0)
            else:
                multi_count = 0
            multi_hop_counts.append(multi_count)
        
        # Filter to plot only non-zero counts
        hop_single_data = [(hop_positions[idx], single_hop_counts[idx]) 
                           for idx in range(len(hop_positions)) if single_hop_counts[idx] > 0]
        hop_multi_data = [(hop_positions[idx], multi_hop_counts[idx]) 
                          for idx in range(len(hop_positions)) if multi_hop_counts[idx] > 0]
        
        if hop_single_data:
            hops_s, counts_s = zip(*hop_single_data)
            ax.plot(hops_s, counts_s, marker='o', linewidth=3,
                    color='red', label='Single-Hop', markersize=8)
        if hop_multi_data:
            hops_m, counts_m = zip(*hop_multi_data)
            ax.plot(hops_m, counts_m, marker='s', linewidth=3,
                    color='blue', label='Multi-Hop', markersize=8, linestyle='--')
        
        ax.set_xlabel('Hop Sequence', fontsize=12)
        ax.set_ylabel('Number of Utterances', fontsize=12)
        ax.set_title('TREATMENT - Utterance Counts', fontsize=13, fontweight='bold')
        ax.set_xticks(hop_positions)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        fig.suptitle(
            f'Control vs Treatment: Single-Hop vs Multi-Hop Analysis (excluding empty hops)\n'
            f'Job: {job_id}',
            fontsize=16, fontweight='bold', y=0.995
        )
        
        plt.tight_layout()
        plot_file = output_dir / f"{job_id}_comparison_single_vs_multi_hop.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {plot_file.name}")


def generate_paired_utterances_plot(
    paired_utterances_file: str,
    output_dir: Path,
    job_id: str,
    k_values: list = None
):
    """
    Generate scatter plot showing control vs treatment scores for each paired utterance.
    
    Args:
        paired_utterances_file: Path to paired utterances JSON file
        output_dir: Directory to save the plot
        job_id: SEVAL job ID for plot title
        k_values: List of top-k values to plot (default: [1, 3, 5])
    """
    import json

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    
    if k_values is None:
        k_values = [1, 3, 5]
    
    # Load paired utterances data
    with open(paired_utterances_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    paired_data = data.get("paired_data", [])
    if not paired_data:
        print("  ⚠ No paired utterances with scores found")
        return
    
    metadata = data.get("metadata", {})
    common_k = metadata.get("common_k_values", k_values)
    
    # Prepare data for plotting
    num_utterances = len(paired_data)
    x_positions = list(range(1, num_utterances + 1))
    
    # Define marker styles
    marker_styles = {
        'all': {'marker': 'o', 'size': 8, 'label': 'All Results'},
        1: {'marker': '^', 'size': 7, 'label': 'Top-1'},
        3: {'marker': 's', 'size': 6, 'label': 'Top-3'},
        5: {'marker': 'D', 'size': 5, 'label': 'Top-5'}
    }
    
    control_color = '#1f77b4'  # Blue
    treatment_color = '#ff7f0e'  # Orange
    
    fig, ax = plt.subplots(1, 1, figsize=(max(12, num_utterances * 0.15), 8))
    
    # Plot each utterance
    for idx, pair in enumerate(paired_data):
        x_pos = x_positions[idx]
        control_hops = pair.get("control", {}).get("hops", {})
        treatment_hops = pair.get("treatment", {}).get("hops", {})
        
        # Get first hop with scores
        control_scores = {}
        treatment_scores = {}
        
        for hop_idx, hop_data in control_hops.items():
            k1_data = hop_data.get("1", {})
            if not k1_data.get("is_empty", True):
                # Handle both dict and float formats
                avg_all_raw = k1_data.get("avg_all_scores")
                control_scores['all'] = avg_all_raw.get("mean") if isinstance(avg_all_raw, dict) else avg_all_raw
                for k in common_k:
                    k_data = hop_data.get(str(k), {})
                    if not k_data.get("is_empty", True):
                        avg_topk_raw = k_data.get("avg_topk_scores")
                        control_scores[k] = avg_topk_raw.get("mean") if isinstance(avg_topk_raw, dict) else avg_topk_raw
                break
        
        for hop_idx, hop_data in treatment_hops.items():
            k1_data = hop_data.get("1", {})
            if not k1_data.get("is_empty", True):
                # Handle both dict and float formats
                avg_all_raw = k1_data.get("avg_all_scores")
                treatment_scores['all'] = avg_all_raw.get("mean") if isinstance(avg_all_raw, dict) else avg_all_raw
                for k in common_k:
                    k_data = hop_data.get(str(k), {})
                    if not k_data.get("is_empty", True):
                        avg_topk_raw = k_data.get("avg_topk_scores")
                        treatment_scores[k] = avg_topk_raw.get("mean") if isinstance(avg_topk_raw, dict) else avg_topk_raw
                break
        
        # Plot control scores (left offset)
        offset = -0.15
        for metric, score in control_scores.items():
            if score is not None:
                style = marker_styles.get(metric, marker_styles['all'])
                ax.scatter(x_pos + offset, score,
                          marker=style['marker'],
                          s=style['size']**2,
                          color=control_color,
                          alpha=0.7,
                          edgecolors='black',
                          linewidths=0.5,
                          zorder=3)
        
        # Plot treatment scores (right offset)
        offset = 0.15
        for metric, score in treatment_scores.items():
            if score is not None:
                style = marker_styles.get(metric, marker_styles['all'])
                ax.scatter(x_pos + offset, score,
                          marker=style['marker'],
                          s=style['size']**2,
                          color=treatment_color,
                          alpha=0.7,
                          edgecolors='black',
                          linewidths=0.5,
                          zorder=3)
    
    # Configure axes
    ax.set_xlabel('Utterance Index', fontsize=13)
    ax.set_ylabel('CiteDCG Score (First Non-Empty Hop)', fontsize=13)
    ax.set_title(
        f'Control vs Treatment: Paired Utterances CiteDCG Scores\n'
        f'Job: {job_id} | {num_utterances} paired utterances',
        fontsize=14, fontweight='bold'
    )
    
    # Set x-axis ticks
    if num_utterances <= 50:
        ax.set_xticks(x_positions)
    else:
        step = 5 if num_utterances <= 100 else 10
        ax.set_xticks(x_positions[::step])
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(0, num_utterances + 1)
    
    # Create legend
    legend_elements = []
    legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=control_color,
                                 markersize=8, label='Control',
                                 markeredgecolor='black', markeredgewidth=0.5))
    legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=treatment_color,
                                 markersize=8, label='Treatment',
                                 markeredgecolor='black', markeredgewidth=0.5))
    legend_elements.append(Line2D([0], [0], marker='', color='none',
                                 linestyle='', label=''))
    
    for metric_key in ['all'] + common_k:
        style = marker_styles.get(metric_key, marker_styles['all'])
        legend_elements.append(
            Line2D([0], [0], marker=style['marker'], color='w',
                  markerfacecolor='gray', markersize=style['size'],
                  label=style['label'],
                  markeredgecolor='black', markeredgewidth=0.5))
    
    ax.legend(handles=legend_elements, fontsize=10, loc='best',
             ncol=2, framealpha=0.9)
    
    plt.tight_layout()
    plot_file = output_dir / f"{job_id}_paired_utterances_comparison.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {plot_file.name}")


if __name__ == "__main__":
    import fire
    fire.Fire({
        'plot_seval_metrics': plot_seval_metrics,
        'generate_plot_statistics': generate_plot_statistics_from_utterance_details,
        'generate_statistics_plots': generate_statistics_plots,
        'generate_comparison_plots': generate_comparison_plots,
        'generate_paired_utterances_plot': generate_paired_utterances_plot
    })




