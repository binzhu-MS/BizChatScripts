#!/usr/bin/env python3
"""
Filter Non-Conflicting Treatment Winners from SEVAL Results

This script reads SEVAL job comparison results and filters utterances where treatment
outperforms control WITHOUT conflicts between the two jobs. It then extracts the 
matching rows from the original test data file.

Selection Criteria (Non-Conflicting Treatment Wins):
- Treatment wins in BOTH jobs (gain > 0 for both), OR
- Treatment wins in one job (gain > 0) while the other is a tie (gain == 0) or has no score (NaN)
- EXCLUDES conflicts where one job wins (gain > 0) and the other loses (gain < 0)

Usage:
    python filter_treatment_winners.py \\
        --seval_results_file=../../seval/results/123665_232361_citedcg_one_centric_all_utterances.tsv \\
        --test_data_file=results/complex_utterances_syntenants_vertedge_all.tsv \\
        --output_file=results/treatment_winners_filtered.tsv
"""

import sys
from pathlib import Path
import pandas as pd
import fire
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def filter_treatment_winners(
    seval_results_file: str,
    test_data_file: str,
    output_file: str,
    utterance_column: str = "utterance",
    show_stats: bool = True,
) -> None:
    """
    Filter utterances with non-conflicting treatment wins and extract matching test data.

    Selects utterances where:
    - Both jobs show treatment wins (gain > 0 for both), OR
    - One job shows treatment win (gain > 0) while other is tie (gain == 0) or missing (NaN)

    Excludes conflicts where one job wins and the other loses.

    Args:
        seval_results_file: Path to SEVAL job comparison TSV file
        test_data_file: Path to original test data TSV file
        output_file: Path for filtered output TSV file
        utterance_column: Column name containing utterances (default: "utterance")
        show_stats: Print detailed statistics (default: True)
    """

    # Read SEVAL results
    logger.info(f"Reading SEVAL results from: {seval_results_file}")
    try:
        seval_df = pd.read_csv(seval_results_file, sep="\t", encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to read SEVAL results file: {e}")
        sys.exit(1)

    logger.info(f"Loaded {len(seval_df)} utterances from SEVAL results")

    # Identify gain columns (they should have 'gain' in the name)
    gain_columns = [
        col
        for col in seval_df.columns
        if "gain" in col.lower() and "combined" not in col.lower()
    ]

    if len(gain_columns) < 2:
        logger.error(f"Expected at least 2 job gain columns, found: {gain_columns}")
        sys.exit(1)

    job1_gain_col = gain_columns[0]
    job2_gain_col = gain_columns[1]

    logger.info(f"Using gain columns: {job1_gain_col}, {job2_gain_col}")

    # Extract utterance column name (might be "utterance", "query", etc.)
    if utterance_column not in seval_df.columns:
        # Try to find it automatically
        possible_names = [
            "utterance",
            "Utterance",
            "query",
            "Query",
            "question",
            "Question",
        ]
        for name in possible_names:
            if name in seval_df.columns:
                utterance_column = name
                logger.info(f"Auto-detected utterance column: {utterance_column}")
                break
        else:
            logger.error(
                f"Could not find utterance column. Available columns: {list(seval_df.columns)}"
            )
            sys.exit(1)

    # Apply non-conflicting treatment winner filter
    total_count = len(seval_df)

    # Non-conflicting treatment wins:
    # 1. Both jobs win (gain > 0 for both)
    # 2. Job1 wins (gain > 0) and Job2 is tie (gain == 0) or missing (NaN)
    # 3. Job2 wins (gain > 0) and Job1 is tie (gain == 0) or missing (NaN)
    mask = (
        # Both jobs win
        ((seval_df[job1_gain_col] > 0) & (seval_df[job2_gain_col] > 0))
        |
        # Job1 wins, Job2 tie or missing
        (
            (seval_df[job1_gain_col] > 0)
            & ((seval_df[job2_gain_col] == 0) | seval_df[job2_gain_col].isna())
        )
        |
        # Job2 wins, Job1 tie or missing
        (
            (seval_df[job2_gain_col] > 0)
            & ((seval_df[job1_gain_col] == 0) | seval_df[job1_gain_col].isna())
        )
    )

    filtered_seval = seval_df[mask].copy()

    logger.info("Non-conflicting treatment wins filter applied")
    logger.info(
        f"Filtered: {len(filtered_seval)} / {total_count} utterances ({len(filtered_seval)/total_count*100:.1f}%)"
    )

    # Get the list of selected utterances
    selected_utterances = set(filtered_seval[utterance_column].str.strip())
    logger.info(f"Unique selected utterances: {len(selected_utterances)}")

    # Read test data file
    logger.info(f"Reading test data from: {test_data_file}")
    try:
        test_df = pd.read_csv(test_data_file, sep="\t", encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to read test data file: {e}")
        sys.exit(1)

    logger.info(f"Loaded {len(test_df)} rows from test data")

    # Find utterance column in test data (might have different case)
    test_utterance_col = None
    for col in test_df.columns:
        if col.lower() == "utterance":
            test_utterance_col = col
            break

    if test_utterance_col is None:
        logger.error(
            f"Could not find 'Utterance' column in test data. Available columns: {list(test_df.columns)}"
        )
        sys.exit(1)

    # Filter test data to only include selected utterances
    test_df["_utterance_stripped"] = test_df[test_utterance_col].str.strip()
    filtered_test = test_df[
        test_df["_utterance_stripped"].isin(selected_utterances)
    ].copy()
    filtered_test = filtered_test.drop(columns=["_utterance_stripped"])

    logger.info(f"Matched {len(filtered_test)} rows in test data")

    # Export filtered test data
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    filtered_test.to_csv(output_path, sep="\t", index=False, encoding="utf-8")

    if show_stats:
        # Calculate detailed statistics
        print("\n" + "=" * 80)
        print("NON-CONFLICTING TREATMENT WINNER FILTERING RESULTS")
        print("=" * 80)
        print("Selection Criteria: Treatment wins without conflicts")
        print("  - Both jobs win (gain > 0 for both), OR")
        print(
            "  - One job wins (gain > 0) and other is tie (gain == 0) or missing (NaN)"
        )
        print()
        print(f"SEVAL Results File: {seval_results_file}")
        print(f"  Total utterances: {total_count}")
        print(
            f"  Selected utterances: {len(filtered_seval)} ({len(filtered_seval)/total_count*100:.1f}%)"
        )
        print()

        # Breakdown by category
        both_win_count = (
            (seval_df[job1_gain_col] > 0) & (seval_df[job2_gain_col] > 0)
        ).sum()
        job1_win_job2_tie_or_missing = (
            (seval_df[job1_gain_col] > 0)
            & ((seval_df[job2_gain_col] == 0) | seval_df[job2_gain_col].isna())
        ).sum()
        job2_win_job1_tie_or_missing = (
            (seval_df[job2_gain_col] > 0)
            & ((seval_df[job1_gain_col] == 0) | seval_df[job1_gain_col].isna())
        ).sum()

        print("Breakdown by Win Pattern (Non-Conflicting Only):")
        print(
            f"  Both jobs win:                {both_win_count:>5} ({both_win_count/total_count*100:>5.1f}%)"
        )
        print(
            f"  Job 1 wins, Job 2 tie/missing: {job1_win_job2_tie_or_missing:>5} ({job1_win_job2_tie_or_missing/total_count*100:>5.1f}%)"
        )
        print(
            f"  Job 2 wins, Job 1 tie/missing: {job2_win_job1_tie_or_missing:>5} ({job2_win_job1_tie_or_missing/total_count*100:>5.1f}%)"
        )
        print()

        print(f"Test Data File: {test_data_file}")
        print(f"  Total rows: {len(test_df)}")
        print(
            f"  Matched rows: {len(filtered_test)} ({len(filtered_test)/len(test_df)*100:.1f}%)"
        )
        print()
        print(f"✓ Output File: {output_path}")
        print("=" * 80)

    logger.info(f"✓ Successfully exported {len(filtered_test)} rows to: {output_path}")


if __name__ == "__main__":
    fire.Fire(filter_treatment_winners)
