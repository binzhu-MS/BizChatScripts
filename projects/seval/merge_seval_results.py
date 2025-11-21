"""
SEVAL Results Merger

This module provides utilities for merging different SEVAL analysis outputs:
- Conversation details (from seval_analysis_toolkit.py)
- CiteDCG scores (from get_seval_metrics.py extract_citedcg)
- Statistics calculation from merged data

The merger creates a unified JSON output with:
- All conversation data
- Scores attached to each search result
- Aggregated statistics in metadata and summary sections

USAGE:

Command line:
    # Merge conversation with CiteDCG scores and calculate statistics
    python merge_seval_results.py merge_citedcg_and_calculate_stats \\
        --conversation_file="conv_details.json" \\
        --citedcg_file="citedcg_scores.json" \\
        --output_file="full_analysis.json" \\
        --stats_file="statistics.json"

Programmatic:
    from merge_seval_results import merge_citedcg_and_calculate_stats
    
    # Merge conversation with CiteDCG and calculate statistics
    merged_data, stats = merge_citedcg_and_calculate_stats(
        conversation_file="conv_details.json",
        citedcg_file="citedcg_scores.json",
        output_file="full_analysis.json",
        stats_file="statistics.json"
    )
"""

import json
import logging
import sys
from typing import Any, Dict, List, Optional

import fire
from fire.core import FireExit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _merge_citedcg_into_conversation(
    conversation_file: str,
    citedcg_file: str,
    output_file: str,
    top_k: int = 5
) -> Dict:
    """
    Internal: Merge CiteDCG scores into conversation details structure.
    
    Args:
        conversation_file: Path to conversation details JSON
            (from seval_analysis_toolkit.py)
        citedcg_file: Path to CiteDCG scores JSON
            (from get_seval_metrics.py extract_citedcg)
        output_file: Path for merged output JSON
        top_k: Number of top results to consider for top-k average (default: 5)
    """
    logger.info("Starting merge operation...")
    logger.info(f"  Conversation file: {conversation_file}")
    logger.info(f"  CiteDCG file: {citedcg_file}")
    logger.info(f"  Output file: {output_file}")
    
    # Load conversation data
    logger.info("Loading conversation data...")
    try:
        with open(conversation_file, 'r', encoding='utf-8') as f:
            conversation_data = json.load(f)
        turns = conversation_data.get(
            'evaluation_data_results', {}
        ).get('turns', [])
        logger.info(f"  Loaded conversation with {len(turns)} turns")
    except Exception as e:
        logger.error(f"Failed to load conversation file: {e}")
        raise
    
    # Load CiteDCG scores from JSONL format (one JSON object per line)
    logger.info("Loading CiteDCG scores...")
    try:
        citedcg_data = []
        with open(citedcg_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        citedcg_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Skipping invalid JSON on line {line_num}: {e}"
                        )
        
        # Build query -> results mapping
        query_map = _build_citedcg_query_map(citedcg_data)
        # Count total results
        total_cit_results = sum(
            len(results) for results in query_map.values()
        )
        logger.info(
            f"  Loaded {total_cit_results} results with CiteDCG scores"
        )
    except Exception as e:
        logger.error(f"Failed to load CiteDCG file: {e}")
        raise
    
    # Merge scores into conversation data
    logger.info("Merging scores into conversation data...")
    merged_data = _add_citedcg_scores_to_conversation(
        conversation_data, query_map, top_k=top_k
    )
    
    # Write output
    logger.info(f"Writing merged output to {output_file}...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        logger.info("✓ Merge completed successfully!")
    except Exception as e:
        logger.error(f"Failed to write output file: {e}")
        raise
    
    return merged_data


def _build_citedcg_query_map(
    citedcg_data: List[Dict]
) -> Dict[str, Dict]:
    """
    Build hierarchical mapping from query text to search data.
    
    New structure uses the enhanced CiteDCG extraction that groups by:
    1. Utterance (user question)
    2. Search domain + query string (each specific search)
    3. Position in results list
    
    Args:
        citedcg_data: List of query groups with hierarchical search data
        
    Returns:
        Dictionary mapping normalized query text to search structure:
        {
            utterance: {
                'searches': [
                    {
                        'search_domain': 'office365_search_files',
                        'plugin_name': 'office365_search_files',
                        'query_string': 'actual query text',
                        'results': [result1, result2, ...]
                    },
                    ...
                ]
            }
        }
    """
    query_map = {}
    
    for query_group in citedcg_data:
        utterance = query_group.get('utterance', '').strip().lower()
        searches = query_group.get('searches', [])
        
        if utterance and searches:
            query_map[utterance] = {'searches': searches}
    
    return query_map


def _add_citedcg_scores_to_conversation(
    conversation_data: Dict,
    query_map: Dict[str, Dict],
    top_k: int = 5
) -> Dict:
    """
    Add CiteDCG scores to conversation data structure.
    
    Matches results by:
    1. Utterance (normalized user question)
    2. Search domain (e.g., 'office365_search_files')
    3. Query string (the actual search query)
    4. Position in results list

    Args:
        conversation_data: Conversation details data structure
        query_map: Map from utterance to CiteDCG search data
        top_k: Number of top results to consider for top-k average (default: 5)

    Updates:
    - Adds 'citedcg_score' field to each search result
    - Adds average scores to turn/invocation level
    - Adds overall statistics to metadata and summary
    
    Args:
        conversation_data: Conversation analysis data
        query_map: Hierarchical mapping from query text to search data
        
    Returns:
        Updated conversation data with scores
    """
    merged = conversation_data.copy()
    
    # Track statistics
    total_results_with_scores = 0
    total_results_without_scores = 0
    total_score_sum = 0
    total_nonempty_hops = 0
    turn_stats = []
    
    # Track matching statistics for debugging
    total_queries_processed = 0
    total_queries_matched = 0
    total_queries_no_match = 0
    
    # Process each turn
    eval_results = merged.get('evaluation_data_results', {})
    turns = eval_results.get('turns', [])
    
    for turn in turns:
        turn_score_sum = 0
        turn_results_with_scores = 0
        turn_all_scores = []  # Collect all scores for top-k calculation
        hop_stats = []
        
        # Get user input for this turn (this is the query text)
        user_input = turn.get('user_input', '').strip().lower()
        
        # Process each hop in the turn
        for hop in turn.get('hops', []):
            hop_score_sum = 0
            hop_results_with_scores = 0
            hop_all_scores = []  # Collect all scores for hop-level top-k
            invocation_stats = []
            
            # Process each invocation in the hop
            for invocation in hop.get('invocations', []):
                inv_score_sum = 0
                inv_results_with_scores = 0
                
                # Process each query in the invocation
                for query in invocation.get('queries', []):
                    # MATCHING HIERARCHY:
                    # 1. Utterance: Match by user_input (turn_index=1)
                    # 2. Turn/Hop/Invocation: Implicit via nested loop order
                    # 3. Search: Match by domain + query_string
                    # 4. Results: Match by position/index in results array
                    
                    total_queries_processed += 1
                    
                    # Get CiteDCG search data for this utterance
                    citedcg_data = query_map.get(user_input, {})
                    searches = citedcg_data.get('searches', [])
                    
                    # Get query metadata
                    query_domain = query.get('domain', '')
                    query_text = query.get('query', '')
                    
                    # Normalize query for matching
                    normalized_query = query_text.strip().lower()
                    
                    # Find matching search by domain and query string (exact match only)
                    matched_search = None
                    search_domain_key = f"office365_search_{query_domain}"
                    
                    for search in searches:
                        search_domain = search.get('search_domain', '')
                        search_query = search.get(
                            'query_string', ''
                        ).strip().lower()
                        
                        # Match by domain and query string
                        if (search_domain == search_domain_key and
                                search_query == normalized_query):
                            matched_search = search
                            break
                    
                    # Track matching stats
                    if matched_search:
                        total_queries_matched += 1
                    else:
                        total_queries_no_match += 1
                    
                    # Add scores to each result by matching position
                    if matched_search:
                        search_results = matched_search.get('results', [])
                        for idx, result in enumerate(
                            query.get('results', [])
                        ):
                            # Match by position in the search's results
                            if idx < len(search_results):
                                score = search_results[idx].get(
                                    'CiteDCGLLMLabel'
                                )
                                if score is not None:
                                    result['citedcg_score'] = score
                                    
                                    inv_score_sum += score
                                    inv_results_with_scores += 1
                                    hop_score_sum += score
                                    hop_results_with_scores += 1
                                    hop_all_scores.append(score)  # For hop-level top-k
                                    turn_score_sum += score
                                    turn_results_with_scores += 1
                                    turn_all_scores.append(score)  # For turn-level top-k
                                    total_score_sum += score
                                    total_results_with_scores += 1
                                else:
                                    total_results_without_scores += 1
                            else:
                                total_results_without_scores += 1
                    else:
                        # Query didn't match - count all results as unmatched
                        total_results_without_scores += len(
                            query.get('results', [])
                        )
                
                # Add invocation-level statistics
                if inv_results_with_scores > 0:
                    avg = round(inv_score_sum / inv_results_with_scores, 3)
                    invocation['results_with_scores'] = inv_results_with_scores
                    invocation['avg_citedcg_score'] = avg
                    invocation_stats.append({
                        'invocation_number': invocation.get('invocation_number'),
                        'avg_score': avg,
                        'results_with_scores': inv_results_with_scores
                    })
            
            # Add hop-level statistics
            if hop_results_with_scores > 0:
                hop_avg = round(hop_score_sum / hop_results_with_scores, 3)
                hop['results_with_scores'] = hop_results_with_scores
                hop['avg_citedcg_score'] = hop_avg
                
                # Calculate hop-level top-k average
                hop_top_k_avg = None
                if hop_all_scores:
                    # Sort scores in descending order and take top k
                    sorted_scores = sorted(hop_all_scores, reverse=True)
                    top_scores = sorted_scores[:min(top_k, len(sorted_scores))]
                    if top_scores:
                        hop_top_k_avg = round(sum(top_scores) / len(top_scores), 3)
                
                hop[f'avg_top_{top_k}_citedcg_score'] = hop_top_k_avg
                hop[f'top_{top_k}_count'] = len(top_scores) if hop_top_k_avg else 0
                
                hop_stats.append({
                    'hop_number': hop.get('hop_number'),
                    'avg_score': hop_avg,
                    f'avg_top_{top_k}_score': hop_top_k_avg,
                    f'top_{top_k}_count': len(top_scores) if hop_top_k_avg else 0,
                    'results_with_scores': hop_results_with_scores,
                    'invocations': invocation_stats
                })
        
        # Add turn-level statistics (averaged across non-empty hops)
        total_hops = len(turn.get('hops', []))
        nonempty_hops = [h for h in hop_stats if h.get('results_with_scores', 0) > 0]
        nonempty_hop_count = len(nonempty_hops)
        total_nonempty_hops += nonempty_hop_count
        
        # Calculate total results across all hops
        total_turn_results = sum(
            h.get('total_results', 0) for h in turn.get('hops', [])
        )
        
        turn['total_results'] = total_turn_results
        turn['results_with_scores'] = turn_results_with_scores
        turn['total_hops'] = total_hops
        turn['nonempty_hops'] = nonempty_hop_count
        
        # Calculate hop-level averages (average of hop averages, excluding empty hops)
        nonempty_hop_stats = {}
        
        if nonempty_hop_count > 0:
            # Get avg_citedcg_score from each non-empty hop
            hop_avgs = []
            hop_top_k_avgs = []
            
            for hop in turn.get('hops', []):
                if hop.get('results_with_scores', 0) > 0:
                    if 'avg_citedcg_score' in hop:
                        hop_avgs.append(hop['avg_citedcg_score'])
                    if f'avg_top_{top_k}_citedcg_score' in hop:
                        hop_top_k_avgs.append(hop[f'avg_top_{top_k}_citedcg_score'])
            
            # Average the hop averages
            if hop_avgs:
                nonempty_hop_stats['hop_avg_citedcg_score'] = round(sum(hop_avgs) / len(hop_avgs), 3)
            if hop_top_k_avgs:
                nonempty_hop_stats[f'hop_avg_top_{top_k}_citedcg_score'] = round(sum(hop_top_k_avgs) / len(hop_top_k_avgs), 3)
        
        if nonempty_hop_stats:
            turn['nonempty_hop_stats'] = nonempty_hop_stats
        
        stats_entry = {
            'turn_number': turn.get('turn_number'),
            'total_hops': total_hops,
            'nonempty_hops': nonempty_hop_count,
            'results_with_scores': turn_results_with_scores,
            'hops': hop_stats
        }
        
        if nonempty_hop_stats:
            stats_entry['nonempty_hop_stats'] = nonempty_hop_stats
        
        turn_stats.append(stats_entry)

    # Add overall statistics to summary (rebuild to control property order)
    summary = eval_results.get('summary', {})
    # Preserve existing properties in desired order
    ordered_summary = {}
    if 'total_turns' in summary:
        ordered_summary['total_turns'] = summary['total_turns']
    if 'total_hops' in summary:
        ordered_summary['total_hops'] = summary['total_hops']
    # Add nonempty_hops right after total_hops
    ordered_summary['nonempty_hops'] = total_nonempty_hops
    if 'total_tool_invocations_count' in summary:
        ordered_summary['total_tool_invocations_count'] = summary['total_tool_invocations_count']
    if 'total_queries' in summary:
        ordered_summary['total_queries'] = summary['total_queries']
    if 'total_search_results' in summary:
        ordered_summary['total_search_results'] = summary['total_search_results']
    if total_results_with_scores > 0:
        ordered_summary['results_with_citedcg_scores'] = total_results_with_scores
    # Replace summary with ordered version
    eval_results['summary'] = ordered_summary

    # Calculate overall average for metadata and logging (not added to summary)
    overall_avg = None
    if total_results_with_scores > 0:
        overall_avg = round(total_score_sum / total_results_with_scores, 3)

    # Add overall statistics to metadata
    if total_results_with_scores > 0:
        merged['metadata']['avg_citedcg_score'] = overall_avg
        merged['metadata']['results_with_citedcg_scores'] = (
            total_results_with_scores
        )

    # Update cited_reference_ids with scores (before adding citedcg_statistics)
    if 'cited_reference_ids' in merged:
        # Build a map of reference_id -> citedcg_score from all results
        ref_score_map = {}
        for turn in turns:
            for hop in turn.get('hops', []):
                for invocation in hop.get('invocations', []):
                    for query in invocation.get('queries', []):
                        for result in query.get('results', []):
                            ref_id = result.get('reference_id')
                            score = result.get('citedcg_score')
                            if ref_id and score is not None:
                                ref_score_map[ref_id] = score
        
        # Enrich cited references with their scores
        cited_refs = []
        for ref_id in merged.get('cited_reference_ids', []):
            ref_entry = {'reference_id': ref_id}
            if ref_id in ref_score_map:
                ref_entry['citedcg_score'] = ref_score_map[ref_id]
            cited_refs.append(ref_entry)
        
        merged['cited_references_with_scores'] = cited_refs

    # Add detailed turn statistics
    if turn_stats:
        overall_score = (
            overall_avg if total_results_with_scores > 0 else None
        )
        merged['citedcg_statistics'] = {
            'overall_avg_score': overall_score,
            'total_results_scored': total_results_with_scores,
            'turn_statistics': turn_stats
        }
    
    # Log matching statistics
    total_results = total_results_with_scores + total_results_without_scores
    result_match_pct = (
        100.0 * total_results_with_scores / max(1, total_results)
    )
    
    logger.info(
        f"  Matched {total_results_with_scores} of {total_results} "
        f"search results ({result_match_pct:.1f}%)"
    )
    avg_msg = overall_avg if total_results_with_scores > 0 else 'N/A'
    logger.info(f"  Overall average CiteDCG score: {avg_msg}")
    
    # Result-level breakdown
    logger.info("")
    logger.info("  Search Result Matching Details:")
    logger.info(
        f"    Total search results: {total_results}"
    )
    logger.info(
        f"    ✓ Successfully matched: {total_results_with_scores}"
    )
    logger.info(
        f"    ✗ Not matched: {total_results_without_scores}"
    )
    
    # Determine overall matching status based on search RESULTS
    if total_results_without_scores == 0 and total_results > 0:
        status_icon = "✅"
        status_msg = "ALL SEARCH RESULTS MATCHED"
    elif total_results == 0:
        status_icon = "⚠️"
        status_msg = "NO SEARCH RESULTS TO MATCH"
    else:
        status_icon = "✅" if result_match_pct >= 95.0 else "⚠️"
        status_msg = (
            f"RESULT MATCHING COMPLETE "
            f"({result_match_pct:.1f}% of {total_results} results matched)"
        )
    
    logger.info("")
    logger.info(f"  {status_icon} Status: {status_msg}")
    
    # Reconstruct merged dict with proper key order:
    # cited_references_with_scores right after cited_reference_ids
    if 'cited_references_with_scores' in merged:
        ordered_merged = {}
        cited_refs_scores = merged['cited_references_with_scores']
        
        for key, value in merged.items():
            if key == 'cited_references_with_scores':
                continue  # Skip, will add after cited_reference_ids
            
            ordered_merged[key] = value
            
            # Add cited_references_with_scores right after this key
            if key == 'cited_reference_ids':
                ordered_merged['cited_references_with_scores'] = (
                    cited_refs_scores
                )
        
        return ordered_merged
    
    return merged


def _calculate_citedcg_statistics(
    merged_file: str,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Internal: Calculate detailed CiteDCG statistics from merged SEVAL data.
    
    This generates:
    - Overall average CiteDCG scores
    - Per-turn average scores
    - Per-invocation average scores
    - Score distribution
    - Results coverage statistics
    
    Args:
        merged_file: Path to merged SEVAL data JSON
            (output from merge_conversation_with_citedcg)
        output_file: Optional path to save statistics JSON.
            If None, prints to console.
        
    Returns:
        Dictionary containing calculated statistics
    """
    logger.info(f"Calculating statistics from: {merged_file}")
    
    # Load merged data
    try:
        with open(merged_file, 'r', encoding='utf-8') as f:
            merged_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load merged file: {e}")
        raise
    
    # Extract metadata
    metadata = merged_data.get('metadata', {})
    
    # Initialize statistics structure
    stats = {
        'source_file': merged_file,
        'conversation_metadata': {
            'conversation_id': metadata.get('conversation_id'),
            'exp_name': metadata.get('exp_name'),
            'query_text': metadata.get('query_text'),
            'seval_job_id': metadata.get('seval_job_id')
        },
        'overall_statistics': {
            'total_turns': 0,
            'total_hops': 0,
            'total_invocations': 0,
            'total_queries': 0,
            'total_search_results': 0,
            'results_with_citedcg': 0,
            'avg_citedcg_overall': None,
            'score_distribution': {}
        },
        'turn_statistics': [],
        'round_statistics': []
    }
    
    all_scores = []
    score_counts = {}
    
    # Process turns
    eval_results = merged_data.get('evaluation_data_results', {})
    turns = eval_results.get('turns', [])
    stats['overall_statistics']['total_turns'] = len(turns)
    
    # Detect top_k value from the first hop that has it
    top_k = None
    for turn in turns:
        for hop in turn.get('hops', []):
            for key in hop.keys():
                if key.startswith('avg_top_') and key.endswith('_citedcg_score'):
                    top_k = int(key.replace('avg_top_', '').replace('_citedcg_score', ''))
                    break
            if top_k:
                break
        if top_k:
            break
    
    for turn in turns:
        # Build turn_stat with dynamic field names
        turn_stat = {
            'turn_number': turn.get('turn_number'),
            'user_input': turn.get('user_input', ''),
            'avg_citedcg_score': turn.get('avg_citedcg_score'),
            'results_with_scores': turn.get('results_with_scores', 0),
            'hops': []
        }
        
        # Add top_k fields if they exist
        if top_k:
            top_k_score_key = f'avg_top_{top_k}_citedcg_score'
            top_k_count_key = f'top_{top_k}_count'
            if top_k_score_key in turn:
                turn_stat[top_k_score_key] = turn.get(top_k_score_key)
            if top_k_count_key in turn:
                turn_stat[top_k_count_key] = turn.get(top_k_count_key, 0)
        
        # Process hops
        for hop in turn.get('hops', []):
            hop_stat = {
                'hop_number': hop.get('hop_number'),
                'avg_citedcg_score': hop.get('avg_citedcg_score'),
                'results_with_scores': hop.get('results_with_scores', 0),
                'invocations': []
            }
            
            # Add top_k fields if they exist
            if top_k:
                top_k_score_key = f'avg_top_{top_k}_citedcg_score'
                top_k_count_key = f'top_{top_k}_count'
                if top_k_score_key in hop:
                    hop_stat[top_k_score_key] = hop.get(top_k_score_key)
                if top_k_count_key in hop:
                    hop_stat[top_k_count_key] = hop.get(top_k_count_key, 0)
            
            stats['overall_statistics']['total_hops'] += 1
            
            # Process invocations in this hop
            for invocation in hop.get('invocations', []):
                inv_stat = {
                    'invocation_number': invocation.get('invocation_number'),
                    'tool_name': invocation.get('tool_name'),
                    'avg_citedcg_score': invocation.get('avg_citedcg_score'),
                    'results_with_scores': invocation.get(
                        'results_with_scores', 0
                    ),
                    'queries_count': len(invocation.get('queries', []))
                }
                
                hop_stat['invocations'].append(inv_stat)
                overall = stats['overall_statistics']
                overall['total_invocations'] += 1
                overall['total_queries'] += inv_stat['queries_count']
                
                # Collect all individual scores for distribution
                for query in invocation.get('queries', []):
                    for result in query.get('results', []):
                        overall['total_search_results'] += 1
                        
                        score = result.get('citedcg_score')
                        if score is not None:
                            overall['results_with_citedcg'] += 1
                            all_scores.append(score)
                            
                            # Count score distribution
                            score_key = f"{score:.1f}"
                            score_counts[score_key] = (
                                score_counts.get(score_key, 0) + 1
                            )
            
            turn_stat['hops'].append(hop_stat)
        
        stats['turn_statistics'].append(turn_stat)
    
    # Get overall_stats reference
    overall_stats = stats['overall_statistics']
    
    # Calculate overall average
    if all_scores:
        avg_score = round(sum(all_scores) / len(all_scores), 3)
        overall_stats['avg_citedcg_overall'] = avg_score
        overall_stats['score_distribution'] = dict(
            sorted(score_counts.items())
        )
    else:
        # No scores available
        overall_stats['avg_citedcg_overall'] = 0.0
        overall_stats['score_distribution'] = {}
    
    # Round statistics (multi-turn conversations group turns into rounds)
    # For now, each turn is a round.
    # Can be enhanced later for multi-round analysis.
    total_rounds = metadata.get('total_rounds', len(turns))
    stats['overall_statistics']['total_rounds'] = total_rounds
    
    round_stat = {
        'round_number': 1,
        'turns_in_round': len(turns),
        'avg_citedcg_score': overall_stats['avg_citedcg_overall'],
        'results_with_scores': overall_stats['results_with_citedcg']
    }
    stats['round_statistics'].append(round_stat)
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("SEVAL STATISTICS SUMMARY")
    logger.info("="*70)
    
    conv_meta = stats['conversation_metadata']
    logger.info(f"Conversation ID: {conv_meta['conversation_id']}")
    logger.info(f"Experiment: {conv_meta['exp_name']}")
    logger.info(f"Query: {conv_meta['query_text']}")
    logger.info("-"*70)
    
    overall = stats['overall_statistics']
    logger.info(f"Total Turns: {overall['total_turns']}")
    logger.info(f"Total Invocations: {overall['total_invocations']}")
    logger.info(f"Total Queries: {overall['total_queries']}")
    logger.info(f"Total Search Results: {overall['total_search_results']}")
    logger.info(f"Results with CiteDCG: {overall['results_with_citedcg']}")
    coverage = 100.0 * overall['results_with_citedcg'] / max(
        1, overall['total_search_results']
    )
    logger.info(f"Coverage: {coverage:.1f}%")
    logger.info("-"*70)
    logger.info(f"Average CiteDCG Score: {overall['avg_citedcg_overall']}")
    
    # Display per-turn statistics with top-k
    if stats['turn_statistics']:
        logger.info("\nPer-Turn Statistics:")
        for turn_stat in stats['turn_statistics']:
            turn_num = turn_stat['turn_number']
            avg_score = turn_stat['avg_citedcg_score']
            
            # Find top_k fields dynamically
            top_k_avg = None
            top_k_cnt = 0
            top_k_val = None
            for key in turn_stat.keys():
                if key.startswith('avg_top_') and key.endswith('_citedcg_score'):
                    top_k_avg = turn_stat[key]
                    top_k_val = int(key.replace('avg_top_', '').replace('_citedcg_score', ''))
                elif key.startswith('top_') and key.endswith('_count'):
                    top_k_cnt = turn_stat[key]
            
            logger.info(f"  Turn {turn_num}:")
            logger.info(f"    Average: {avg_score}")
            if top_k_avg is not None:
                logger.info(f"    Top-{top_k_val} Average: {top_k_avg} (from {top_k_cnt} results)")
    
    logger.info("-"*70)
    logger.info("\nScore Distribution:")
    
    for score, count in overall['score_distribution'].items():
        logger.info(f"  {score}: {count} results")
    logger.info("="*70)
    
    # Save if output file specified
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            logger.info(f"\n✓ Statistics saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to write statistics file: {e}")
            raise
    
    return stats


def merge_citedcg_and_calculate_stats(
    conversation_file: str,
    citedcg_file: str,
    output_file: str,
    stats_file: Optional[str] = None,
    top_k: int = 5
) -> tuple[Dict, Dict[str, Any]]:
    """
    Merge CiteDCG scores into conversation and calculate comprehensive
    statistics.
    
    This is the main entry point that performs both operations:
    1. Merges conversation details with CiteDCG scores
    2. Calculates comprehensive statistics from merged data
    
    Args:
        conversation_file: Path to conversation details JSON
            (from seval_analysis_toolkit.py extract_conversation_details)
        citedcg_file: Path to CiteDCG scores JSON
            (from get_seval_metrics.py extract_per_result_citedcg)
        output_file: Path for merged output JSON
        stats_file: Optional path to save statistics JSON.
            If None, statistics are returned but not saved.
        top_k: Number of top results to consider for top-k average (default: 5)
    
    Returns:
        Tuple of (merged_data, statistics)
    
    Example:
        merged, stats = merge_citedcg_and_calculate_stats(
            conversation_file="130949_control_conv_details.json",
            citedcg_file="130949_citedcg_scores_control.json",
            output_file="130949_merged_control.json",
            stats_file="130949_statistics_control.json",
            top_k=5
        )
    """
    logger.info("="*70)
    logger.info("SEVAL DATA PROCESSING")
    logger.info("="*70)
    
    # Step 1: Merge conversation with CiteDCG scores
    logger.info(f"\nStep 1/2: Merging conversation with CiteDCG scores (top-k={top_k})...")
    merged_data = _merge_citedcg_into_conversation(
        conversation_file=conversation_file,
        citedcg_file=citedcg_file,
        output_file=output_file,
        top_k=top_k
    )
    
    # Step 2: Calculate CiteDCG statistics from merged data
    logger.info("\nStep 2/2: Calculating CiteDCG statistics...")
    stats = _calculate_citedcg_statistics(
        merged_file=output_file,
        output_file=stats_file
    )
    
    logger.info("\n" + "="*70)
    logger.info("✓ PROCESSING COMPLETE")
    logger.info("="*70)
    logger.info(f"Merged data: {output_file}")
    if stats_file:
        logger.info(f"Statistics: {stats_file}")
    logger.info("="*70)
    
    return merged_data, stats


if __name__ == "__main__":
    # Use fire for command line arguments
    # Exposes the main merge_citedcg_and_calculate_stats function
    try:
        # Suppress Fire's automatic output serialization
        fire.Fire(
            {
                'merge_citedcg_and_calculate_stats':
                    merge_citedcg_and_calculate_stats
            },
            serialize=lambda x: None
        )
    except FireExit as e:
        # Handle Fire's exit (including --help) gracefully in debug mode
        # FireExit with code 0 means successful exit (like --help)
        # FireExit with non-zero code means error exit
        sys.exit(e.code)
