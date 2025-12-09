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

MATCHING STRATEGY:
CiteDCG data is organized by utterance only (no turn/hop/invocation markers).
Conversation data has nested structure: turns → hops → invocations → queries.
Matching relies on processing searches in order within each utterance to
maintain correct turn/hop/invocation alignment.

USAGE:

Command line:
    # Merge conversation with CiteDCG scores
    python merge_seval_results.py merge_citedcg_and_calculate_stats \\
        --conversation_file="conv_details.json" \\
        --citedcg_file="citedcg_scores.json" \\
        --output_file="full_analysis.json"

Programmatic:
    from merge_seval_results import merge_citedcg_and_calculate_stats
    
    # Merge conversation with CiteDCG scores
    merged_data, _ = merge_citedcg_and_calculate_stats(
        conversation_file="conv_details.json",
        citedcg_file="citedcg_scores.json",
        output_file="full_analysis.json"
    )
"""

import io
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

# Configure UTF-8 encoding BEFORE importing any packages that may wrap stdout
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import fire
from fire.core import FireExit

from .seval_plotting import generate_plot_statistics_from_utterance_details

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _merge_citedcg_into_conversation(
    conversation_file: str,
    citedcg_file: str,
    output_file: str
) -> Dict:
    """
    Internal: Merge CiteDCG scores into conversation details structure.
    
    This function ONLY assigns individual CiteDCG scores to search results.
    It does NOT calculate statistics. Use calculate_citedcg_statistics()
    separately for statistics calculation.
    
    Args:
        conversation_file: Path to conversation details JSON
            (from seval_analysis_toolkit.py)
        citedcg_file: Path to CiteDCG scores JSON
            (from get_seval_metrics.py extract_citedcg)
        output_file: Path for merged output JSON
    
    Returns:
        Merged conversation data with citedcg_score fields
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
        
        # Build query mapping (preserves search order for correct matching)
        logger.info("Building CiteDCG query map...")
        query_map = _build_citedcg_query_map(citedcg_data)
        
        # Count total searches
        total_searches = sum(
            len(data.get('searches', []))
            for data in query_map.values()
        )
        logger.info(
            f"  Built map for {len(query_map)} utterances "
            f"with {total_searches} total searches"
        )
    except Exception as e:
        logger.error(f"Failed to load CiteDCG file: {e}")
        raise
    
    # Merge scores into conversation data (scores only, no statistics)
    logger.info("Merging scores into conversation data...")
    merged_data = _add_citedcg_scores_to_conversation(
        conversation_data, query_map
    )
    
    # Write output
    logger.info("="*70)
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
    Build mapping from utterance to complete CiteDCG data.
    
    Simply maps normalized utterance text to the full CiteDCG entry,
    preserving all fields exactly as stored.
    
    Args:
        citedcg_data: List of CiteDCG entries from JSONL file
        
    Returns:
        Dictionary mapping normalized utterance to full CiteDCG entry:
        {
            "utterance text": {
                "utterance": "Utterance text",
                "searches": [...],
                "has_cite_dcg_scores": true,
                ...all other fields...
            }
        }
    """
    query_map = {}
    duplicate_count = 0
    
    for entry in citedcg_data:
        utterance = entry.get('utterance', '').strip().lower()
        
        if not utterance:
            logger.warning(
                f"Skipping CiteDCG entry with empty utterance "
                f"({len(entry.get('searches', []))} searches)"
            )
            continue
        
        if utterance in query_map:
            # Duplicate utterance - merge searches
            duplicate_count += 1
            existing_searches = query_map[utterance].get('searches', [])
            new_searches = entry.get('searches', [])
            existing_searches.extend(new_searches)
        else:
            # Store the complete entry as-is
            query_map[utterance] = entry
    
    if duplicate_count > 0:
        logger.info(
            f"Merged {duplicate_count} duplicate utterances in CiteDCG data"
        )
    
    return query_map
    
    return query_map


def _add_citedcg_scores_to_conversation(
    conversation_data: Dict,
    query_map: Dict[str, Dict]
) -> Dict:
    """
    Add CiteDCG scores to conversation data structure.
    
    This function ONLY assigns individual CiteDCG scores to search results.
    It does NOT calculate any statistics (averages, top-k, etc.).
    Statistics should be calculated separately using calculate_statistics_from_merged().
    
    IMPROVED MATCHING STRATEGY:
    The reasoning model can issue parallel queries in any order, so we cannot
    assume queries appear in the same order in conversation vs CiteDCG data.
    
    New approach:
    1. Match by utterance first
    2. For each conversation query (skip if 0 results):
       - Normalize tool+domain to same format (tool_domain style)
       - Search ALL CiteDCG searches for matching (tool+domain, query)
       - Mark matched CiteDCG search as "used" to avoid re-matching
    3. Validate result counts match when query found
    4. Assign scores by position within results
    
    Optimizations:
    - Pre-normalize and cache lowercase versions for fast comparison
    - Track used searches to skip already-matched entries
    - Only process queries with results (skip 0-result queries)
    
    Args:
        conversation_data: Conversation analysis data
        query_map: Map from utterance to ordered list of searches
        
    Returns:
        Updated conversation data with individual citedcg_score fields on results
    """
    merged = conversation_data.copy()
    
    # Extract conversation_id for debugging
    conversation_id = conversation_data.get('metadata', {}).get('conversation_id', 'UNKNOWN')
    
    # Track score assignment counts (for logging only)
    total_results_with_scores = 0
    total_results_without_scores = 0
    
    # Track matching statistics for debugging
    total_queries_processed = 0
    total_queries_matched = 0
    total_queries_no_match = 0
    utterances_not_in_dcg = 0  # Track utterances missing from DCG data
    no_match_utterances = set()  # Track unique utterances with queries that didn't match
    
    # Track result count mismatch statistics (SEVAL bug workaround)
    mismatch_utterances = set()  # Track unique utterances with mismatches
    mismatch_queries = 0  # Count queries with mismatches
    mismatch_details = []  # Store details for summary logging
    
    # Track multi-turn statistics
    is_multi_turn = False
    turns_with_hops = []  # Track which turn indices have hops
    turn_used_for_scoring = None  # Which turn was actually used for scoring
    
    # Process each turn
    eval_results = merged.get('evaluation_data_results', {})
    turns = eval_results.get('turns', [])
    
    # Detect multi-turn conversations
    if len(turns) > 1:
        is_multi_turn = True
    
    for turn_idx, turn in enumerate(turns, 1):
        # Skip turns with no hops (failed turns in retry scenarios)
        # For multi-turn conversations, only the successful turn has hops
        hops = turn.get('hops', [])
        if not hops:
            logger.debug(
                f"Skipping turn {turn_idx} (no hops - likely failed retry)"
            )
            continue
        
        # Track this turn as having hops
        turns_with_hops.append(turn_idx)
        
        # Get user input for this turn (this is the utterance/query_text)
        user_input = turn.get('user_input', '').strip().lower()
        
        # First, check if this turn has any search results to match
        # Count total results across all hops in this turn
        total_results_in_turn = sum(
            sum(
                inv.get('total_results', 0)
                for inv in hop.get('invocations', [])
            )
            for hop in hops
        )
        
        # If turn has no search results, skip it
        if total_results_in_turn == 0:
            logger.debug(
                f"Skipping turn {turn_idx} (no search results to match)"
            )
            continue
        
        # Mark this turn as the one used for scoring
        turn_used_for_scoring = turn_idx
        
        # Get CiteDCG data for this utterance
        citedcg_data = query_map.get(user_input, {})
        searches = citedcg_data.get('searches', [])
        
        if not searches:
            # Turn has search results but no CiteDCG data - not in DCG file
            utterances_not_in_dcg += 1
            logger.warning(
                f"Conversation ID: {conversation_id}\n"
                f"Turn {turn_idx} has {total_results_in_turn} search results "
                f"but utterance not found in CiteDCG data:\n"
                f"  Query: '{user_input}'\n"
                f"  Length: {len(user_input)} chars\n"
                f"  (Note: Only {len(query_map)} of conversations have "
                f"CiteDCG scores)"
            )
            # Skip this turn since we can't match scores
            continue
        
        # Pre-process CiteDCG searches: normalize and track usage
        # Format: List of dicts with normalized fields + usage tracking
        normalized_searches = []
        for search in searches:
            # Get content_domain_name from search level (no fallback - exact match only)
            content_domain_name = search.get('content_domain_name')
            plugin_name = search.get('plugin_name', '')
            
            normalized_searches.append({
                'original': search,
                'hop': search.get('hop', ''),  # Hop number from CiteDCG
                'plugin_name': plugin_name,
                'domain': search.get('domain', ''),  # For office365_search domain matching (files/people/etc.)
                'type': search.get('type', ''),  # For web search type matching (webpages, news, etc.)
                'content_domain_name': content_domain_name,  # For Graph Connectors
                'query_lower': search.get('query_string', '').lower(),
                'query_original': search.get('query_string', ''),
                'results': search.get('results', []),
                'used': False  # Track if already matched
            })
        
        # Process each hop in the turn
        for hop in turn.get('hops', []):
            # Process each invocation in the hop
            for invocation in hop.get('invocations', []):
                
                # Get tool name for this invocation
                invocation_tool = invocation.get('tool_name', '')
                
                # Process each query in the invocation
                for query in invocation.get('queries', []):
                    # NEW MATCHING STRATEGY:
                    # Parallel queries can be in any order, so we search
                    # all CiteDCG searches for each conversation query
                    
                    # Get query metadata
                    query_domain = query.get('domain', '')
                    query_text_original = query.get('query', '')
                    query_text_lower = query_text_original.lower()
                    query_results = query.get('results', [])
                    
                    # Skip queries with no results (nothing to score)
                    if not query_results:
                        logger.debug(
                            f"Skipping 0-result query: "
                            f"{invocation_tool}_{query_domain} / "
                            f"'{query_text_lower[:60]}'"
                        )
                        continue
                    
                    total_queries_processed += 1
                    
                    # Normalize tool+domain for matching CiteDCG data
                    # Conversation: tool_name="office365_search" + domain="files"
                    #   -> Match CiteDCG: plugin_name="office365_search_files", search_domain="office365_search_files"
                    # Conversation: tool_name="search_web" + domain="webpages"  
                    #   -> Match CiteDCG: plugin_name="search_web", search_domain="webpages"
                    
                    # Search for matching CiteDCG entry
                    matched_search = None
                    hop_number = hop.get('hop_number', '')
                    
                    for norm_search in normalized_searches:
                        # Skip already-used searches
                        if norm_search['used']:
                            continue
                        
                        # CRITICAL: Match hop number first
                        # CiteDCG hop is stored as string (e.g., "1", "2")
                        # Conversation hop_number is also string
                        hop_match = (str(norm_search.get('hop', '')) == str(hop_number))
                        if not hop_match:
                            continue
                        
                        # Match logic depends on plugin type:
                        # 1. search_web: Match on plugin_name="search_web", type (webpages/news/etc), and query
                        # 2. office365_search: Match on plugin_name="office365_search" AND domain (files/people/etc)
                        #    OR match combined format plugin_name="office365_search_files" etc.
                        # 3. Graph Connector (search_enterprise_connectors_*): 
                        #    Match on ContentDomainName (e.g., "Viva Learning") extracted from both sides
                        plugin_match = False
                        domain_match = False
                        type_match = True  # Default to True for non-web searches
                        
                        if invocation_tool == "search_web":
                            # Web search: match plugin_name AND type (to distinguish webpages from news, etc.)
                            plugin_match = (norm_search['plugin_name'] == "search_web")
                            domain_match = True  # Domain not checked for web search
                            
                            # Match type: conversation has domain="webpages", CiteDCG has type="search_web_webpages"
                            expected_type = f"search_web_{query_domain}"
                            type_match = (norm_search.get('type') == expected_type)
                            
                            # WORKAROUND: CiteDCG data for search_web has empty PluginInvocation
                            # (PluginInvocation is literally "search_web({})"), so query_string is empty.
                            # Fall back to matching by result reference_ids when query is unavailable.
                            if not norm_search['query_lower']:
                                # Extract reference_ids from conversation results
                                conv_ref_ids = set()
                                for result in query.get('results', []):
                                    ref_id = result.get('reference_id', '')
                                    if ref_id:
                                        conv_ref_ids.add(ref_id.lower())
                                
                                # Extract reference_ids from CiteDCG results
                                dcg_ref_ids = set()
                                for result in norm_search.get('results', []):
                                    ref_id = result.get('reference_id', '')
                                    if ref_id:
                                        dcg_ref_ids.add(ref_id.lower())
                                
                                # Match if there's significant overlap in reference_ids
                                # (at least one common reference_id, or both are empty)
                                if conv_ref_ids and dcg_ref_ids:
                                    ref_id_match = len(conv_ref_ids & dcg_ref_ids) > 0
                                else:
                                    # Both empty - not a valid match
                                    ref_id_match = False
                                
                                if plugin_match and domain_match and type_match and ref_id_match:
                                    matched_search = norm_search
                                    norm_search['used'] = True  # Mark as used
                                    break  # Found match via reference_ids, skip query_text check
                        elif "search_enterprise_connectors" in invocation_tool or "search_enterprise" in invocation_tool:
                            # Graph Connector: match on ContentDomainName
                            # Conversation: tool_name="search_enterprise_connectors_LearningAppConnectorV12", 
                            #               domain="Viva Learning" (extracted from sourceJson)
                            # CiteDCG: plugin_name="search_enterprise_connectors_LearningAppConnectorV12", 
                            #          content_domain_name="Viva Learning" (extracted from ContentDomain.Name)
                            plugin_match = (norm_search['plugin_name'] == invocation_tool)
                            domain_match = (norm_search.get('content_domain_name') == query_domain)
                        elif invocation_tool == "office365_search":
                            # Office365: Two possible formats in CiteDCG data:
                            # Format 1 (separate): plugin_name="office365_search", domain="files"
                            # Format 2 (combined): plugin_name="office365_search_files", domain=""
                            # Conversation always uses: tool_name="office365_search", domain="files"
                            citedcg_plugin = norm_search['plugin_name']
                            citedcg_domain = norm_search.get('domain', '')
                            
                            # Check Format 1: separate plugin_name and domain
                            if citedcg_plugin == "office365_search" and citedcg_domain == query_domain:
                                plugin_match = True
                                domain_match = True
                            # Check Format 2: combined plugin_name (e.g., "office365_search_files")
                            elif citedcg_plugin == f"office365_search_{query_domain}":
                                plugin_match = True
                                domain_match = True
                        elif invocation_tool == "fetch_file":
                            # fetch_file: Match on plugin_name only (no query or domain)
                            # CiteDCG: plugin_name="fetch_file", query_string="" (empty)
                            # Conversation: tool_name="fetch_file", domain="fetch_file", query="" (empty)
                            plugin_match = (norm_search['plugin_name'] == "fetch_file")
                            domain_match = True  # No domain check for fetch_file
                        else:
                            # Other plugins: combined format (plugin_name = tool_domain)
                            search_domain_key = f"{invocation_tool}_{query_domain}"
                            plugin_match = (norm_search['plugin_name'] == search_domain_key)
                            domain_match = True  # Domain is encoded in plugin_name
                        
                        # For fetch_file, match by hop and plugin only (no query text)
                        if invocation_tool == "fetch_file":
                            if plugin_match and domain_match and type_match:
                                matched_search = norm_search
                                norm_search['used'] = True  # Mark as used
                                break
                        elif (plugin_match and domain_match and type_match and 
                                norm_search['query_lower'] == query_text_lower):
                            matched_search = norm_search
                            norm_search['used'] = True  # Mark as used
                            break
                    
                    # Build display key for logging
                    if invocation_tool == "search_web":
                        search_domain_key = f"{invocation_tool}:{query_domain}"
                    elif "search_enterprise_connectors" in invocation_tool or "search_enterprise" in invocation_tool:
                        # Graph Connector: show full tool name with connector
                        search_domain_key = f"{invocation_tool}_{query_domain}"
                    else:
                        search_domain_key = f"{invocation_tool}_{query_domain}"
                    
                    # Track matching stats
                    if matched_search:
                        total_queries_matched += 1
                        
                        # Validate result counts match
                        search_result_count = len(matched_search['results'])
                        conv_result_count = len(query_results)
                        
                        # WORKAROUND: Handle result count mismatch due to SEVAL extraction bug
                        # This should not occur - indicates bug in CiteDCG extraction or conversation extraction
                        # When fixed upstream, this workaround can be removed.
                        if search_result_count != conv_result_count:
                            hop_number = hop.get('hop_number', 'unknown')
                            
                            # Track mismatch statistics
                            mismatch_utterances.add(user_input[:100])  # Use truncated utterance as key
                            mismatch_queries += 1
                            mismatch_details.append({
                                'conversation_id': conversation_id,
                                'utterance': user_input[:50],
                                'turn': turn_idx,
                                'hop': hop_number,
                                'tool_domain': search_domain_key,
                                'query': query_text_lower[:60],
                                'conv_results': conv_result_count,
                                'citedcg_results': search_result_count
                            })
                            
                            logger.warning(
                                f"Conversation ID: {conversation_id}\n"
                                f"Result count mismatch (SEVAL bug - applying workaround):\n"
                                f"  Utterance: '{user_input[:50]}'\n"
                                f"  Turn: {turn_idx}, Hop: {hop_number}\n"
                                f"  Tool+Domain: {search_domain_key}\n"
                                f"  Query: '{query_text_lower[:60]}'\n"
                                f"  Conversation results: {conv_result_count}\n"
                                f"  CiteDCG results: {search_result_count}\n"
                                f"  → Assigning top {conv_result_count} CiteDCG scores to conversation results"
                            )
                    else:
                        total_queries_no_match += 1
                        no_match_utterances.add(user_input[:100])  # Track unique utterances
                        hop_number = hop.get('hop_number', 'unknown')
                        
                        # For Graph Connectors, provide detailed debugging info
                        if "search_enterprise_connectors" in invocation_tool or "search_enterprise" in invocation_tool:
                            logger.warning(
                                f"Conversation ID: {conversation_id}\n"
                                f"No CiteDCG match for Graph Connector query:\n"
                                f"  Utterance: '{user_input[:50]}'\n"
                                f"  Turn: {turn_idx}, Hop: {hop_number}\n"
                                f"  Tool: {invocation_tool}\n"
                                f"  Domain: {query_domain}\n"
                                f"  Query: '{query_text_lower[:60]}'\n"
                                f"  Available CiteDCG searches: {len(normalized_searches)}\n"
                                f"  (Check if content_domain_name exists in CiteDCG for this connector)"
                            )
                        else:
                            logger.warning(
                                f"Conversation ID: {conversation_id}\n"
                                f"No CiteDCG match for conversation query:\n"
                                f"  Utterance: '{user_input[:50]}'\n"
                                f"  Turn: {turn_idx}, Hop: {hop_number}\n"
                                f"  Tool+Domain: {search_domain_key}\n"
                                f"  Query: '{query_text_lower[:60]}'\n"
                                f"  Result count: {len(query_results)}"
                            )                    # Add scores to each result by matching position
                    if matched_search:
                        search_results = matched_search['results']
                        
                        # WORKAROUND: Handle result count mismatch
                        # If CiteDCG has more results than conversation, assign top scores only
                        # This shouldn't happen - indicates SEVAL bug in extraction
                        if len(search_results) != len(query_results):
                            # Sort CiteDCG results by score (descending) to get top scores
                            sorted_search_results = sorted(
                                search_results,
                                key=lambda r: r.get('CiteDCGLLMLabel', 0),
                                reverse=True
                            )
                            # Use top N scores where N = conversation result count
                            search_results = sorted_search_results[:len(query_results)]
                        
                        for idx, result in enumerate(query_results):
                            # Match by position in the search's results
                            if idx < len(search_results):
                                score = search_results[idx].get(
                                    'CiteDCGLLMLabel'
                                )
                                if score is not None:
                                    # ONLY assign score - no statistics
                                    result['citedcg_score'] = score
                                    total_results_with_scores += 1
                                else:
                                    total_results_without_scores += 1
                            else:
                                total_results_without_scores += 1
                    else:
                        # Query didn't match - count all results as unmatched
                        total_results_without_scores += len(query_results)

    # Update summary with score assignment count (for logging only)
    eval_results = merged.get('evaluation_data_results', {})
    summary = eval_results.get('summary', {})
    if total_results_with_scores > 0:
        summary['results_with_citedcg_scores'] = total_results_with_scores
    
    # Add matching statistics to summary (for batch processor)
    summary['utterances_not_in_dcg'] = utterances_not_in_dcg
    summary['queries_matched'] = total_queries_matched
    summary['queries_no_match'] = total_queries_no_match
    summary['no_match_utterances'] = len(no_match_utterances)  # Utterances with unmatched queries
    summary['mismatch_queries'] = mismatch_queries
    summary['mismatch_utterances'] = len(mismatch_utterances)
    
    # Add multi-turn statistics to summary
    if is_multi_turn:
        summary['is_multi_turn'] = True
        summary['total_turns'] = len(turns)
        summary['turns_with_hops'] = turns_with_hops
        summary['turn_used_for_scoring'] = turn_used_for_scoring
        logger.info(
            f"  Multi-turn conversation: {len(turns)} turns, "
            f"turn {turn_used_for_scoring} used for scoring"
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

    # Log matching statistics
    total_results = total_results_with_scores + total_results_without_scores
    result_match_pct = (
        100.0 * total_results_with_scores / max(1, total_results)
    )
    query_match_pct = (
        100.0 * total_queries_matched / max(1, total_queries_processed)
    )
    
    logger.info(
        f"  Matched {total_queries_matched} of {total_queries_processed} "
        f"queries ({query_match_pct:.1f}%)"
    )
    logger.info(
        f"  Matched {total_results_with_scores} of {total_results} "
        f"search results ({result_match_pct:.1f}%)"
    )
    logger.info(
        f"  Assigned scores to {total_results_with_scores} results"
    )
    
    # Result-level breakdown
    logger.info("")
    logger.info("="*70)
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
    
    # Log result count mismatch summary (SEVAL bug workaround)
    if mismatch_queries > 0:
        logger.info("")
        logger.info("="*70)
        logger.info("  Result Count Mismatch Summary (SEVAL Bug Workaround):")
        logger.info(
            f"    Affected utterances: {len(mismatch_utterances)}"
        )
        logger.info(
            f"    Affected queries: {mismatch_queries}"
        )
        logger.info(
            f"    Workaround applied: Assigned top CiteDCG scores to conversation results"
        )
        logger.info(
            f"    Note: This indicates a bug in SEVAL extraction - should be fixed upstream"
        )
        logger.info("="*70)
    
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
) -> tuple[Dict, None]:
    """
    Merge CiteDCG scores into conversation.
    
    Note: stats_file and top_k parameters are kept for backward compatibility
    but are ignored. Statistics calculation has been moved to step 4/5.
    
    Args:
        conversation_file: Path to conversation details JSON
            (from seval_analysis_toolkit.py extract_conversation_details)
        citedcg_file: Path to CiteDCG scores JSON
            (from get_seval_metrics.py extract_per_result_citedcg)
        output_file: Path for merged output JSON
        stats_file: DEPRECATED - ignored for backward compatibility
        top_k: DEPRECATED - ignored for backward compatibility
    
    Returns:
        Tuple of (merged_data, None) for backward compatibility
    
    Example:
        merged_data, _ = merge_citedcg_and_calculate_stats(
            conversation_file="130949_control_conv_details.json",
            citedcg_file="130949_citedcg_scores_control.json",
            output_file="130949_merged_control.json"
        )
    """
    logger.info("="*70)
    logger.info("MERGING CITEDCG SCORES WITH CONVERSATION")
    logger.info("="*70)
    
    merged_data = _merge_citedcg_into_conversation(
        conversation_file=conversation_file,
        citedcg_file=citedcg_file,
        output_file=output_file
    )
    
    logger.info("\n" + "="*70)
    logger.info("✓ MERGE COMPLETE")
    logger.info("="*70)
    logger.info(f"Merged data: {output_file}")
    logger.info("="*70)
    
    return merged_data, None


def calculate_statistics_from_merged(
    merged_dir: str,
    top_k: int,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate CiteDCG statistics from merged conversation files.
    
    Reads all merged conversation files from a directory and calculates
    aggregate statistics for a specific top-k value, including:
    - Overall average scores
    - Per-conversation scores
    - Score distributions
    - Results coverage
    
    Args:
        merged_dir: Directory containing merged conversation JSON files
        top_k: Top-k value for calculating average scores
        output_file: Optional path to save statistics JSON
        
    Returns:
        Dictionary containing aggregate statistics
        
    Example:
        stats = calculate_statistics_from_merged(
            merged_dir="results/132951_conversation_w_citedcg_details",
            top_k=3,
            output_file="results/132951_statistics_k3.json"
        )
    """
    import json
    from collections import defaultdict
    from pathlib import Path
    
    merged_path = Path(merged_dir)
    if not merged_path.exists():
        raise ValueError(f"Merged directory does not exist: {merged_dir}")
    
    # Find all merged files
    merged_files = list(merged_path.rglob("*_merged.json"))
    if not merged_files:
        raise ValueError(f"No merged files found in: {merged_dir}")
    
    logger.info(f"Processing {len(merged_files)} merged files for k={top_k}")
    
    # Initialize aggregated statistics
    stats = {
        "merged_dir": str(merged_dir),
        "top_k": top_k,
        "total_utterances": len(merged_files),
        "utterances_with_scores": 0,
        "total_search_results": 0,
        "results_with_scores": 0,
        "per_hop": defaultdict(lambda: {
            "all_scores": [],
            "top_k_scores": [],
            "total_utterances": 0,
            "utterances_with_scores": 0,
            "utterances_empty": 0
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
    
    # Process each merged file (each represents one utterance/conversation)
    for merged_file in merged_files:
        try:
            with open(merged_file, 'r', encoding='utf-8') as f:
                merged_data = json.load(f)
            
            eval_results = merged_data.get("evaluation_data_results", {})
            
            utterance_has_scores = False
            
            # Collect scores organized by hop number
            # Track both hop_number (from source data) and hop_sequence (only hops with results)
            for turn in eval_results.get("turns", []):
                hop_sequence = 0  # Counter for hops with actual results
                for hop in turn.get("hops", []):
                    hop_number = hop.get("hop_number", 0)
                    if hop_number == 0:
                        continue  # Skip hops without valid hop_number
                    
                    hop_all_scores = []
                    hop_top_k_scores = []
                    
                    for invocation in hop.get("invocations", []):
                        for query in invocation.get("queries", []):
                            results = query.get("results", [])
                            stats["total_search_results"] += len(results)
                            
                            # Collect scores from this query
                            query_scores = []
                            for result in results:
                                score = result.get("citedcg_score")
                                if score is not None:
                                    stats["results_with_scores"] += 1
                                    hop_all_scores.append(score)
                                    query_scores.append(score)
                            
                            # Get top-k scores from this query's results
                            if query_scores:
                                query_scores_sorted = sorted(
                                    query_scores, reverse=True
                                )
                                top_k_from_query = query_scores_sorted[:top_k]
                                hop_top_k_scores.extend(top_k_from_query)
                    
                    # Add scores to this hop's statistics
                    if hop_all_scores:
                        utterance_has_scores = True
                        hop_sequence += 1  # Increment only for hops with results
                        
                        # Store by hop number (from source data, 1-indexed)
                        stats["per_hop"][hop_number]["all_scores"].extend(
                            hop_all_scores
                        )
                        stats["per_hop"][hop_number]["top_k_scores"].extend(
                            hop_top_k_scores
                        )
                        
                        # Also store by hop sequence (only counting non-empty hops)
                        stats["per_hop_sequence"][hop_sequence]["all_scores"].extend(
                            hop_all_scores
                        )
                        stats["per_hop_sequence"][hop_sequence]["top_k_scores"].extend(
                            hop_top_k_scores
                        )
            
            # Track utterances that contributed scores
            # Also track total utterances per hop and empty hops
            if utterance_has_scores:
                stats["utterances_with_scores"] += 1
            
            # Collect hop data for single-hop vs multi-hop classification
            utterance_hops_with_scores = []
            
            # Mark which hops this utterance touched (for total count)
            # and which had scores (for scores count)
            hop_sequence = 0
            for turn in eval_results.get("turns", []):
                for hop in turn.get("hops", []):
                    hop_number = hop.get("hop_number", 0)
                    if hop_number == 0:
                        continue  # Skip hops without valid hop_number
                    
                    # Count this utterance as having this hop number
                    stats["per_hop"][hop_number]["total_utterances"] += 1
                    
                    has_scores_in_hop = False
                    hop_scores_all = []
                    hop_scores_topk = []
                    
                    for invocation in hop.get("invocations", []):
                        for query in invocation.get("queries", []):
                            results = query.get("results", [])
                            query_scores = []
                            for result in results:
                                score = result.get("citedcg_score")
                                if score is not None:
                                    has_scores_in_hop = True
                                    hop_scores_all.append(score)
                                    query_scores.append(score)
                            
                            if query_scores:
                                query_scores_sorted = sorted(
                                    query_scores, reverse=True
                                )
                                top_k_from_query = query_scores_sorted[:top_k]
                                hop_scores_topk.extend(top_k_from_query)
                    
                    if has_scores_in_hop:
                        hop_sequence += 1
                        stats["per_hop"][hop_number]["utterances_with_scores"] += 1
                        seq_key = hop_sequence
                        stats["per_hop_sequence"][seq_key][
                            "utterances_with_scores"
                        ] += 1
                        
                        # Store for single/multi-hop classification
                        utterance_hops_with_scores.append({
                            'hop_sequence': hop_sequence,
                            'all_scores': hop_scores_all,
                            'top_k_scores': hop_scores_topk
                        })
                    else:
                        # This hop exists but has no scores (empty)
                        stats["per_hop"][hop_number]["utterances_empty"] += 1
            
            # Classify as single-hop or multi-hop based on non-empty hops
            num_nonempty_hops = len(utterance_hops_with_scores)
            if num_nonempty_hops == 1:
                # Single-hop utterance
                hop_data = utterance_hops_with_scores[0]
                stats["single_hop"][1]["all_scores"].extend(
                    hop_data['all_scores']
                )
                stats["single_hop"][1]["top_k_scores"].extend(
                    hop_data['top_k_scores']
                )
                stats["single_hop"][1]["utterances_count"] += 1
            elif num_nonempty_hops > 1:
                # Multi-hop utterance
                for hop_data in utterance_hops_with_scores:
                    seq = hop_data['hop_sequence']
                    stats["multi_hop"][seq]["all_scores"].extend(
                        hop_data['all_scores']
                    )
                    stats["multi_hop"][seq]["top_k_scores"].extend(
                        hop_data['top_k_scores']
                    )
                    stats["multi_hop"][seq]["utterances_count"] += 1
            
        except Exception as e:
            logger.warning(f"Failed to process {merged_file.name}: {e}")
            continue
    
    # Calculate per-hop averages and convert to regular dict
    per_hop_final = {}
    for hop_num in sorted(stats["per_hop"].keys()):
        hop_data = stats["per_hop"][hop_num]
        
        avg_all = None
        if hop_data["all_scores"]:
            avg_all = sum(hop_data["all_scores"]) / len(hop_data["all_scores"])
        
        avg_topk = None
        if hop_data["top_k_scores"]:
            avg_topk = (
                sum(hop_data["top_k_scores"]) / len(hop_data["top_k_scores"])
            )
        
        per_hop_final[hop_num] = {
            "total_utterances": hop_data["total_utterances"],
            "utterances_with_scores": hop_data["utterances_with_scores"],
            "utterances_empty": hop_data["utterances_empty"],
            "total_results_with_scores": len(hop_data["all_scores"]),
            "avg_all_scores": avg_all,
            "avg_topk_scores": avg_topk
        }
    
    stats["per_hop"] = per_hop_final
    
    # Calculate per-hop-sequence averages (only hops with results)
    per_hop_sequence_final = {}
    for hop_seq in sorted(stats["per_hop_sequence"].keys()):
        hop_data = stats["per_hop_sequence"][hop_seq]
        
        avg_all = None
        if hop_data["all_scores"]:
            avg_all = sum(hop_data["all_scores"]) / len(hop_data["all_scores"])
        
        avg_topk = None
        if hop_data["top_k_scores"]:
            avg_topk = (
                sum(hop_data["top_k_scores"]) / len(hop_data["top_k_scores"])
            )
        
        per_hop_sequence_final[hop_seq] = {
            "utterances_with_scores": hop_data["utterances_with_scores"],
            "total_results_with_scores": len(hop_data["all_scores"]),
            "avg_all_scores": avg_all,
            "avg_topk_scores": avg_topk
        }
    
    stats["per_hop_sequence"] = per_hop_sequence_final
    
    # Calculate single-hop averages
    single_hop_final = {}
    for hop_seq in sorted(stats["single_hop"].keys()):
        hop_data = stats["single_hop"][hop_seq]
        
        avg_all = None
        if hop_data["all_scores"]:
            avg_all = sum(hop_data["all_scores"]) / len(hop_data["all_scores"])
        
        avg_topk = None
        if hop_data["top_k_scores"]:
            avg_topk = (
                sum(hop_data["top_k_scores"]) / len(hop_data["top_k_scores"])
            )
        
        single_hop_final[hop_seq] = {
            "utterances_count": hop_data["utterances_count"],
            "total_results_with_scores": len(hop_data["all_scores"]),
            "avg_all_scores": avg_all,
            "avg_topk_scores": avg_topk
        }
    
    stats["single_hop"] = single_hop_final
    
    # Calculate multi-hop averages
    multi_hop_final = {}
    for hop_seq in sorted(stats["multi_hop"].keys()):
        hop_data = stats["multi_hop"][hop_seq]
        
        avg_all = None
        if hop_data["all_scores"]:
            avg_all = sum(hop_data["all_scores"]) / len(hop_data["all_scores"])
        
        avg_topk = None
        if hop_data["top_k_scores"]:
            avg_topk = (
                sum(hop_data["top_k_scores"]) / len(hop_data["top_k_scores"])
            )
        
        multi_hop_final[hop_seq] = {
            "utterances_count": hop_data["utterances_count"],
            "total_results_with_scores": len(hop_data["all_scores"]),
            "avg_all_scores": avg_all,
            "avg_topk_scores": avg_topk
        }
    
    stats["multi_hop"] = multi_hop_final
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Statistics saved to: {output_file}")
    
    return stats


def build_utterance_details_with_top_k(
    merged_dir: str,
    top_k_list: List[int],
    output_file: str,
    experiment: str = "control",
    existing_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build or update per-utterance details file with hop-level scores.
    
    This creates a JSON file containing:
    - Metadata about what k-values have been calculated
    - Per-utterance data with hop-level averages for each k-value
    
    If existing_file is provided, it will load existing data and add new
    k-values, avoiding recalculation.
    
    Args:
        merged_dir: Directory containing merged conversation JSON files
        top_k_list: List of top-k values to calculate
        output_file: Path to save utterance details JSON
        experiment: Experiment name ("control" or "treatment")
        existing_file: Optional path to existing details file to update
        
    Returns:
        Dictionary containing per-utterance details and metadata
        
    Example:
        details = build_utterance_details_with_top_k(
            merged_dir="results/merged/control",
            top_k_list=[1, 3, 5],
            output_file="results/133560_utterance_hop_citedcg_scores/control_utterance_details.json",
            experiment="control"
        )
    """
    import json
    from collections import defaultdict
    from pathlib import Path
    
    merged_path = Path(merged_dir)
    if not merged_path.exists():
        raise ValueError(f"Merged directory does not exist: {merged_dir}")
    
    # Load existing data if available
    existing_data = {}
    existing_k_values = set()
    
    if existing_file and Path(existing_file).exists():
        logger.info(f"Loading existing utterance details from: {existing_file}")
        with open(existing_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        existing_k_values = set(existing_data.get("metadata", {}).get("k_values_calculated", []))
        logger.info(f"  Existing k-values: {sorted(existing_k_values)}")
    
    # Determine which k-values need calculation
    requested_k_values = set(top_k_list)
    new_k_values = requested_k_values - existing_k_values
    
    if not new_k_values:
        logger.info("All requested k-values already calculated, reusing existing file")
        return existing_data
    
    logger.info(f"Calculating new k-values: {sorted(new_k_values)}")
    
    # Find all merged files
    merged_files = list(merged_path.rglob("*_merged.json"))
    if not merged_files:
        raise ValueError(f"No merged files found in: {merged_dir}")
    
    logger.info(f"Processing {len(merged_files)} utterances")
    
    # Initialize or update utterances dict
    utterances = existing_data.get("utterances", {})
    
    # Process each merged file (one per utterance)
    for merged_file in merged_files:
        utterance_id = merged_file.stem.replace("_merged", "")
        
        # Initialize utterance entry if new
        if utterance_id not in utterances:
            utterances[utterance_id] = {
                "utterance_id": utterance_id,
                "file": merged_file.name,
                "query_text": None,  # Will be populated from merged file
                "hops": {}  # {hop_index: {k: {all_avg, topk_avg, count}}}
            }
        
        try:
            with open(merged_file, 'r', encoding='utf-8') as f:
                merged_data = json.load(f)
            
            # Extract query text from metadata if not already set
            if utterances[utterance_id]["query_text"] is None:
                query_text = merged_data.get("metadata", {}).get(
                    "query_text", ""
                )
                utterances[utterance_id]["query_text"] = query_text
            
            eval_results = merged_data.get("evaluation_data_results", {})
            
            # Collect scores by hop for each turn
            hop_sequence = 0  # Only non-empty hops
            
            for turn in eval_results.get("turns", []):
                for hop in turn.get("hops", []):
                    # Get the original hop_number from the data (1-indexed)
                    hop_number = hop.get("hop_number", 0)
                    if hop_number == 0:
                        # Skip if hop_number is missing
                        continue
                    
                    hop_all_scores = []
                    
                    # Collect all scores from this hop
                    for invocation in hop.get("invocations", []):
                        for query in invocation.get("queries", []):
                            for result in query.get("results", []):
                                score = result.get("citedcg_score")
                                if score is not None:
                                    hop_all_scores.append(score)
                    
                    # Initialize hop entry if needed (use hop_number as key)
                    hop_key = str(hop_number)
                    if hop_key not in utterances[utterance_id]["hops"]:
                        utterances[utterance_id]["hops"][hop_key] = {}
                    
                    # Increment hop_sequence if this hop has scores
                    # (same sequence number for ALL k-values)
                    if hop_all_scores:
                        hop_sequence += 1
                    
                    # Calculate averages for each new k-value
                    for k in new_k_values:
                        if hop_all_scores:
                            avg_all = sum(hop_all_scores) / len(hop_all_scores)
                            
                            # Top-k average
                            sorted_scores = sorted(hop_all_scores, reverse=True)
                            top_k_scores = sorted_scores[:k]
                            avg_topk = sum(top_k_scores) / len(top_k_scores)
                            
                            utterances[utterance_id]["hops"][hop_key][str(k)] = {
                                "avg_all_scores": avg_all,
                                "avg_topk_scores": avg_topk,
                                "result_count": len(hop_all_scores),
                                "is_empty": False,
                                "hop_sequence": hop_sequence,
                                "hop_number": hop_number
                            }
                        else:
                            # Empty hop
                            utterances[utterance_id]["hops"][hop_key][str(k)] = {
                                "avg_all_scores": None,
                                "avg_topk_scores": None,
                                "result_count": 0,
                                "is_empty": True,
                                "hop_sequence": None,
                                "hop_number": hop_number
                            }
        
        except Exception as e:
            logger.error(f"Error processing {merged_file.name}: {e}")
            continue
    
    # Build metadata
    all_k_values = sorted(existing_k_values | requested_k_values)
    metadata = {
        "experiment": experiment,
        "k_values_calculated": all_k_values,
        "total_utterances": len(utterances),
        "merged_dir": str(merged_dir),
        "description": "Per-utterance hop-level CiteDCG score averages for multiple k-values"
    }
    
    # Build output structure
    output_data = {
        "metadata": metadata,
        "utterances": utterances
    }
    
    # Save to file
    logger.info(f"Saving utterance details to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Utterance details saved with k-values: {all_k_values}")
    
    return output_data


def find_paired_utterances_with_scores(
    control_details_file: str,
    treatment_details_file: str,
    output_file: str
) -> Dict[str, Any]:
    """
    Find utterances with scores in both control and treatment experiments.
    
    This function:
    1. Loads both control and treatment utterance details files
    2. Identifies utterances present in both experiments
    3. For each paired utterance, checks if it has scores (non-empty hops)
    4. Creates comprehensive comparison file with actual hop scores
    
    Args:
        control_details_file: Path to control utterance details JSON
        treatment_details_file: Path to treatment utterance details JSON
        output_file: Path to save paired utterances comparison JSON
        
    Returns:
        Dictionary containing:
        - metadata: Summary statistics and k-values
        - paired_data: Dict mapping query_text to control/treatment scores
        - summary_lists: Categorized query text lists for reference
        
    Output Structure:
        {
            "metadata": {
                "total_utterances": int,
                "paired_with_scores": int,
                "common_k_values": [1, 3, 5],
                ...
            },
            "paired_data": [
                {
                    "pair_id": 1,
                    "query_text": "...",
                    "control": {
                        "utterance_id": "...",
                        "hops": {
                            "0": {"1": {...}, "3": {...}, "5": {...}},
                            "1": {"1": {...}, "3": {...}, "5": {...}}
                        }
                    },
                    "treatment": {
                        "utterance_id": "...",
                        "hops": {...}
                    }
                },
                ...
            ],
            "summary_lists": {
                "paired_utterances": [...],
                "control_only": [...],
                "treatment_only": [...],
                "no_scores": [...]
            }
        }
    """
    logger.info("="*70)
    logger.info("Finding paired utterances with scores in both experiments")
    logger.info("="*70)
    
    # Load control details
    logger.info(f"Loading control details: {control_details_file}")
    with open(control_details_file, 'r', encoding='utf-8') as f:
        control_data = json.load(f)
    
    control_utterances = control_data.get("utterances", {})
    control_metadata = control_data.get("metadata", {})
    control_k_values = control_metadata.get("k_values_calculated", [])
    
    # Load treatment details
    logger.info(f"Loading treatment details: {treatment_details_file}")
    with open(treatment_details_file, 'r', encoding='utf-8') as f:
        treatment_data = json.load(f)
    
    treatment_utterances = treatment_data.get("utterances", {})
    treatment_metadata = treatment_data.get("metadata", {})
    treatment_k_values = treatment_metadata.get("k_values_calculated", [])
    
    logger.info(f"  Control utterances: {len(control_utterances)}")
    logger.info(f"  Treatment utterances: {len(treatment_utterances)}")
    
    # Find common k-values
    common_k_values = sorted(set(control_k_values) & set(treatment_k_values))
    if not common_k_values:
        raise ValueError(
            f"No common k-values found. Control: {control_k_values}, "
            f"Treatment: {treatment_k_values}"
        )
    logger.info(f"  Common k-values: {common_k_values}")
    
    # Helper function to check if utterance has scores
    def has_scores(utterance_data, k_value):
        """Check if utterance has at least one non-empty hop for given k."""
        hops = utterance_data.get("hops", {})
        k_str = str(k_value)
        
        for hop_idx, hop_data in hops.items():
            k_data = hop_data.get(k_str, {})
            if not k_data.get("is_empty", True):
                return True
        return False
    
    # Build mappings from query text to utterance data
    control_by_query = {}
    for utt_id, utt_data in control_utterances.items():
        query_text = utt_data.get("query_text", "").strip().lower()
        if query_text:
            control_by_query[query_text] = {
                "id": utt_id,
                "data": utt_data
            }
    
    treatment_by_query = {}
    for utt_id, utt_data in treatment_utterances.items():
        query_text = utt_data.get("query_text", "").strip().lower()
        if query_text:
            treatment_by_query[query_text] = {
                "id": utt_id,
                "data": utt_data
            }
    
    logger.info(f"  Control queries mapped: {len(control_by_query)}")
    logger.info(f"  Treatment queries mapped: {len(treatment_by_query)}")
    
    # Find all unique queries from both experiments
    all_queries = (set(control_by_query.keys()) |
                   set(treatment_by_query.keys()))
    
    # Categorize utterances and build paired data
    paired_data = []  # List of paired utterance objects
    control_only = []
    treatment_only = []
    no_scores = []
    
    # Use first common k-value to check for scores
    check_k = common_k_values[0]
    
    pair_id = 1
    for query_text in sorted(all_queries):
        in_control = query_text in control_by_query
        in_treatment = query_text in treatment_by_query
        
        has_control_scores = (in_control and has_scores(
            control_by_query[query_text]["data"], check_k
        ))
        has_treatment_scores = (in_treatment and has_scores(
            treatment_by_query[query_text]["data"], check_k
        ))
        
        if has_control_scores and has_treatment_scores:
            # Build paired data with full hop scores
            control_utt = control_by_query[query_text]["data"]
            treatment_utt = treatment_by_query[query_text]["data"]
            
            paired_data.append({
                "pair_id": pair_id,
                "query_text": query_text,
                "control": {
                    "utterance_id": control_utt.get("utterance_id", ""),
                    "file": control_utt.get("file", ""),
                    "hops": control_utt.get("hops", {})
                },
                "treatment": {
                    "utterance_id": treatment_utt.get("utterance_id", ""),
                    "file": treatment_utt.get("file", ""),
                    "hops": treatment_utt.get("hops", {})
                }
            })
            pair_id += 1
        elif has_control_scores and not has_treatment_scores:
            control_only.append(query_text)
        elif has_treatment_scores and not has_control_scores:
            treatment_only.append(query_text)
        else:
            no_scores.append(query_text)
    
    # Build output data
    output_data = {
        "metadata": {
            "control_file": control_details_file,
            "treatment_file": treatment_details_file,
            "total_utterances": len(all_queries),
            "paired_with_scores": len(paired_data),
            "control_only_with_scores": len(control_only),
            "treatment_only_with_scores": len(treatment_only),
            "no_scores_in_either": len(no_scores),
            "common_k_values": common_k_values,
            "description": (
                "Paired utterances with hop-level CiteDCG scores "
                "from both control and treatment experiments"
            )
        },
        "paired_data": paired_data,
        "summary_lists": {
            "paired_utterances": [p["query_text"] for p in paired_data],
            "control_only": control_only,
            "treatment_only": treatment_only,
            "no_scores": no_scores
        }
    }
    
    # Save to file
    logger.info(f"\nSaving paired utterances to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("PAIRING SUMMARY")
    logger.info("="*70)
    logger.info(f"Total unique utterances: {len(all_queries)}")
    logger.info(f"  ✓ Paired (scores in both): {len(paired_data)}")
    logger.info(f"  • Control only: {len(control_only)}")
    logger.info(f"  • Treatment only: {len(treatment_only)}")
    logger.info(f"  • No scores in either: {len(no_scores)}")
    logger.info("="*70)
    
    pct_paired = 100.0 * len(paired_data) / max(1, len(all_queries))
    logger.info(f"Paired coverage: {pct_paired:.1f}%")
    logger.info("="*70)
    
    return output_data


if __name__ == "__main__":
    # Use fire for command line arguments
    # Exposes the main merge_citedcg_and_calculate_stats function
    try:
        # Suppress Fire's automatic output serialization
        fire.Fire(
            {
                'merge_citedcg_and_calculate_stats':
                    merge_citedcg_and_calculate_stats,
                'build_utterance_details_with_top_k':
                    build_utterance_details_with_top_k,
                'generate_plot_statistics':
                    generate_plot_statistics_from_utterance_details,
                'find_paired_utterances_with_scores':
                    find_paired_utterances_with_scores
            },
            serialize=lambda x: None
        )
    except FireExit as e:
        # Handle Fire's exit (including --help) gracefully in debug mode
        # FireExit with code 0 means successful exit (like --help)
        # FireExit with non-zero code means error exit
        sys.exit(e.code)
