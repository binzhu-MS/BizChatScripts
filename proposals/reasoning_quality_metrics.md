# Metrics for Evaluating Chain-of-Thought Reasoning Models in BizChat Copilot

## Executive Summary

This document proposes a comprehensive evaluation framework for comparing reasoning models (control vs treatment) in BizChat Copilot. The framework separates **retrieval quality** from **retrieval efficiency**, enabling fair comparison of different reasoning strategies without biasing against multi-hop or thorough search approaches.

## Background

### Scenario
- Reasoning models perform **fanout search** across multiple domains (emails, chats, files, calendar events)
- Models formulate queries and make tool invocations based on user utterances
- Search results are evaluated by an **external LLM** (not the reasoning model itself) using **CiteDCG scoring**
- CiteDCG measures: "Likelihood this search result will be cited in the final response"
- Models synthesize final responses from top ~3-5 most relevant results

### Evaluation Goals
1. Assess which reasoning model better **collects relevant information** through queries and tool calls
2. Evaluate **query formulation quality** and **search strategy effectiveness**
3. Compare models fairly without penalizing thorough multi-hop reasoning
4. Distinguish retrieval quality from retrieval efficiency

## Core Metrics

This framework proposes **two primary metrics** that together provide comprehensive evaluation of reasoning quality:

### 1. Combined @k Across All Hops (Quality Metric)

**Definition:** Average of top-k CiteDCG scores from the **combined pool** of all search results across all non-empty hops for an utterance.

For each paired utterance, aggregate all search results from all non-empty hops, sort by CiteDCG score, and calculate the mean of the top k scores.

**Key Values:**
- **@3**: Average of top 3 results across all hops (primary, matches synthesis behavior)
- **@5**: Average of top 5 results across all hops (broader quality assessment)

**Measures:** "Can you find the best information?" (capability, recall of good results)

### 2. Average All Results Across All Hops (Efficiency Metric)

**Definition:** Average CiteDCG score of **all search results** retrieved across all non-empty hops for an utterance.

For each paired utterance, aggregate all search results from all non-empty hops and calculate the mean score across the entire pool.

**Measures:** "Is what you find typically good quality?" (efficiency, precision, query targeting)

### Justification for Combined @k

#### 1. **Focus on Capability, Not Strategy**
- Measures: "Did the reasoning model's tool invocations collectively retrieve relevant information?"
- Doesn't care whether results came from 1 hop or 7 hops
- Evaluates the model's ability to formulate effective queries, regardless of search pattern

#### 2. **Removes Hop Count Bias**
- **Problem with per-hop averaging**: `average(@3 across hops)` penalizes models that do more hops
  - Example: 3 hops with scores [3.0, 2.5, 2.0] → average = 2.5
  - Example: 1 hop with score [2.5] → average = 2.5
  - The first model found MORE good results but gets same score
- **Combined @3 solution**: All results compete equally
  - Picks absolute best 3 results regardless of which hop found them
  - Model that found [3.0, 2.5, 2.0] across hops gets same treatment as model that found them in one hop

#### 3. **Empty Hop Neutral**
- Empty hops (invocations with 0 results) simply contribute zero good results
- Natural penalty: Wasted search effort doesn't help, but doesn't artificially lower quality scores
- Can't distinguish search failures from bad queries, but that's fine for quality metric

#### 4. **Aligns with Synthesis Behavior**
- BizChat models select top ~3-5 results to cite in final response
- Combined @3 directly measures quality of the pool from which synthesis occurs
- Real-world impact: Better combined @3 → better source material → better final responses

#### 5. **Fair Comparison of Strategies**
- **Multi-hop/fanout strategy**: Comprehensive search across domains → many results → best 3 measured
- **Focused strategy**: Targeted search → fewer results → best 3 measured
- **Both evaluated on**: Absolute quality of top results, not search pattern

### Justification for Average All Results

#### 1. **Measures Query Targeting Quality**
- **Focus**: Are your queries retrieving relevant results, or just casting a wide net?
- **High average**: Targeted, well-formulated queries that find relevant information
- **Low average**: Broad or poorly-targeted queries retrieving lots of low-quality results
- **Natural efficiency metric**: Rewards precision without requiring invocation counts

#### 2. **Complements Combined @k**
- **Combined @3**: Measures best-case quality (top results you'll actually use)
- **Average all**: Measures typical quality (overall search effectiveness)
- **Gap analysis**: Large gap = good at finding gems in noise; Small gap = consistently high quality
- **Together**: Complete picture of retrieval capability vs efficiency

#### 3. **Natural Empty Hop Handling**
- **Empty hops** (0 results): Contribute 0 results to average → neutral impact (correct!)
- **Bad queries** (many low-quality results): Lower the average → penalty (correct!)
- **Distinguishes**: "No results" (neutral) from "junk results" (negative)
- **Fair comparison**: Doesn't penalize search failures, does penalize poor targeting

#### 4. **Reveals Search Strategy Quality**
- **High avg + high @3**: Efficient, targeted search (best case)
- **Low avg + high @3**: Finds gems but retrieves lots of noise (inefficient but capable)
- **High avg + low @3**: Consistent quality but misses peaks (overly conservative?)
- **Low avg + low @3**: Poor search strategy overall (needs improvement)

#### 5. **Measures Precision vs Recall Trade-off**
- **Combined @k**: Recall-focused (did you find any good stuff?)
- **Average all**: Precision-focused (is most of what you found good?)
- **Classic IR metrics**: Together capture quality/quantity balance
- **Production relevance**: High precision → less noise for synthesis, better UX

### Known Loopholes and Limitations

#### ⚠️ Loophole 1: Combined @k Encourages Extensive Searchnsive Search
- **Issue**: Metric rewards "more is better" - exhaustive search will always maximize combined @k
- **Why it's a problem**: 
  - Could encourage inefficient brute-force strategies
  - More tool calls → higher latency, cost, resource usage
  - No incentive to be strategic or targeted
- **Mitigation**: Average all results metric naturally penalizes noise; also track invocation counts

#### ⚠️ Loophole 2: Average All Results Discourages Exploration
- **Issue**: Metric penalizes retrieving any low-quality results, even if also finding great ones
- **Why it's a problem**:
  - Could encourage overly conservative search (retrieve very few results)
  - May discourage exploratory queries that cast wider net
  - Model might skip potentially valuable tangential searches
- **Mitigation**: Use Combined @k alongside to ensure models still find the best results

#### ⚠️ Loophole 3: No Credit for Consistency
- **Issue**: Model gets same score for [3.0, 0.5, 0.5] as [2.5, 2.4, 2.3] when using @3
- **Why it's a problem**: 
  - Second distribution shows more consistent retrieval quality
  - Reliability matters in production systems
  - Doesn't capture "depth" of good results
- **Mitigation**: Also report @5, @10, or score distribution metrics

#### ⚠️ Loophole 4: Ignores Hop Sequencing
- **Issue**: No distinction between finding best result in hop 1 vs hop 7
- **Why it's a problem**:
  - Early good results → faster response, better UX
  - Late good results → model struggled, luck-based recovery
  - Doesn't reward efficient reasoning progression
- **Mitigation**: Track "Hop 1 @3" and "Best hop position" as supplementary metrics

#### ⚠️ Limitation 1: External Scoring Dependency
- **Issue**: CiteDCG scores from external LLM may have bias or noise
- **Why it's a problem**:
  - Evaluation quality depends on external LLM's judgment
  - May not perfectly align with final response quality
  - Could introduce systematic bias toward certain result types
- **Mitigation**: Validate with human evaluation on subset; analyze score distributions

#### ⚠️ Limitation 2: Doesn't Measure Synthesis Quality
- **Issue**: Only measures search result quality, not how well model uses them
- **Why it's a problem**:
  - Model could retrieve great results but synthesize poorly
  - Doesn't capture reasoning model's core value-add
  - Misses citation accuracy, answer completeness, coherence
- **Mitigation**: This is intentional - separate metrics needed for synthesis evaluation

## Supplementary Metrics

To provide additional context beyond the two core metrics:

### 1. **Hop Statistics**
- **Total hops**: Number of non-empty hops per utterance
- **Average hops**: Mean across paired utterances
- **Empty hop rate**: Percentage of hops with 0 results
- Shows search pattern differences

### 2. **Best Hop Position**
- **Definition**: Hop number where the highest-scoring result was found
- Measures: "How quickly did you find the best result?"
- Rewards efficient early discovery
- Lower is better

### 3. **Hop 1 Quality**
- **Definition**: Average of top 3 results from first non-empty hop only
- Measures: "Quality of initial search strategy"
- Complements combined @3 by showing early performance
- High Hop 1 @3 + high Combined @3 = efficient and thorough

### 4. **Recovery Analysis**
- **Definition**: Percentage of utterances where Hop 1 @3 is below threshold but Combined @3 exceeds Hop 1 @3 by meaningful margin
- Measures: "Can model recover from poor initial search?"
- Shows adaptive reasoning capability
- Distinguishes luck from strategy

### 5. **Invocation Breakdown**
- Total invocations per utterance
- Invocations per hop
- Invocations per domain (email/chat/file/calendar)
- Shows search behavior patterns

## Recommended Evaluation Framework

### Primary Metrics (The Core Pair)
1. **Combined @3** - Best-case retrieval quality (what you'll use)
2. **Average All Results** - Typical retrieval quality (search efficiency)

### Secondary Quality Metrics
- Combined @5 (broader quality assessment)
- Hop 1 @3 (initial strategy quality)
- Gap analysis (Combined @3 - Average All)

### Supplementary Metrics
- Average hops per utterance
- Best hop position
- Empty hop rate
- Total invocations per utterance

### Pattern Analysis
- Win/loss/tie counts on Combined @3
- Recovery rate analysis
- Multi-hop vs single-hop breakdowns
- Domain-specific performance

### Statistical Tests
- Paired t-test on Combined @3 (is difference significant?)
- Cohen's d effect size (how large is the difference?)
- Wilcoxon signed-rank test (non-parametric alternative)

## Example Comparison Report Structure

```markdown
# Job 133560: Control vs Treatment Comparison

## Core Metrics (438 paired utterances)

### Combined @3 (Quality)
- Control: 2.45 ± 0.82
- Treatment: 2.67 ± 0.75
- Difference: +0.22 (p=0.001, Cohen's d=0.28)
- Win/Loss/Tie: 245/178/15

### Average All Results (Efficiency)
- Control: 1.82 ± 0.65
- Treatment: 1.76 ± 0.68
- Difference: -0.06 (p=0.234, n.s.)
- Win/Loss/Tie: 198/215/25

### Gap Analysis (Combined @3 - Average All)
- Control: 0.63 (finds gems in moderate noise)
- Treatment: 0.91 (finds gems but retrieves more low-quality results)
- Interpretation: Treatment casts wider net, gets better peaks but lower precision

## Secondary Quality Metrics

### Hop 1 @3
- Control: 2.38 ± 0.85
- Treatment: 2.41 ± 0.83
- Difference: +0.03 (p=0.423, n.s.)

## Supplementary Metrics

### Search Pattern
- Average hops: Control 2.1, Treatment 3.4
- Empty hop rate: Control 8%, Treatment 12%
- Total invocations: Control 5.8, Treatment 7.1

### Best Hop Position
- Control: 1.4 (median=1)
- Treatment: 1.8 (median=1)
- Analysis: Both find best results early, treatment does more follow-up

## Interpretation

Treatment model achieves **9% higher Combined @3** (better peak quality) but **3% lower Average All Results** (lower precision).

- **Quality vs Efficiency Trade-off**: Treatment finds better results but retrieves more noise
- **Gap Analysis**: Treatment has 44% larger gap (0.91 vs 0.63), indicating less targeted search
- **Search Strategy**: Treatment casts wider net with more exploratory queries
- **Invocation Cost**: Treatment uses 22% more tool calls for the quality improvement
- **Hop 1 Performance**: Similar initial strategies, treatment continues broader exploration
- **Recovery**: Treatment shows 15% higher recovery rate from poor initial hops

## Recommendation

[Depends on whether quality improvement justifies efficiency cost]
```

## Visualization Recommendations

- **Scatter plot**: Control vs Treatment Combined @3 with 1:1 reference line
- **Histogram**: Distribution of score differences (treatment - control)
- **Box plot**: Score distributions by hop position
- **Bar chart**: Win/loss/tie breakdown by score buckets
- **Trajectory plot**: Hop 1 → Combined @3 improvement paths

## Limitations and Future Work

### Current Limitations
1. **No synthesis evaluation** - Only measures search quality, not final response quality
2. **External scoring dependency** - Relies on CiteDCG LLM's judgment accuracy
3. **No domain weighting** - Treats email/chat/file results equally
4. **Static threshold** - CiteDCG scores may not be calibrated across domains

### Future Enhancements
1. **Human evaluation correlation** - Validate Combined @3 against human relevance judgments
2. **Final response quality** - Add metrics for synthesis quality, citation accuracy
3. **Domain-specific analysis** - Break down by email/chat/file performance
4. **Temporal patterns** - Analyze if reasoning improves/degrades over conversation turns
5. **Query quality scoring** - Direct evaluation of generated query strings
6. **Cost-benefit analysis** - Model latency, API costs vs quality gains

## Conclusion

The **dual-metric framework** (Combined @k + Average All Results) provides comprehensive evaluation of reasoning model quality:

**Combined @k:**
- ✅ Measures capability: "Can you find the best information?"
- ✅ Removes bias against multi-hop strategies
- ✅ Aligns with synthesis behavior (top results actually used)
- ✅ Evaluates recall of high-quality results

**Average All Results:**
- ✅ Measures efficiency: "Is what you find typically good?"
- ✅ Natural penalty for poorly-targeted queries
- ✅ Rewards precision without requiring invocation counts
- ✅ Reveals search strategy quality through gap analysis

**Together:**
- ✅ Complete quality vs efficiency picture
- ✅ Prevents gaming either metric individually
- ✅ Enables nuanced understanding of reasoning strategies
- ✅ Supports informed trade-off decisions between capability and precision

---

**Document Version**: 1.0  
**Date**: November 27, 2025  
**Author**: BizChat Copilot Evaluation Team  
**Status**: Proposal for Discussion
