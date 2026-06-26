# Reasoning Model Evaluation Metrics Discussion

**Document Purpose:** Complete record of metrics discussion for evaluating CoT reasoning models in BizChat Copilot  
**Date:** November 27-28, 2025  
**Status:** In Progress - Paused for other work, resume from this document  

---

## Table of Contents
1. [Background & Context](#background--context)
2. [Core Metrics Framework](#core-metrics-framework)
3. [Multi-Hop Capability Metric](#multi-hop-capability-metric)
4. [Across-Hop Strategy Metrics](#across-hop-strategy-metrics)
5. [Open Questions & Next Steps](#open-questions--next-steps)

---

## Background & Context

### What We're Evaluating

**Scenario:**
- Reasoning models perform **fanout search** across multiple domains (emails, chats, files, calendar events)
- Models formulate queries and make tool invocations based on user utterances
- Search results are evaluated by an **external LLM** (not the reasoning model itself) using **CiteDCG scoring**
- CiteDCG measures: "Likelihood this search result will be cited in the final response"
- Models synthesize final responses from top ~3-5 most relevant results

**NOT Evaluating:**
- Model's ranking ability (we don't have access to model's internal ranking)
- Final response quality (separate metrics already exist for this)

**ARE Evaluating:**
- How well the reasoning model formulates queries and selects tools/domains
- Quality of information collected (measured by external CiteDCG scorer)
- Reasoning strategy effectiveness (single vs multi-hop, recovery, adaptive refinement)

**Key Point:** The @1, @3, @5 scores are **external evaluations** of the top-ranked results by the search system, NOT the reasoning model's ranking.

### Dataset
- **Job 133560**: Control vs Treatment comparison
- **438 paired utterances**: Both experiments have scores
- **520 control-only**: Only control has scores
- **74 treatment-only**: Only treatment has scores
- **277 no scores**: Neither experiment has scores
- **Hop structure**: Sequential non-empty hop positions (1, 2, 3...) with scores stored per k-value

### Evaluation Goal
Determine which reasoning model (control vs treatment) better collects relevant information through queries and tool calls, with focus on:
1. **Quality**: Can you find the best information?
2. **Efficiency**: Is what you find typically good quality?
3. **Multi-hop capability**: Can you recover from poor initial searches?
4. **Strategy**: How do you explore and refine across hops?

---

## Core Metrics Framework

### Primary Metric Pair (Complementary Quality + Efficiency)

#### 1. Combined @k Across All Hops (Quality Metric)

**Definition:**
Average of top-k CiteDCG scores from the **combined pool** of all search results across all non-empty hops for an utterance.

For each paired utterance, aggregate all search results from all non-empty hops, sort by CiteDCG score, and calculate the mean of the top k scores.

**Key Values:**
- **@3**: Average of top 3 results across all hops (primary, matches synthesis behavior)
- **@5**: Average of top 5 results across all hops (broader quality assessment)

**Measures:** "Can you find the best information?" (capability, recall of good results)

**Why this metric:**
1. **Focus on capability, not strategy**: Evaluates ability to formulate effective queries regardless of hop count
2. **Removes hop count bias**: Doesn't penalize multi-hop strategies (all results compete equally)
3. **Empty hop neutral**: Empty hops contribute zero results (natural, not artificial penalty)
4. **Aligns with synthesis**: Models use top ~3-5 results, this directly measures that pool quality
5. **Fair comparison**: Multi-hop vs single-hop strategies evaluated on absolute quality

**Known loopholes:**
- ⚠️ Encourages extensive search (more searching → more chances to find gems)
- Mitigation: Pair with Average All Results metric

#### 2. Average All Results Across All Hops (Efficiency Metric)

**Definition:**
Average CiteDCG score of **all search results** retrieved across all non-empty hops for an utterance.

For each paired utterance, aggregate all search results from all non-empty hops and calculate the mean score across the entire pool.

**Measures:** "Is what you find typically good quality?" (efficiency, precision, query targeting)

**Why this metric:**
1. **Measures query targeting quality**: High average = targeted queries, low average = casting wide net
2. **Complements Combined @k**: Together provide quality vs efficiency picture
3. **Natural empty hop handling**: Empty hops (0 results) contribute nothing (neutral), bad queries (junk results) lower average (penalty)
4. **Reveals search strategy**: Gap between Combined @k and Average All shows noise tolerance
5. **Measures precision vs recall**: Combined @k = recall, Average All = precision

**Known loopholes:**
- ⚠️ Discourages exploration (penalizes any low-quality results even if also finding great ones)
- Mitigation: Use Combined @k alongside to ensure best results still found

**Gap Analysis (Combined @3 - Average All):**
- **Large gap**: Model finds gems but retrieves lots of noise (capable but inefficient)
- **Small gap**: Model consistently retrieves high-quality results (targeted, efficient)
- Reveals: Different search strategies without making value judgment

### Recommended Usage

**For Paired Comparisons:**
- Run separate paired t-tests for Combined @3 and Average All
- Do NOT compare the gap directly (it's derived, not independent)
- Interpret together for complete picture

**Decision Framework:**

| Combined @3 | Average All | Interpretation                                               |
| ----------- | ----------- | ------------------------------------------------------------ |
| ✅ Better    | ✅ Better    | Treatment dominates (better quality AND efficiency)          |
| ✅ Better    | ❌ Worse     | Trade-off: Quality vs efficiency (thorough but noisy)        |
| ❌ Worse     | ✅ Better    | Trade-off: Precision vs recall (conservative but consistent) |
| ❌ Worse     | ❌ Worse     | Control dominates (worse on both dimensions)                 |

**Supplementary Metrics:**
- Combined @5 (broader quality assessment)
- Hop 1 @3 (initial strategy quality)
- Gap analysis (search strategy characterization)
- Hop statistics (counts, empty hop rates)
- Invocation counts (search volume)

---

## Multi-Hop Capability Metric

### The Core Question

**Goal:** Single metric with p-value that captures multi-hop problem-solving capability

**What we want to test:**
When the reasoning model's initial search fails, can it recognize the failure and successfully recover through multi-hop reasoning?

### Recovery Improvement Score (RECOMMENDED)

#### Definition

**Universal Recovery Score (Primary Approach):**

Calculate for ALL paired utterances, regardless of hop 1 quality:
```
Recovery Score = (Best @3 across all hops) - (Hop 1 @3)
```

**Interpretation:**
- **Positive score**: Multi-hop reasoning added value (found better results later)
- **Zero/negative**: Hop 1 was peak quality (no benefit from additional hops)
- **Larger positive**: Greater multi-hop capability

#### Why This Is The Best Single Metric

**1. Directly Measures Multi-Hop Capability**
- Only meaningful when model does multiple hops
- Captures ability to recognize poor results and improve
- Distinguishes adaptive reasoning from lucky first tries
- Focuses on value-add of multi-hop strategy

**2. Clean Paired Statistical Comparison**
- ✅ Every utterance has exactly one score (control) and one score (treatment)
- ✅ Perfect for paired t-test → clean p-value
- ✅ Win/loss/tie counts straightforward
- ✅ Effect size (Cohen's d) interpretable

**3. Captures Complete Recovery Process**
- **Recognition**: Model must detect hop 1 was suboptimal (evidenced by doing hop 2+)
- **Action**: Model must reformulate query or change strategy
- **Success**: Model must actually find better results
- All three steps required for positive score

**4. Natural Interpretation**
- Positive = "multi-hop reasoning helped"
- Zero = "hop 1 was sufficient" (efficient)
- Negative = "hop 1 was best, later hops wasted" (inefficient)

**5. Practical Business Value**
- High recovery score = robust reasoning that handles difficult queries
- Low recovery score = model struggles to self-correct
- Directly relates to user experience

#### The Pairing Problem & Solution

**❌ PROBLEM with threshold-based filtering:**

Original idea: "Filter to utterances where Hop 1 @3 < median"
- **Issue**: Creates unpaired data
- **Example**: Control Hop 1 = 1.2 (below median), Treatment Hop 1 = 2.8 (above median)
  - Only control gets a score, treatment excluded → cannot compare
- **Problem**: Loses paired structure needed for statistical test

**✅ SOLUTION 1: Combined Threshold (Ensures Pairing)**

1. Pool all Hop 1 @3 scores from BOTH control AND treatment
2. Calculate median across combined pool
3. Utterance is "challenging" if EITHER control OR treatment Hop 1 @3 < median
4. Calculate Recovery Score for BOTH experiments on all challenging utterances

**Ensures:**
- Every challenging utterance has BOTH scores
- Includes cases where one model struggled but the other didn't (reveals robustness)
- Measures: "In situations where at least one model had difficulty, who recovers better?"

**✅ SOLUTION 2: Universal Recovery Score (RECOMMENDED)**

Calculate recovery for ALL paired utterances (no filtering):
```
Recovery Score = (Best @3 - Hop 1 @3) for every utterance
```

**Advantages:**
- ✅ Always paired (every utterance has both scores)
- ✅ Captures full spectrum:
  - Good hop 1 + improvement = strategic exploration
  - Poor hop 1 + improvement = successful recovery
  - Good hop 1 + no improvement = efficient (hop 1 sufficient)
  - Poor hop 1 + no improvement = failed recovery
- ✅ Natural interpretation (positive = value-add)
- ✅ No arbitrary threshold
- ✅ Maximum statistical power (uses all data)

**Trade-off:**
- Includes utterances where hop 1 was already good (score ≈ 0)
- These dilute signal from true recovery cases

**Mitigation:**
- Primary analysis: Universal Recovery Score (all utterances)
- Subgroup analysis: Filter to utterances where BOTH control AND treatment Hop 1 < median
- This shows: Overall multi-hop value + specific recovery capability

#### Statistical Analysis

**Primary Test (Universal Recovery Score):**
```
For each of 438 paired utterances:
- Control Recovery Score = Control (Best @3 - Hop 1 @3)
- Treatment Recovery Score = Treatment (Best @3 - Hop 1 @3)
- Difference = Treatment - Control

Paired t-test:
- Null: Mean difference = 0 (no difference in multi-hop capability)
- Alternative: Mean difference ≠ 0 (treatment differs from control)
- Output: t-statistic, p-value, Cohen's d, 95% CI
```

**Win/Loss/Tie:**
- Treatment better: Treatment Recovery > Control Recovery
- Control better: Control Recovery > Treatment Recovery
- Tie: Equal recovery scores

**Subgroup Analysis (Poor Hop 1 Cases):**
```
1. Calculate median Hop 1 @3 (combined pool)
2. Filter: Keep utterances where BOTH control AND treatment Hop 1 < median
3. Run same paired t-test on this subset
4. Compare: Does recovery capability specifically differ in challenging cases?
```

#### Interpretation Guide

**Scenario A: Treatment Significantly Better**
- Treatment mean recovery: +0.85
- Control mean recovery: +0.42
- Difference: +0.43 (p=0.002, Cohen's d=0.52, medium-large effect)
- Win/Loss/Tie: 245/178/15 (treatment wins 56%)

**Interpretation:**
- Treatment has significantly better multi-hop problem-solving
- When searches extend beyond hop 1, treatment finds better improvements
- Medium-large effect size = practically meaningful difference

**Scenario B: No Significant Difference**
- Treatment mean recovery: +0.58
- Control mean recovery: +0.54
- Difference: +0.04 (p=0.421, n.s.)

**Interpretation:**
- Both models have similar multi-hop capability
- Multi-hop reasoning improvements not substantially different
- No evidence treatment handles multi-hop better

**Scenario C: High Variance**
- Treatment mean recovery: +0.65 (SD=1.2)
- Control mean recovery: +0.48 (SD=0.8)
- Difference: +0.17 (p=0.083, marginal)

**Interpretation:**
- Treatment sometimes recovers much better, but inconsistent
- High variance suggests recovery is situation-dependent
- May need subgroup analysis (which utterances does treatment excel at?)

#### Complementary Metrics (Secondary Analysis)

While Recovery Score is the primary metric for p-value, also report:

1. **Recovery Attempt Rate**
   - % of utterances with hop 2+ attempts
   - Tests: Does model try multi-hop reasoning?

2. **Recovery Success Rate**
   - % of utterances with positive recovery score
   - Tests: When model tries multi-hop, does it succeed?

3. **Recovery Magnitude Distribution**
   - Histogram of recovery scores
   - Shows: How much improvement when successful?

4. **Non-Recovery Analysis**
   - Cases where Recovery Score ≤ 0
   - Why: Stopped at hop 1? Later hops worse? Empty hops?

#### Why NOT Other Candidates

**Improvement Rate (% that improved):**
- ❌ Binary metric, loses magnitude information
- ❌51% with tiny gains = 51% with huge gains
- ❌ Doesn't capture HOW MUCH better

**Average @3 Across Hops:**
- ❌ Confounds initial quality with improvement
- ❌ Good hop 1 + no improvement looks good
- ❌ Doesn't isolate multi-hop value

**Peak Hop Position:**
- ❌ Measures efficiency, not capability
- ❌ Hop 5 peak could be thorough OR inefficient
- ❌ Doesn't measure improvement magnitude

**Marginal Value Per Hop:**
- ❌ Complex to explain and interpret
- ❌ Multiple comparisons needed
- ❌ No single p-value

#### Implementation Notes

**Data Requirements:**
- All hop-level @3 scores for each utterance (both control and treatment)
- Ability to identify hop 1 vs later hops
- Paired utterances only (438 in Job 133560)

**Calculation Steps (Universal Recovery Score):**
1. For each paired utterance:
   - Control: Find best @3 across all hops, subtract hop 1 @3
   - Treatment: Find best @3 across all hops, subtract hop 1 @3
2. Calculate differences (treatment - control)
3. Run paired t-test on differences
4. Calculate Cohen's d effect size
5. Generate win/loss/tie counts
6. Create visualization (scatter plot, histogram of differences)

**Subgroup Analysis:**
1. Calculate median Hop 1 @3 from combined pool
2. Filter to utterances where BOTH control AND treatment < median
3. Repeat analysis on filtered set
4. Report both overall and subgroup results

**Edge Cases:**
- **Single-hop utterances**: Recovery Score = 0 (best = hop 1, no additional hops)
- **Empty hops after hop 1**: Recovery Score = 0 if no results found
- **Worse later hops**: If best hop is still hop 1, Recovery Score = 0
- **All valid**: These are correct interpretations (no multi-hop benefit)

---

## Across-Hop Strategy Metrics

### Overview

Beyond the primary quality/efficiency/recovery metrics, additional metrics characterize HOW models explore and refine across hops.

**Key dimensions:**
- **Adaptability**: Can model improve over time?
- **Efficiency**: Does model find good results quickly?
- **Resilience**: Can model recover from failures?
- **Coverage**: Does model explore comprehensively?
- **Strategy Type**: What pattern does model follow?

### Recommended Across-Hop Metrics

#### 1. Improvement Rate

**Definition:** % utterances where best hop score > hop 1 score

**Measures:** Adaptability - can model improve through iteration?

**Comparison:**
- Treatment: 65% of utterances improved
- Control: 52% of utterances improved
- Interpretation: Treatment more adaptive

**Complements Recovery Score:** This is binary (improved Y/N), Recovery Score is magnitude (how much)

---

#### 2. Average Peak Hop Position

**Definition:** Mean/median hop number where best score achieved

**Measures:** Efficiency - how quickly does model find best results?

**Comparison:**
- Treatment median: 2 (best results typically in hop 2)
- Control median: 1 (best results typically in hop 1)
- Interpretation: Control more front-loaded, treatment explores more

**Lower is more efficient** (finds good info early)

---

#### 3. Marginal Value Curve

**Definition:** Score improvement per additional hop (cumulative contribution)

**Measures:** Utilization - does each hop add value?

**Analysis:**
- Hop 1 contribution to final quality
- Hop 2 additional contribution beyond hop 1
- Hop 3 additional contribution beyond hop 1-2
- Plot should show diminishing returns if model stops appropriately

**Comparison:**
- Treatment: Hop 1 = 70% of final value, Hop 2 = 20%, Hop 3+ = 10%
- Control: Hop 1 = 85% of final value, Hop 2 = 12%, Hop 3+ = 3%
- Interpretation: Control front-loads value (efficient), treatment continues exploring

**Good strategy:** High early value, stops when marginal benefit drops

---

#### 4. Domain Coverage Diversity

**Definition:** Average unique domains (email/chat/file/calendar) searched per utterance

**Measures:** Strategy breadth - fanout vs focused search

**Comparison:**
- Treatment avg: 3.2 domains per utterance
- Control avg: 2.1 domains per utterance
- Interpretation: Treatment does broader fanout exploration

**Not inherently good or bad:** Depends on whether breadth helps

**Additional sub-metrics:**
- Domain switching pattern (exhaust one domain first vs interleave)
- Redundant domain queries (same domain repeatedly without new results)
- Coverage completeness (% utterances searching all relevant domains)

---

#### 5. Post-Empty Recovery Rate

**Definition:** After an empty hop (0 results), % of next hops that find good results (above threshold)

**Measures:** Resilience - can model recover from search failures?

**Comparison:**
- Treatment: 42% recovery after empty hop
- Control: 28% recovery after empty hop
- Interpretation: Treatment more resilient, adapts strategy after failure

**Related metrics:**
- Average hops attempted after first empty hop (persistence)
- Strategy change detection (does query differ after empty hop?)

---

#### 6. Hop Progression Pattern Classification

**Definition:** Classify each utterance's hop progression pattern

**Patterns:**
- **Improving (↗)**: Scores generally increase over hops (adaptive refinement)
- **Declining (↘)**: Scores decrease (query exhaustion, going off track)
- **Stable (→)**: Consistent scores across hops (redundant searching)
- **Erratic (↕)**: Alternating high/low scores (exploratory/unfocused)

**Comparison:**
- Treatment: 45% improving, 20% declining, 25% stable, 10% erratic
- Control: 30% improving, 35% declining, 30% stable, 5% erratic
- Interpretation: Treatment more adaptive (more improving), control more prone to decline

---

#### 7. Query Refinement Effectiveness

**Definition:** Score improvement between consecutive hops (for hops that improved)

**Measures:** Is model learning from previous results?

**Metrics:**
- Average improvement per hop (for improving hops only)
- Worsening rate (% hops where score decreased from previous)
- Consecutive improvement streaks (max improving hops in sequence)

**Comparison:**
- Treatment avg improvement: +0.42 per hop
- Control avg improvement: +0.28 per hop
- Treatment worsening rate: 18%
- Control worsening rate: 25%
- Interpretation: Treatment better at query refinement

---

#### 8. Early vs Late Quality Comparison

**Definition:** Compare performance in different search phases

**Phases:**
- Hop 1 @3 (initial strategy)
- Hop 2-3 @3 (refinement phase)
- Hop 4+ @3 (deep search phase)
- Best hop @3 (peak quality anywhere)

**Comparison:**
- Treatment: Hop 1=2.41, Hop 2-3=2.68, Hop 4+=2.52, Best=2.89
- Control: Hop 1=2.38, Hop 2-3=2.51, Hop 4+=2.19, Best=2.67
- Interpretation: Treatment benefits from exploration, maintains quality in deep search; Control peaks early then declines

**Reveals strategy types:**
- **Front-loaded**: High hop 1, diminishing returns later
- **Exploratory**: Moderate hop 1, improves significantly later
- **Deep-search**: Maintains quality even in hop 4+

---

### Strategy Pattern Classification Framework

Based on above metrics, classify overall reasoning patterns:

#### Efficient Targeted
- Hop 1 peak quality
- Low improvement rate
- Small domain coverage
- Low hop count

**Strategy:** Formulate precise queries upfront, minimal iteration  
**Pros:** Fast, low cost, efficient  
**Cons:** May miss information from unexplored domains

#### Adaptive Refinement
- Moderate hop 1 quality
- High improvement rate
- Moderate domain coverage
- Medium hop count

**Strategy:** Start reasonable, refine based on results  
**Pros:** Balanced quality and efficiency  
**Cons:** Requires multiple rounds, moderate cost

#### Broad Exploration
- Lower hop 1 quality
- High improvement rate
- High domain coverage
- Higher hop count

**Strategy:** Cast wide net, find gems through exploration  
**Pros:** Thorough, high peak quality, good recovery  
**Cons:** Expensive, many low-value results, high latency

#### Inefficient Recovery
- Low hop 1 quality
- High peak hop position (late)
- High hop count
- Low improvement rate

**Strategy:** Poor initial queries, eventually stumbles on good results  
**Pros:** Eventually finds information (sometimes)  
**Cons:** Very inefficient, luck-based, high cost

---

## Open Questions & Next Steps

### Questions to Resolve Before Implementation

1. **Primary metric finalization:**
   - Confirm: Use both Combined @3 AND Average All as dual primary metrics?
   - Or: One primary, one secondary?

2. **Recovery Score approach:**
   - Confirm: Universal Recovery Score (all utterances) as primary?
   - With subgroup analysis (both models poor hop 1) as supplementary?

3. **Threshold values:**
   - What threshold defines "poor" hop 1 for subgroup analysis?
   - Options: Median, quartile, fixed value (e.g., < 2.0)

4. **Statistical significance level:**
   - Standard p < 0.05?
   - Bonferroni correction if testing multiple metrics?

5. **Across-hop metrics priority:**
   - Which 3-5 across-hop metrics are most valuable?
   - Focus on: Improvement rate, peak hop position, domain diversity, post-empty recovery?

6. **Visualization priorities:**
   - Which plots are most important for decision-making?
   - Scatter plots (control vs treatment)?
   - Histograms (distribution of differences)?
   - Trajectory plots (hop progression)?

### Next Steps for Implementation

**Phase 1: Core Metrics Calculation**
1. Implement Combined @3 calculation (all hops combined)
2. Implement Average All Results calculation
3. Implement Universal Recovery Score
4. Calculate for all 438 paired utterances

**Phase 2: Statistical Analysis**
1. Paired t-tests for all three core metrics
2. Effect sizes (Cohen's d)
3. Win/loss/tie counts
4. 95% confidence intervals

**Phase 3: Subgroup Analysis**
1. Identify poor hop 1 cases (both models)
2. Repeat Recovery Score analysis on subgroup
3. Compare overall vs subgroup results

**Phase 4: Across-Hop Metrics**
1. Calculate top 5 across-hop metrics
2. Characterize strategy patterns
3. Generate visualizations

**Phase 5: Reporting**
1. Generate comprehensive comparison report
2. Create decision framework
3. Provide recommendations

### Files to Update

**Primary proposal document:**
- `proposals/reasoning_quality_metrics.md` - Core metrics framework already drafted
- Need to add: Recovery Score details, evaluation methodology

**Implementation:**
- `projects/seval/seval_batch_processor.py` - Add metric calculations
- `projects/seval/merge_seval_results.py` - May need data structure updates

**Visualization:**
- New plots for Recovery Score, gap analysis, trajectory plots

### Key Decisions Pending

**For next session:**
1. Review and finalize metric definitions
2. Decide on Universal vs Combined Threshold for Recovery Score
3. Select priority across-hop metrics (top 5)
4. Confirm statistical approach
5. Begin implementation plan

---

## Summary of Key Insights from Discussion

### Major Breakthroughs

1. **Dual-metric framework prevents gaming:**
   - Combined @3 alone would encourage exhaustive search
   - Average All alone would discourage exploration
   - Together they balance quality vs efficiency

2. **Recovery Score pairing issue solved:**
   - Original threshold approach breaks pairing
   - Universal Recovery Score maintains pairing
   - Subgroup analysis provides focused insights

3. **Gap analysis reveals strategy without judgment:**
   - Large gap = capable but noisy
   - Small gap = efficient and targeted
   - Neither inherently better, depends on priorities

4. **Multi-hop value clarified:**
   - Not just "more searching"
   - Adaptive improvement from failed initial attempts
   - Recovery Score captures this value proposition

### Concerns Addressed

1. **Average @3 across hops penalizes multi-hop:**
   - Solved by Combined @3 (pool all results)

2. **Empty hop handling:**
   - Combined @3: Empty hops contribute 0 results (neutral)
   - Average All: Empty hops contribute 0 results (neutral)
   - Recovery Score: Empty hops → score = 0 (failed recovery)
   - All handle naturally without special cases

3. **Fair comparison across strategies:**
   - Combined @3 doesn't bias against exploration
   - Average All rewards precision
   - Recovery Score isolates multi-hop value
   - Across-hop metrics characterize without prescribing

---

**Resume Discussion From Here:**
- Review this document
- Finalize open questions
- Proceed to implementation

**Related Files:**
- `proposals/reasoning_quality_metrics.md` - Formal proposal (already drafted)
- `temp/mem.md` - Working notes with all ideas
- `projects/seval/` - Implementation location

