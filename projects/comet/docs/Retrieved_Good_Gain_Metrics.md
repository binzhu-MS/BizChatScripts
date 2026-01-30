# Retrieved Good Gain Metrics — Implementation Guideline

This document defines the notation and formulas for the core **Retrieved Good Gain** metrics.

**Key Principles:**
- **Good Gain**: Only results with relevance gain ≥ 2 contribute to gain calculations.
- **Deduplication**: Good gain is counted only once per unique result (duplicates in later iterations do not add to gain).
- **Result Counts**: Total result counts ($R^i$, $R@i$) **include duplicates** to accurately measure search efficiency and redundancy.

---

## Preliminaries

### Scope

**Multi-Turn Conversations:** For multi-turn conversations, metrics are computed using **only the last turn**. Earlier turns are excluded from the calculation.

### Basic Definitions

- `i`: iteration index (1 ≤ i ≤ N). **Multiple search invocations may occur within a single iteration**.
- `N`: total number of iterations where **search was invoked**.
- `gain(c)`: relevance label ∈ {0, 1, 2, 3, 4} assigned by the LLM NDCG labeller.
- **Good Gain**: A result `c` has "good gain" if `gain(c) ≥ 2`.

### Deduplication Strategy

Deduplication ensures each unique result's gain is counted only once. We use a **two-phase approach**:

1. **Phase 1: Candidate Generation** — Fast O(1) hash lookup using multiple keys (domain-specific ID, generic ID, normalized URL, content hash)
2. **Phase 2: Verification** — Multi-step verification to confirm candidates are truly the same document

**Key Principles:**
- Domain-specific IDs and generic IDs are mutually exclusive
- Content-based hashing (snippet, title) captures duplicates with different IDs
- Any field mismatch in verification → reject; matching fields accumulate confidence

> **📄 Full Details:** See [Deduplication_Design.md](Deduplication_Design.md) for complete documentation including:


### Notation Conventions

| Notation | Meaning | Example |
|----------|---------|---------|
| $X^i$ | Set at iteration $i$ | $R^i$, $GR^i$ |
| $\|X^i\|$ | Cardinality (count) of set | $\|R^i\|$ = number of results |
| $X_i$ | Scalar value at iteration $i$ | $G_i$ = gain at iteration $i$ |
| $X@i$ | Cumulative value through iteration $i$ | $CG@i$, $R@i$ |

### Per-Iteration Sets

| Set | Description |
|-----|-------------|
| $R^i$ | **All** results at iteration $i$ (**including duplicates**, both within-iteration and cross-iteration) |
| $UR^i$ | **Unique new** results at iteration $i$ (first occurrence only; excludes duplicates within iteration $i$ **and** results seen in iterations 1 to $i-1$) |
| $GR^i$ | **Good unique new** results at iteration $i$ (gain ≥ 2, first occurrence seen from iterations 1 to i only) |
| $Dup^i$ | **Duplicate** results at iteration $i$ (results already seen earlier in iteration $i$ **or** in iterations 1 to $i-1$) |

**Relationship:** $|R^i| = |UR^i| + |Dup^i|$

> **Note on Within-Iteration Duplicates:** Since multiple search queries can be issued within a single iteration, the same result may appear multiple times within iteration $i$. Only the **first occurrence** contributes to $UR^i$ and $GR^i$; subsequent occurrences (within the same iteration or later iterations) are counted in $Dup^i$.

### Per-Iteration Gain

$$
G_i = \sum_{c \in GR^i} gain(c)
$$

$G_i$ sums gains **only** from good unique results at iteration $i$. Duplicates and results with gain < 2 contribute zero.

### Cumulative Counts

| Cumulative | Formula | Description |
|------------|---------|-------------|
| $R@i$ | $\sum_{k=1}^{i} \|R^k\|$ | Total results through iteration $i$ (**including duplicates**) |
| $UR@i$ | $\sum_{k=1}^{i} \|UR^k\|$ | Total **unique** results through iteration $i$ |
| $GR@i$ | $\sum_{k=1}^{i} \|GR^k\|$ | Total **good unique** results through iteration $i$ |
| $DupR@i$ | $\sum_{k=1}^{i} \|Dup^k\|$ | Total **duplicate** results through iteration $i$ |

**Relationship:** $R@i = UR@i + DupR@i$

### Discount Weight

$$
w(i) = \frac{1}{\log_2(i+1)}
$$

This gives higher weight to earlier iterations: $w(1) = 1.0$, $w(2) ≈ 0.63$, $w(3) = 0.5$, etc.

---

# 1. CG@i — Cumulative Good Gain

**Definition:** Total sum of good gains from iteration 1 through iteration $i$.

$$
CG@i = \sum_{k=1}^{i} G_k
$$

**Where:** $G_k$ = sum of gains for good unique results at iteration $k$ (i.e., $\sum_{c \in GR^k} gain(c)$)

**Interpretation:** Measures the total "quality" of search results accumulated over iterations. Higher is better.

---

# 2. RG@i — Rate of Good Gain

**Definition:** Average good gain per iteration.

$$
RG@i = \frac{CG@i}{i}
$$

**Where:** $CG@i$ = cumulative good gain through iteration $i$; $i$ = number of iterations

**Interpretation:** Measures average gain yield per iteration. Useful for comparing searches with different iteration counts.

---

# 3. DCG@i — Discounted Cumulative Good Gain

**Definition:** Cumulative gain with logarithmic discount — earlier iterations contribute more.

$$
DCG@i = \sum_{k=1}^{i} w(k) \cdot G_k
$$

**Where:** $G_k$ = gain at iteration $k$; $w(k) = \frac{1}{\log_2(k+1)}$

**Interpretation:** Rewards finding good results early. A system that retrieves good results in iteration 1 scores higher than one that finds them in iteration 5.

---

# 4. DRG@i — Discounted Rate of Good Gain

**Definition:** Average discounted gain per iteration.

$$
DRG@i = \frac{DCG@i}{i}
$$

**Where:** $DCG@i$ = discounted cumulative good gain; $i$ = number of iterations

**Interpretation:** Normalized version of DCG@i for comparing across different iteration counts.

---

# 5. AvgGain_i — Average Good Gain per Result at Iteration i

**Definition:** The average good gain per result **at iteration $i$**, computed as gain at iteration $i$ divided by total results at iteration $i$ (including duplicates).

$$
AvgGain_i =
\begin{cases}
\dfrac{G_i}{|R^i|}, & |R^i| > 0 \\[8pt]
0, & |R^i| = 0
\end{cases}
$$

**Where:** $G_i$ = good gain at iteration $i$ (deduplicated); $|R^i|$ = total results at iteration $i$ (**including duplicates**)

**Interpretation:** Measures the "yield" of good gain per result at a specific iteration. The numerator excludes duplicates (each result's gain counted once), while the denominator includes all results (including duplicates). This penalizes iterations that return many duplicate results.

---

# 6. RAG@i — Rate of Average Gain

**Definition:** The mean of per-iteration average gains from iteration 1 through $i$.

$$
RAG@i = \frac{1}{i} \sum_{k=1}^{i} AvgGain_k
$$

**Where:** $AvgGain_k = G_k / |R^k|$ = average good gain per result at iteration $k$

**Interpretation:** Measures overall search efficiency across iterations. Higher values indicate consistently good results with low redundancy.

---

# 7. DRAG@i — Discounted Rate of Average Gain

**Definition:** Weighted mean of per-iteration average gains, with earlier iterations weighted more heavily.

$$
DRAG@i = \frac{1}{i} \sum_{k=1}^{i} w(k) \cdot AvgGain_k
$$

**Where:** $AvgGain_k = G_k / |R^k|$; $w(k) = \frac{1}{\log_2(k+1)}$

**Interpretation:** Similar to RAG@i but rewards good early-iteration efficiency more than late-iteration efficiency.

---

# 8. SRE@i — Search Result Efficiency

**Definition:** The fraction of results that are "good" (gain ≥ 2, deduplicated) relative to total results through iteration $i$.

$$
SRE@i =
\begin{cases}
\dfrac{GR@i}{R@i}, & R@i > 0 \\[8pt]
0, & R@i = 0
\end{cases}
$$

**Where:** $GR@i$ = total good unique results through iteration $i$; $R@i$ = total results through iteration $i$ (**including duplicates**)

**Interpretation:** What fraction of all returned results were useful? Higher is better. Values close to 1 indicate highly efficient search.

---

# 9. SRR@i — Search Result Redundancy

**Definition:** The fraction of all returned results that are duplicates through iteration $i$.

$$
SRR@i =
\begin{cases}
\dfrac{DupR@i}{R@i}, & R@i > 0 \\[8pt]
0, & R@i = 0
\end{cases}
$$

**Where:** $DupR@i$ = total duplicate results through iteration $i$; $R@i$ = total results (**including duplicates**)

**Interpretation:** What fraction of results were redundant? Lower is better. High SRR indicates the search system returns many previously-seen results.

**Note:** Since $R@i = UR@i + DupR@i$, equivalently: $SRR@i = \frac{DupR@i}{UR@i + DupR@i}$

---

# 10. IterationsForAllGoodResults

**Definition:** The minimum number of iterations needed to retrieve all good results (capped at 100).

$$
IterationsForAllGoodResults = \min(k, 100)
$$

**Where:** $k$ = smallest iteration index at which all good results have been retrieved

**Interpretation:** Measures how quickly the search system finds all relevant results. Lower is better.

---

# Summary Table

| Metric | Formula | Duplicates in Numerator? | Duplicates in Denominator? | Higher is Better? |
|--------|---------|--------------------------|----------------------------|-------------------|
| CG@i | $\sum G_k$ | No | N/A | Yes |
| RG@i | $CG@i / i$ | No | N/A | Yes |
| DCG@i | $\sum w(k) \cdot G_k$ | No | N/A | Yes |
| DRG@i | $DCG@i / i$ | No | N/A | Yes |
| AvgGain_i | $G_i / \|R^i\|$ | No | **Yes** | Yes |
| RAG@i | $\frac{1}{i}\sum AvgGain_k$ | No | **Yes** | Yes |
| DRAG@i | $\frac{1}{i}\sum w(k) \cdot AvgGain_k$ | No | **Yes** | Yes |
| SRE@i | $GR@i / R@i$ | No | **Yes** | Yes |
| SRR@i | $DupR@i / R@i$ | **Yes** | **Yes** | **No** |

---
