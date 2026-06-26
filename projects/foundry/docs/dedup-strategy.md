# Utterance Deduplication Strategy for SEVAL Querysets

**Created:** 2026-04-08  
**Updated:** 2026-04-10

## Problem

SEVAL does not allow duplicate utterances in a dataset. Some session files contain identical utterances (Type 1: exact input duplicates) or the same utterance with different input fields (Type 2: same utterance, different input). To generate a valid queryset, each utterance string must be unique.

---

## Final Strategy: Punctuation-Only Dedup (Cartesian Product)

Based on validation results (see [Validation Results](#validation-results) below), **only punctuation marks** should be used for dedup. The `(dupX)` approach was rejected because it pollutes 44% of responses with dup-related commentary.

### Core Principle

1. **Extract** the **base text** from the original utterance by iteratively stripping whitespace, paired marks, and trailing punctuation (see [Base Text Extraction](#base-text-extraction) below).
2. **Determine** the set of 4 compatible suffix marks, choosing script-appropriate marks (Latin or CJK) based on the base text.
3. **Enumerate** suffixes as the Cartesian product of compatible marks, ordered by length: empty (length 0) → single marks (length 1) → 2-mark combinations (length 2) → …
4. **Assign** each copy the next suffix in order.

### Base Text Extraction

The base text is obtained by iteratively applying three operations until the text stabilizes (no more changes):

1. **Strip whitespace** from both ends.
2. **URL guard** — if the text ends within a URL (the last `https://` or `http://` has no space between it and the end of the string), **stop immediately**. Characters like `?`, `.` are valid URL components and must not be stripped.
3. **Strip paired marks** — if the text starts with an opener AND ends with the matching closer, remove both. Only one pair per iteration.
4. **Strip a single trailing mark** — remove exactly one trailing punctuation character (Latin or CJK). Only one mark per iteration to avoid orphaning an opener on the left.

Then repeat from step 1. The loop terminates when an iteration produces no change.

**Why one trailing mark at a time?** Consider `¡hola!.`:
- Iter 1: strip `.` → `¡hola!`
- Iter 2: pair `¡!` matched → `hola`
- Iter 3: stable → base = `hola`

If we stripped all trailing marks at once, `!.` would both be removed in one step, orphaning `¡`.

#### Paired Marks

| Opener | Closer | Usage |
|--------|--------|-------|
| `¿` | `?` | Spanish questions |
| `¡` | `!` | Spanish exclamations |
| `"` | `"` | ASCII double quotes |
| `'` | `'` | ASCII single quotes |
| `\u201c` (`"`) | `\u201d` (`"`) | Smart double quotes |
| `\u2018` (`'`) | `\u2019` (`'`) | Smart single quotes |
| `«` | `»` | Guillemets (French, Russian, etc.) |
| `「` | `」` | CJK single corner brackets |
| `『` | `』` | CJK double corner brackets |
| `（` | `）` | CJK fullwidth parentheses |
| `【` | `】` | CJK lenticular brackets |
| `《` | `》` | CJK double angle brackets |
| `〈` | `〉` | CJK single angle brackets |
| `〔` | `〕` | CJK tortoise shell brackets |

Pair stripping requires BOTH the opener at the start AND the closer at the end. Unpaired marks are left alone.

#### Trailing Marks

Both Latin and CJK punctuation marks are recognized:

| Latin | CJK equivalent |
|-------|---------------|
| `.` | `。` |
| `,` | `、` |
| `;` | `；` |
| `?` | `？` |
| `!` | `！` |

Full set: `. , ; ? ! 。 ？ ！ 、 ；`

#### Trace Examples

| Original | → Base | Steps |
|----------|--------|-------|
| `hello.` | `hello` | strip `.` |
| `hello ?` | `hello` | strip ws → `hello ?`, strip `?` → `hello `, strip ws → `hello` |
| `¿mañana?` | `mañana` | strip `?` → `¿mañana`, no pair, strip — no mark, stable. Wait — `¿` at start, no closer. Actually: strip `?` → `¿mañana` — hmm, `¿` is an opener but no closer remains. Stable → base = `¿mañana`. No: re-trace. `¿mañana?`: ws ok, pair `¿`…`?` matched → `mañana`, stable. |
| `¡hola!.` | `hola` | strip `.` → `¡hola!`, pair `¡!` → `hola` |
| `"hello?"` | `hello` | pair `""`→ `hello?`, strip `?` → `hello` |
| `...ください。` | `...ください` | strip `。` → `...ください` |
| `...ますか？` | `...ますか` | strip `？` → `...ますか` |
| `「こんにちは」。` | `こんにちは` | strip `。` → `「こんにちは」`, pair `「」` → `こんにちは` |
| `《 hello 》 .` | `hello` | strip ws → `《 hello 》 .`, strip `.` → `《 hello 》`, strip ws → `《 hello 》`, pair `《》` → ` hello `, strip ws → `hello` |
| `...AllItems.aspx?` | `...AllItems.aspx?` | strip ws → ok, URL guard: `https://...aspx?` ends with URL → **stop** |
| `text https://a.com/b?x=1` | `text https://a.com/b?x=1` | strip ws → ok, URL guard: last `https://` at pos 5, no space after → **stop** |
| `오늘의 회의` | `오늘의 회의` | no marks to strip → stable |

### Script-Aware Suffix Marks

The suffix marks are chosen based on the **script** of the base text's last content character. "Content character" means the last character that is NOT a quote, paired mark, punctuation, or part of a trailing URL — i.e., the last character of actual word content. This is found by:

1. **Strip trailing URL** — if the base text ends with a URL (`https://...` or `http://...` running to the end with no spaces), ignore it for script detection. This ensures `中文内容 https://example.com` detects as CJK.
2. **Scan backward** past quote marks, paired mark characters, and trailing marks to find the actual last content character.

This matters for mixed-script utterances like `カレンダー上の会議でカテゴリが 'deepseek'` — the base text ends with `'` (a quote), but the last content word is `deepseek` (Latin), so Latin marks are used. Conversely, `今年作成したファイルについて` ends with `て` (hiragana), so CJK marks are used.

**Korean exception:** Modern Korean uses Latin punctuation marks, not CJK ideographic marks. Although Hangul falls within CJK Unicode ranges, Korean text is assigned Latin marks (`.`, `,`, `;`, `?`/`!`). The script detection distinguishes Hangul from Chinese/Japanese characters specifically for mark selection.

After finding the last content character:
- **Chinese/Japanese** (CJK ideographs, hiragana, katakana) → CJK marks
- **Korean** (Hangul) → Latin marks
- **Everything else** (Latin, Cyrillic, Arabic, etc.) → Latin marks

| Script | Period | Comma | Semicolon | Question/Exclamation |
|--------|--------|-------|-----------|---------------------|
| Chinese/Japanese | `。` | `、` | `；` | `？` / `！` |
| Korean | `.` | `,` | `;` | `?` / `!` |
| Latin / other | `.` | `,` | `;` | `?` / `!` |

4 marks per utterance (3 neutral + 1 type-specific).

### Compatible Mark Set

| Mark | Condition |
|---|---|
| `.` / `。` | Always |
| `,` / `、` | Always |
| `;` / `；` | Always |
| `?` / `？` | Utterance is a question |
| `!` / `！` | Utterance is NOT a question |

(CJK variants apply only to Chinese/Japanese; Korean uses the Latin variants.)

### Suffix Enumeration Order

Suffixes are generated as the **Cartesian product** of the 4 compatible marks, ordered by:
1. **Length** — shorter suffixes first (0, 1, 2, 3, …)
2. **Lexicographic order** within the same length

**For questions** (Latin marks: `.`, `,`, `;`, `?`):

| # | Suffix | Len |
|---:|---|---:|
| 1 | *(empty)* | 0 |
| 2 | `.` | 1 |
| 3 | `,` | 1 |
| 4 | `;` | 1 |
| 5 | `?` | 1 |
| 6 | `..` | 2 |
| 7 | `.,` | 2 |
| … | … | … |
| 21 | `??` | 2 |
| 22–85 | *(length 3: 64 combos)* | 3 |

**For CJK questions** (marks: `。`, `、`, `；`, `？`): same structure with CJK marks.

### Capacity

| Max suffix length | Suffixes available | Formula |
|---:|---:|---|
| 0 | 1 | 4⁰ |
| 1 | 5 | 1 + 4¹ |
| 2 | 21 | 1 + 4 + 4² |
| 3 | 85 | 1 + 4 + 4² + 4³ |
| N | (4^(N+1) − 1) / 3 | geometric sum |

With 4 marks, capacity grows exponentially. Any practical duplication count is covered at very short suffix lengths.

### Grouping by Base Text

After extracting the base text, multiple originally-different utterances may share the same base. For example:

- `"hello"`, `"hello."`, `"hello?"` → all become base `"hello"`
- `"¿mañana?"`, `"¿mañana?"` → both become base `"mañana"`
- `"...ください。"`, `"...ください"` → both become base `"...ください"`

All copies sharing the same base must draw from the **same** suffix pool to avoid collisions. The question/non-question classification is determined from the **original text** (before stripping), not from the base text.

### Question Detection

An utterance is classified as a question if:
- It ends with `?` or `？` (after stripping whitespace, before stripping marks), **or**
- It starts with `¿` (after stripping whitespace), **or**
- Its first word (case-insensitive) is one of: `what`, `when`, `where`, `who`, `whom`, `which`, `whose`, `how`, `why`, `did`, `do`, `does`, `is`, `are`, `was`, `were`, `can`, `could`, `will`, `would`, `should`, `shall`, `have`, `has`, `had`

### Coverage of Actual Dataset

| Copies | Utterances | Max suffix length needed |
|---:|---:|---:|
| 2 | ~230 | 1 |
| 3 | ~93 | 1 |
| 4 | ~4 | 1 |
| 5 | ~378 | 1 |
| 6 | ~3 | 2 |
| 9–10 | ~55 | 2 |
| 37 | 1 | 3 |

All cases in the dataset are covered with suffix length ≤ 3.

### Examples

**2 copies (English)** — Original: `"Did I miss any meeting today"`

Base: `Did I miss any meeting today` (question — starts with "Did")

| # | Suffix | Utterance |
|---:|---|---|
| 1 | *(empty)* | `Did I miss any meeting today` |
| 2 | `.` | `Did I miss any meeting today.` |

**5 copies (English)** — Original: `"Show meetings I joined last week."`

Base: `Show meetings I joined last week` (non-question)

| # | Suffix | Utterance |
|---:|---|---|
| 1 | *(empty)* | `Show meetings I joined last week` |
| 2 | `.` | `Show meetings I joined last week.` |
| 3 | `,` | `Show meetings I joined last week,` |
| 4 | `;` | `Show meetings I joined last week;` |
| 5 | `!` | `Show meetings I joined last week!` |

**2 copies (Spanish)** — Original: `"¿Tengo una reunión con mi gerente mañana?"`

Base: `Tengo una reunión con mi gerente mañana` (question — original has `¿`)

| # | Suffix | Utterance |
|---:|---|---|
| 1 | *(empty)* | `Tengo una reunión con mi gerente mañana` |
| 2 | `.` | `Tengo una reunión con mi gerente mañana.` |

**2 copies (Japanese)** — Original: `"ファイルを探してください。"`

Base: `ファイルを探してください` (non-question, CJK)

| # | Suffix | Utterance |
|---:|---|---|
| 1 | *(empty)* | `ファイルを探してください` |
| 2 | `。` | `ファイルを探してください。` |

**2 copies (Japanese question)** — Original: `"教えていただけますか？"`

Base: `教えていただけますか` (question — original ends with `？`, CJK)

| # | Suffix | Utterance |
|---:|---|---|
| 1 | *(empty)* | `教えていただけますか` |
| 2 | `。` | `教えていただけますか。` |

**10 copies (English)** — Original: `"What information do you have about me?"`

Base: `What information do you have about me` (question)

| # | Suffix | Utterance |
|---:|---|---|
| 1 | *(empty)* | `What information do you have about me` |
| 2 | `.` | `What information do you have about me.` |
| 3 | `,` | `What information do you have about me,` |
| 4 | `;` | `What information do you have about me;` |
| 5 | `?` | `What information do you have about me?` |
| 6 | `..` | `What information do you have about me..` |
| 7 | `.,` | `What information do you have about me.,` |
| 8 | `.;` | `What information do you have about me.;` |
| 9 | `.?` | `What information do you have about me.?` |
| 10 | `,.` | `What information do you have about me,.` |

**37 copies** — needs suffixes up to length 3:

Uses all 21 length-≤2 suffixes, then 16 of the 64 length-3 combinations (e.g., `...`, `..,`, `..;`, `..?`, `.,.`, `.,,`, …).

---

## Explored Strategies (Historical)

The following sections document strategies that were explored and tested before arriving at the final punctuation-only approach above.

### Strategy A: Two-Tier Suffix Approach (Superseded)

#### Tier 1 — Trailing Punctuation (up to 7 extra copies)

Append semantics-neutral trailing punctuation to extra copies. The first copy is kept as-is. This produces natural-looking variations similar to how humans type in practice.

| Copy | Suffix | Condition |
|---:|---|---|
| 1 | *(original, unchanged)* | — |
| 2 | `.` | Always safe |
| 3 | `..` | Always safe |
| 4 | `...` | Always safe |
| 5 | `,` | Always safe |
| 6 | `;` | Always safe |
| 7 | `?` | Only if the utterance **is** a question |
| 7 | `!` | Only if the utterance is **not** a question |

**Total capacity:** 7 variants + original = **8 copies** before fallback.

#### Tier 2 — `(dupX)` + Punctuation Combinations (8+ copies)

When Tier 1 punctuation variants are exhausted, use `(dupN)` as a base and combine it with the same punctuation suffixes to generate up to **7 variations per base**:

| Copy | Suffix |
|---:|---|
| 8 | `(dup1)` |
| 9 | `(dup1).` |
| 10 | `(dup1)..` |
| 11 | `(dup1)...` |
| 12 | `(dup1),` |
| 13 | `(dup1);` |
| 14 | `(dup1)?` or `(dup1)!` |
| 15 | `(dup2)` |
| 16 | `(dup2).` |
| ... | ... |

**Capacity:** Tier 1 provides 8 copies. Each `(dupN)` base provides 7 more. Total capacity = **8 + 7N** where N is the number of `(dupX)` bases used.

| Copies needed | Tier 1 | Tier 2 bases needed | Max `(dupX)` |
|---:|---:|---:|---:|
| ≤ 8 | 8 | 0 | — |
| ≤ 15 | 8 | 1 | `(dup1)` |
| ≤ 22 | 8 | 2 | `(dup2)` |
| ≤ 29 | 8 | 3 | `(dup3)` |
| ≤ 36 | 8 | 4 | `(dup4)` |
| ≤ 43 | 8 | 5 | `(dup5)` |

#### Rules

##### Collision avoidance

If the original utterance already ends with one of the punctuation suffixes (e.g., original ends with `.`), **skip that variant** to avoid producing an identical string. Move to the next available suffix.

##### Question detection

An utterance is classified as a question if:
- It ends with `?` (after stripping whitespace), **or**
- It starts with a question word: `what`, `when`, `where`, `who`, `whom`, `which`, `whose`, `how`, `why`, `did`, `do`, `does`, `is`, `are`, `was`, `were`, `can`, `could`, `will`, `would`, `should`, `shall`, `have`, `has`, `had`

##### Suffix selection for `?` / `!`

| Utterance type | Suffix used |
|---|---|
| Question | `?` — reinforces the question, no meaning change |
| Non-question (statement/command) | `!` — natural for imperative/declarative, no meaning change |

##### Why certain punctuation is excluded

| Mark | Reason for exclusion |
|---|---|
| `?` on non-questions | Changes statement to question — alters meaning |
| `!` on questions | Changes tone from inquiry to exclamation — alters meaning |
| `[N]` | Sydney uses `[1]`, `[2]` as citation markers — LLM may interpret as citation reference |

#### Examples

##### Type 1 — 2 copies (common case)

Original: `Did I miss any meeting today`

| Copy | Utterance |
|---:|---|
| 1 | `Did I miss any meeting today` |
| 2 | `Did I miss any meeting today.` |

##### Type 1 — 5 copies

Original: `Show meetings I joined last week`

| Copy | Utterance |
|---:|---|
| 1 | `Show meetings I joined last week` |
| 2 | `Show meetings I joined last week.` |
| 3 | `Show meetings I joined last week..` |
| 4 | `Show meetings I joined last week...` |
| 5 | `Show meetings I joined last week,` |

##### Type 1 — 10 copies (uses Tier 2)

Original: `What information do you have about me?`

| Copy | Utterance | Tier |
|---:|---|---|
| 1 | `What information do you have about me?` | original |
| 2 | `What information do you have about me?.` | Tier 1 |
| 3 | `What information do you have about me?..` | Tier 1 |
| 4 | `What information do you have about me?...` | Tier 1 |
| 5 | `What information do you have about me?,` | Tier 1 |
| 6 | `What information do you have about me?;` | Tier 1 |
| 7 | *(skipped — original already ends with `?`)* | — |
| 7 | `What information do you have about me? (dup1)` | Tier 2 |
| 8 | `What information do you have about me? (dup1).` | Tier 2 |
| 9 | `What information do you have about me? (dup1)..` | Tier 2 |
| 10 | `What information do you have about me? (dup1)...` | Tier 2 |

##### Type 2 — Same utterance, different input

Same dedup logic applies. The utterance string is made unique across all copies regardless of whether the input fields differ.

#### Applicability

This strategy applies to **both** Type 1 and Type 2 duplicates uniformly. All copies of the same utterance (whether exact input dups or intentional variants) are suffixed so that every utterance string in the final queryset is unique.

### Validation Results

**Test setup:** Generated 100 copies of a single utterance (`"What is Katherine Gu title?"`) using both tiers, submitted to SEVAL for scraping, and analyzed the results across three dimensions: tool calls (reasoning), final response (answer quality), and marker leakage.

**Source files:**
- Queryset: `local/Results/dedup_test_100.tsv`
- Results: `C:\Users\binzhu\Downloads\control_conversations.tsv`

#### Tier 1 — Trailing Punctuation (copies 1–7): Perfect

| Metric | Result |
|---|---|
| Correct answer (title found) | **7/7** (100%) |
| Tool call type | `office365_search` with `domain: people` — all 7 |
| Search query leakage | **0** — no punctuation artifacts in search query |
| Dup marker in response | **0** — model never mentioned trailing punctuation |
| Confusion signals | **0** — no "did you mean" or disclaimers |

**Search queries generated (Tier 1):**

| Query | Count |
|---|---:|
| `Katherine Gu title` | 4 |
| `Katherine Gu` | 3 |

Both queries return the correct people result. The variation is normal LLM behavior (same with or without dedup suffixes).

**Conclusion:** Trailing punctuation is **completely invisible** to the model. It does not affect tool calls, reasoning, or response quality. This tier is safe for evaluation.

#### Tier 2 — `(dupX)` Markers (copies 8–100): Functional but noisy

| Metric | Result |
|---|---|
| Correct answer (title found) | **92/93** (99%) |
| Tool call type | `office365_search` with `domain: people` — all 93 |
| Search query leakage | **0** — `(dup)` marker never leaked into search query |
| Dup marker echoed in response | **41/93** (44%) |
| "Confused" signals in response | **29/93** (31%) |
| ToolChainAnalysis outcome | `Proceed` — all 93 |

**Search queries generated (Tier 2):**

| Query | Count |
|---|---:|
| `Katherine Gu` | 53 |
| `Katherine Gu title` | 37 |
| `Katherine Gu job title` | 1 |
| `Katherine Gu profile` | 1 |
| `Katherine Gu's job title` | 1 |

All search queries are clean — the model correctly strips `(dupX)` before formulating the search.

**Response issues observed:**

1. **Echoing the marker (44%):** The model mentions `(dup)` in its response, e.g.:
   > "If by 'dup1' you meant a different Katherine..."
   > "Note on 'dup1': I only see one active profile..."

2. **Confusion disclaimers (31%):** The model adds unnecessary caveats:
   > "If you're referring to a duplicate entry, alias, or..."
   > "If you meant a *different* Katherine (e.g., another org)..."

3. **Missing title (1%):** One response omitted the explicit title string (though it still discussed Katherine Gu).

**Conclusion:** `(dupX)` does **not** affect **tool calls or the core answer**, but it **pollutes the response text** with dup-related commentary in ~44% of cases. This could impact text-based evaluation metrics (e.g., response quality grading, conciseness checks) even though the factual answer is correct.

#### Overall Assessment

| Aspect | Tier 1 (punctuation) | Tier 2 (`(dupX)`) |
|---|---|---|
| Correct answer | 100% | 99% |
| Tool call accuracy | Perfect | Perfect |
| Search query leakage | None | None |
| Response pollution | None | 44% echo markers |
| Eval metric impact | **None** | **Moderate** — may affect quality scores |
| Recommended for | All cases ≤ 7 extra copies | Cases > 7 copies, with caveat |

#### Recommendations

1. **Maximize Tier 1 usage.** Most duplicates are 2–5 copies (per the duplication distribution), well within Tier 1's 7-copy capacity.

2. **Accept Tier 2 for outliers.** The 10-copy and 37-copy cases require Tier 2. The response pollution is cosmetic — the model still gets the right answer and makes the right tool calls.

3. **For eval metrics sensitive to response text quality**, consider whether Tier 2 responses should be filtered or whether the `(dup)` commentary should be excluded from scoring. Alternatively, for sessions with >7 copies, consider dropping excess copies rather than using `(dupX)` if response quality metrics are critical.
