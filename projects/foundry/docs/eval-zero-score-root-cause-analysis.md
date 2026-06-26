# Foundry Evaluation: Zero/Low-Score Root Cause Analysis

**Date:** 2026-03-04  
**Dataset:** Full scraper dataset (8,529 queryset rows, 17,058 scraper files, 49 sessions)  
**Run:** `20260303-170809-results`

## Summary

All 0-score sessions were investigated through a 3-step process:

1. **Eval correctness** — Simulated pre-eval + eval on packed data → scores are correct given the packed data.
2. **Packing correctness** — Compared packed data against raw scraper files → packing is accurate, no data loss.
3. **Raw scraper data** — Checked whether 0 scores reflect genuine model behavior → confirmed, the model simply doesn't invoke the tools the criteria check for.

**Conclusion:** The packing script (`build_foundry_eval_sessions.py`) is correct. All 0% scores are genuine — they reflect either tool-name convention mismatches between eval criteria and DeepLeo telemetry, or test-data configuration issues (mock user mismatch).

---

## Zero-Score Sessions

| Session | Score | Root Cause |
|---------|-------|------------|
| `canvas_create` | 0% | Criteria expects `canmore_create_textdoc` tool invocation. Model invokes `python` (19/23 items), `office365_search` (3), `search_web` (1). The string `canmore_create_textdoc` only appears in prompt instruction metrics (`ExtensionRunner:ext:create-text-doc-instruction-v11-filtered`), never as an actual tool invocation in DeepLeoImprovedNetworking. |
| `canvas_create_overtriggering` | 0% | Same as `canvas_create` — but here 0% is actually **good** since the criteria says "lower is better" (checking for over-triggering). |
| `canvas_create_overtriggering_doc_CI` | 0% | Same pattern: `python` (15), `office365_search` (4), no `canmore_create_textdoc`. |
| `international_my_manager` | 0% | **Two issues:** (1) Criteria hardcodes `name = "Dmitrij Petters"` but scraper queries contain "Laura Park" — mock user profile mismatch. (2) ~50% of items use `search_office365` but criteria only checks `office365_search` / `office365.search`. |
| `meeting_intent_parameter` ("Has Meeting Tool") | 0% treatment arm | Criteria expects `search_enterprise_meetings`; DeepLeo telemetry records `office365_search` / `search_office365`. **Tool name convention mismatch.** |
| `triggering-email` | 0% | Criteria expects `search_enterprise_emails`; actual tools are `search_office365` / `office365_search`. Same convention mismatch. |
| `email_contains_key_info` (people query) | 0% | Filter condition produces 0 matching rows — no items to evaluate. |
| `query-correctness` | 0% | Criteria script is `None` / broken — no evaluation logic present. |

## Low-Score Sessions

| Session | Score | Notes |
|---------|-------|-------|
| `completeness_hints` (CHI - LB) | ~1–5% | Low but non-zero; different issue from tool name mismatch. |
| `people_eval_domain` | ~3.7% | Low but non-zero. |
| `personalization` (TruePositive) | ~1.1% | Low but non-zero. |

## Key Pattern: Tool Name Convention Mismatch

The dominant root cause across multiple 0-score sessions is a **tool name convention mismatch**:

- **Eval criteria expect** tool names like:
  - `search_enterprise_meetings`
  - `search_enterprise_emails`
  - `canmore_create_textdoc`

- **DeepLeo telemetry records** different names:
  - `office365_search`
  - `search_office365`
  - `python`

The pre-eval script correctly extracts `toolInvocations` from `DeepLeoImprovedNetworking` metrics with `fluxv3:invokingfunction`, but the tool names in the telemetry don't match what the per-session criteria scripts check for.

## Diagnostic Scripts Created

All located at `local/code/`:

| Script | Purpose |
|--------|---------|
| `_debug_packed.py` | Initial packed session inspector |
| `_debug_packed2.py` | Fixed version — shows metrics distribution per arm |
| `_simulate_eval.py` | Simulates full pre-eval + eval pipeline, shows criteria scripts and tool distribution |
| `_verify_packing.py` | Compares packed data against raw scraper files (full scan) |
| `_verify_packing_fast.py` | Fast version — spot-checks + exhaustive string search |
| `_find_canmore.py` | Located where `canmore_create_textdoc` appears in raw responses |
| `_show_criteria.py` | Dumps criteria scripts from base sessions |
| `_summarize_eval.py` | Comprehensive analysis across all 49 sessions vs results.md |

## Next Steps (TBD)

- Determine if tool name convention mismatches are expected or should be fixed in eval criteria.
- Decide whether `international_my_manager` mock user mismatch ("Laura Park" vs "Dmitrij Petters") is a config issue to fix.
- Investigate low-score sessions (`completeness_hints`, `people_eval_domain`, `personalization`) if needed.
- Consider updating pre-eval script or criteria to handle both naming conventions.
