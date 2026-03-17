# Task Queue

| id | status | title | prompt |
| --- | --- | --- | --- |
| 1 | pending | Measure phase-one nodes | Run a practical short-batch evaluation for the phase-one anti-predict layer and summarize the six node frequencies versus targets using saved CSVs. Do not implement second-phase sizing work in this task. |
| 2 | pending | Tighten turn check nodes | Based on the measured phase-one frequencies, adjust only first-phase node-action targets or bias strength so `flop_check_to_us` and `turn_check_to_us` move closer to their target raise frequencies without adding any sizing-distribution logic. Keep edits inside submission only. |
| 3 | pending | Fix river overcall leak | Review `facing_river_raise` behavior under the phase-one anti-predict system and tighten only the first-phase action-frequency logic if short-batch evidence still shows overcalling. Do not add second-phase sizing logic. |
| 4 | pending | Re-run phase-one validation | Re-run a short validation batch after the phase-one tuning changes and report node-frequency deltas, bankroll outcome, and any runtime issues. Stay within first-phase scope only. |
| 5 | pending | Document phase-one status | Add a short repository note describing the current first-phase anti-predict design, which six nodes are covered, what remains unresolved, and why second-phase sizing is intentionally deferred. |

Notes:
- Keep one task per row.
- Use `pending`, `in_progress`, `done`, or `blocked`.
- Put the actual actionable instruction in the `prompt` column.
- Prefer small, concrete tasks that can be completed in one turn.
- For this queue, do not implement second-phase node-level bet sizing unless the user later changes scope explicitly.
