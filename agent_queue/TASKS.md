# Task Queue

| id | status | title | prompt |
| --- | --- | --- | --- |
| 1 | done | Extract anti-predict module | Refactor anti-predict queue work so node target profiles, bias logic, and sizing profiles move out of the large submission files into a dedicated submission module, without changing external behavior. |
| 2 | done | Split action helpers | Refactor submission action logic into smaller modules or helper files so street decision policy, sizing logic, and shared EV utilities are no longer concentrated in one large file. Preserve behavior as much as possible. |
| 3 | done | Split player orchestration | Refactor submission player orchestration so exploration tracking, anti-predict tracking, and opponent-model integration are separated into clearer modules or classes under submission. Preserve behavior as much as possible. |
| 4 | done | Validate modular refactor | Run static checks and a short smoke match after the modularization tasks, and summarize whether behavior appears preserved plus any residual risks. |
| 5 | done | Document architecture status | Add a short repository note describing the current anti-predict architecture, what first- and second-phase mechanisms exist, and how the modularized submission layout is organized for future work. |
| 6 | done | Measure phase-one nodes | Run a practical short-batch evaluation for the phase-one anti-predict layer and summarize the six node frequencies versus targets using saved CSVs. |
| 7 | done | Tighten turn check nodes | Based on the measured phase-one frequencies, adjust only first-phase node-action targets or bias strength so `flop_check_to_us` and `turn_check_to_us` move closer to their target raise frequencies. Keep edits inside submission only. |
| 8 | done | Fix river overcall leak | Review `facing_river_raise` behavior under the phase-one anti-predict system and tighten the first-phase action-frequency logic if short-batch evidence still shows overcalling. |
| 9 | done | Re-run phase-one validation | Re-run a short validation batch after the phase-one tuning changes and report node-frequency deltas, bankroll outcome, and any runtime issues. |
| 10 | done | Add second-phase sizing check | Run a practical short-batch evaluation for the second-phase node-level sizing layer and summarize whether sizing diversity appears in the key proactive nodes. |
| 11 | done | Tune sizing buckets | Adjust second-phase sizing bucket weights for the key proactive nodes if the short-batch results show sizing is still too concentrated or too passive. Keep the change lightweight and inside submission only. |
| 12 | done | Validate phase-two behavior | Re-run a short validation batch after the second-phase sizing tuning and report node-frequency and sizing-distribution deltas, bankroll outcome, and runtime issues. |

Notes:
- Keep one task per row.
- Use `pending`, `in_progress`, `done`, or `blocked`.
- Put the actual actionable instruction in the `prompt` column.
- Prefer small, concrete tasks that can be completed in one turn.
- Queue scope now includes both anti-predict phase-one and phase-two work, plus modular refactors under `submission/`.
- Preserve behavior during modularization unless a queued task explicitly asks for a behavioral adjustment.
