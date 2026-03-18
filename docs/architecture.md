# Submission Architecture

## Module Layout

```
submission/
  player.py            — PlayerAgent: the top-level orchestrator.
  exploration.py       — RecentPerformanceTracker: rolling match-result drift
                         detector and exploration-window manager.
  anti_predict.py      — AntiPredictTracker: node-level action-frequency targets
                         and sizing profiles (Phases 1 & 2).
  action.py            — Compatibility facade: re-exports public symbols and
                         provides street action wrappers (preflop/flop/turn/river/
                         discard) that compose winrate estimation with EV decision.
  action_shared.py     — Shared utilities: ExplorationSettings, raise-fraction
                         helpers, sizing-bucket logic, randomized_raise_amount,
                         exploration_candidates, select_action_with_exploration.
  action_ev.py         — Core EV decision engine: ev_action_decision and the
                         bias applier it uses internally.
  opponent_modeling.py — DBBROpponentModel: double-bet/bet-ratio Bayesian model
                         that estimates opponent action probabilities per game-tree
                         node and optionally overrides EV actions.
  strategies/
    basic.py           — Low-level simulation helpers: predict_hand_winrate,
                         hand_rank_class, board_completion_threat, pool tracking.
```

## Anti-Predict System

### Phase 1 — Action Frequency Targeting

Six named game-tree nodes are tracked in `AntiPredictTracker`:

| node key | default raise target | purpose |
|---|---|---|
| `preflop_first_action` | 0.40 | unpredictable open frequency |
| `preflop_facing_raise` | 0.22 | 3-bet mix |
| `flop_check_to_us` | 0.38 | delayed c-bet / probe |
| `flop_facing_raise` | 0.15 | float / fold discipline |
| `turn_check_to_us` | 0.32 | turn probe |
| `facing_river_raise` | 0.10 | thin river value / re-raise |

When the observed raise rate for a node drifts from its target the tracker
injects `action_biases` — small additive EV adjustments — that nudge the
frequency back toward the target over the next few decisions.

### Phase 2 — Bet Sizing Distribution

Each proactive node (preflop first action, flop / turn checks) carries a
sizing profile with three buckets:

- **small** — undersized bet (~78–84 % of base), used to polarise range.
- **standard** — base raise from the EV layer.
- **pressure** — oversized bet (~122–128 % of base), used for thin value / deny equity.

The bucket for each raise decision is sampled by weight from
`_choose_sizing_bucket`, then fed to `_apply_sizing_bucket` before the
jitter layer in `randomized_raise_amount`.

## Key Design Decisions

- **Behavior-preserving split**: `action.py` remains the single import point
  for all external callers; internal helpers live in `action_shared.py` and
  `action_ev.py`.
- **Re-exports for backward compat**: `player.py` re-exports
  `RecentPerformanceTracker` so existing tests do not need to change their
  import paths.
- **EV policy is primary**: DBBR overrides and anti-predict biases can only
  nudge the EV outcome — neither replaces a free check with a fold nor
  forces a call when EV says fold.
