import os
import random
from dataclasses import dataclass


DEBUG_ACTION = os.getenv("BOT_DEBUG_ACTION", "0") == "1"


def debug_log(msg: str) -> None:
    if DEBUG_ACTION:
        print(msg)


@dataclass(frozen=True)
class ExplorationSettings:
    """Small bounded-variance layer over the deterministic EV policy.

    `mix_probability` controls whether we occasionally sample among near-equal EV
    actions. `raise_jitter_pct` adds a mild amount of raise-size variation around
    the existing winrate-based sizing curve.
    """

    mix_probability: float = 0.0
    max_candidate_actions: int = 3
    ev_margin_pct: float = 0.04
    ev_margin_floor: float = 1.5
    raise_jitter_pct: float = 0.05


def _clamp_float(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def _raise_fraction_for_spot(
    winrate: int,
    street: int,
    board_threat: float,
    call_amount: int,
    pot: int,
) -> float:
    # Bigger value bets on later streets deny equity on dangerous runouts.
    if winrate >= 88:
        base = 0.34 if street >= 2 else 0.26
    elif winrate >= 80:
        base = 0.26 if street >= 2 else 0.20
    elif winrate >= 72:
        base = 0.20 if street >= 2 else 0.16
    else:
        base = 0.14

    pressure = call_amount / max(1.0, float(pot))
    if street >= 2 and board_threat >= 0.55 and winrate >= 72:
        base += 0.05
    if street == 1 and board_threat >= 0.55 and winrate >= 62:
        base += 0.04
    if pressure >= 0.45 and winrate < 60:
        base = min(base, 0.14)

    return _clamp_float(base, 0.10, 0.42)


def _base_raise_amount_for_spot(
    winrate: int,
    min_raise: int,
    max_raise: int,
    street: int,
    board_threat: float,
    call_amount: int,
    pot: int,
) -> int:
    frac = _raise_fraction_for_spot(
        winrate=winrate,
        street=street,
        board_threat=board_threat,
        call_amount=call_amount,
        pot=pot,
    )
    target = int(max_raise * frac)
    # On flop, protect medium-strong made hands and draws on wet boards with a
    # more stable sizing floor so we do not give cheap cards.
    if street == 1 and board_threat >= 0.55 and winrate >= 62:
        target = max(target, 15)
    return _clamp_int(max(min_raise, target), min_raise, max_raise)


def _made_hand_raise_bonus(street: int, hand_class: int, board_threat: float, pot: int) -> float:
    """Reward protection/value betting with made hands on flop/turn.

    This nudges medium-strength made hands away from passive check lines that
    repeatedly showed up in the CSV losses.
    """
    if street not in (1, 2):
        return 0.0
    if hand_class >= 9:
        return 0.0

    if hand_class <= 7:
        base = 0.10 * pot
    elif hand_class == 8:
        base = 0.06 * pot
    else:
        base = 0.0

    if board_threat >= 0.55:
        base += 0.05 * pot
    return base


def _winrate_trials_for_street(street: int, pot_size: int, my_bet: int, opp_bet: int, board_threat: float = 0.0) -> int:
    base_by_street = {0: 100, 1: 130, 2: 170, 3: 190}
    base = base_by_street.get(street, 180)
    call_amount = max(0, int(opp_bet) - int(my_bet))
    pressure = call_amount / max(1.0, float(max(1, pot_size)))
    multiplier = 1.0 + min(0.55, 0.65 * pressure + 0.45 * max(0.0, min(1.0, board_threat)))
    return _clamp_int(int(base * multiplier), 80, 300)


def _choose_sizing_bucket(weights: dict[str, float], rng=None) -> str:
    chooser = rng if rng is not None else random
    total = sum(max(0.0, float(weight)) for weight in weights.values())
    if total <= 0.0:
        return "standard"

    threshold = chooser.random() * total
    cumulative = 0.0
    for bucket in ("small", "standard", "pressure"):
        cumulative += max(0.0, float(weights.get(bucket, 0.0)))
        if threshold <= cumulative:
            return bucket
    return "standard"


def _apply_sizing_bucket(
    base_raise: int,
    bucket: str,
    min_raise: int,
    max_raise: int,
    pot: int,
    call_amount: int,
    street: int,
) -> int:
    base_raise = _clamp_int(base_raise, min_raise, max_raise)
    if bucket == "small":
        target = int(base_raise * (0.78 if call_amount == 0 else 0.84))
        if call_amount == 0:
            target = max(target, int(max(1, pot) * (0.16 if street == 0 else 0.22)))
    elif bucket == "pressure":
        target = int(base_raise * (1.28 if call_amount == 0 else 1.22))
        if call_amount == 0:
            target = max(target, int(max(1, pot) * (0.24 if street <= 1 else 0.30)))
    else:
        target = base_raise
    return _clamp_int(max(min_raise, target), min_raise, max_raise)


def randomized_raise_amount(
    base_raise: int,
    min_raise: int,
    max_raise: int,
    exploration: ExplorationSettings | None = None,
    raise_sizing: dict | None = None,
    pot: int = 0,
    call_amount: int = 0,
    street: int = 0,
    rng=None,
) -> int:
    """Add bounded size variation around the base size, optionally via node-level size buckets."""
    if max_raise < min_raise:
        return base_raise

    settings = exploration or ExplorationSettings()
    base_raise = _clamp_int(base_raise, min_raise, max_raise)

    if raise_sizing and raise_sizing.get("weights"):
        bucket = _choose_sizing_bucket(raise_sizing["weights"], rng=rng)
        base_raise = _apply_sizing_bucket(base_raise, bucket, min_raise, max_raise, pot=pot, call_amount=call_amount, street=street)

    jitter_pct = max(0.0, float(settings.raise_jitter_pct))
    if jitter_pct <= 0.0:
        return base_raise

    jitter_span = max(1, int(round(base_raise * jitter_pct)))
    low = max(min_raise, base_raise - jitter_span)
    high = min(max_raise, base_raise + jitter_span)
    if high <= low:
        return low

    chooser = rng if rng is not None else random
    return chooser.randint(low, high)


def exploration_candidates(
    evs: dict[str, float],
    pot: int,
    exploration: ExplorationSettings | None = None,
) -> list[tuple[str, float]]:
    """Return only sensible near-optimal actions for optional exploration mixing."""
    if not evs:
        return []

    settings = exploration or ExplorationSettings()
    ranked = sorted(evs.items(), key=lambda item: item[1], reverse=True)
    best_action, best_ev = ranked[0]

    # Never turn a strong fold into a loose gamble.
    if best_action == "FOLD":
        return [ranked[0]]

    margin = max(float(settings.ev_margin_floor), float(max(1, pot)) * float(settings.ev_margin_pct))
    lower_bound = best_ev - margin
    if best_ev > 0.0:
        lower_bound = max(0.0, lower_bound)

    candidates = [item for item in ranked if item[1] >= lower_bound]
    return candidates[: max(1, int(settings.max_candidate_actions))]


def select_action_with_exploration(
    evs: dict[str, float],
    pot: int,
    exploration: ExplorationSettings | None = None,
    rng=None,
) -> tuple[str, list[tuple[str, float]]]:
    """Choose the best EV action, with rare mixing among near-equal candidates."""
    if not evs:
        raise ValueError("evs must contain at least one legal action")

    settings = exploration or ExplorationSettings()
    candidates = exploration_candidates(evs, pot=pot, exploration=settings)
    best_action = candidates[0][0]

    mix_probability = _clamp_float(float(settings.mix_probability), 0.0, 1.0)
    if mix_probability <= 0.0 or len(candidates) < 2:
        return best_action, candidates

    chooser = rng if rng is not None else random
    if chooser.random() >= mix_probability:
        return best_action, candidates

    best_ev = candidates[0][1]
    margin = max(float(settings.ev_margin_floor), float(max(1, pot)) * float(settings.ev_margin_pct))
    weights = [max(0.01, margin - (best_ev - ev) + 0.01) for _, ev in candidates]
    total_weight = sum(weights)
    threshold = chooser.random() * total_weight

    cumulative = 0.0
    for (action_name, _), weight in zip(candidates, weights):
        cumulative += weight
        if threshold <= cumulative:
            return action_name, candidates
    return candidates[-1][0], candidates
