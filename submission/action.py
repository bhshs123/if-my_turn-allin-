import os
import random
from dataclasses import dataclass
from itertools import combinations

from gym_env import PokerEnv
from submission.strategies.basic import (
    predict_hand_winrate,
    update_pool,
    board_completion_threat,
    adjust_winrate_for_opp_bet,
    hand_rank_class,
)


DEBUG_ACTION = os.getenv("BOT_DEBUG_ACTION", "0") == "1"


def _debug_log(msg: str) -> None:
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


def _raise_fraction_for_winrate(winrate: int) -> float:
    if winrate > 80:
        return 0.20
    if winrate > 70:
        return 0.15
    return 0.12


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


def _base_raise_amount(winrate: int, min_raise: int, max_raise: int) -> int:
    target = int(max_raise * _raise_fraction_for_winrate(winrate))
    return _clamp_int(max(min_raise, target), min_raise, max_raise)


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


def _apply_action_biases(evs: dict[str, float], action_biases: dict[str, float] | None) -> dict[str, float]:
    if not action_biases:
        return evs

    adjusted = dict(evs)
    for action_name, bias in action_biases.items():
        if action_name in adjusted:
            adjusted[action_name] += float(bias)
    return adjusted


def _refresh_remaining_pool(remaining_card_pool, valid_hole, valid_community=None, extra_dead=None):
    remaining_card_pool.clear()
    remaining_card_pool.update(range(27))

    seen_cards = list(valid_hole)
    if valid_community is not None:
        seen_cards.extend(valid_community)
    # Exclude known discarded cards (mine + opponent''s) -- they''re out of play.
    if extra_dead is not None:
        seen_cards.extend(extra_dead)

    update_pool(seen_cards)
    for c in seen_cards:
        remaining_card_pool.discard(c)


def ev_action_decision(
    winrate: int,
    valid_actions: list,
    min_raise: int,
    max_raise: int,
    pot_size: int,
    my_bet: int,
    opp_bet: int,
    opp_action_probs: dict,
    street: int,
    board_threat: float = 0.0,
    opp_pressure: float = 0.0,
    hand_class: int = 9,
    aggression_scale: float = 1.0,
    action_biases: dict[str, float] | None = None,
    raise_sizing: dict | None = None,
    exploration: ExplorationSettings | None = None,
    rng=None,
) -> tuple:
    """EV-based fold/call/raise/check decision (game tree spec, Sections 4-5).

    EV(fold)    = 0
    EV(call)    = P(win) * (pot + call_amount) - call_amount
    EV(raise r) = P(opp_fold) * pot
                  + P(opp_continues) * (P(win) * (pot + call_amount + 2r) - (call_amount + r))

    P(opp_fold) = gamma[n, "FOLD"] = sum_b beta[n,b] * sigma[n,b,fold]
    from the DBBR opponent model (or baseline heuristic during warmup).

    Argmax EV determines the action. CHECK always dominates FOLD (it is free).
    """
    action_types = PokerEnv.ActionType
    p_win = max(0.0, min(1.0, winrate / 100.0))
    aggression_scale = _clamp_float(float(aggression_scale), 0.6, 1.1)
    call_amount = max(0, opp_bet - my_bet)
    effective_pot = max(int(pot_size or 0), int(my_bet or 0) + int(opp_bet or 0))
    pot = max(1, effective_pot)

    can_fold = bool(valid_actions[action_types.FOLD.value])
    can_call = bool(valid_actions[action_types.CALL.value])
    can_check = bool(valid_actions[action_types.CHECK.value])
    can_raise = bool(valid_actions[action_types.RAISE.value]) and max_raise >= min_raise > 0

    # EV(fold) = 0 -- give up, no further chip commitment
    ev_fold = 0.0

    # EV(call/check): if no bet to call this is a free check
    if call_amount > 0:
        # Range-compression discount: opponent's range when betting is stronger
        # than random. ProbabilityAgent raises with equity > 0.75, so our
        # random-sample winrate overestimates equity against their actual range.
        # Discount scales with bet/pot ratio, capped at 25%.
        p_win_call = p_win
        if street >= 1:
            bet_ratio = call_amount / max(1, pot)
            range_discount = 1.0 - min(0.25, bet_ratio * 0.55)
            if street >= 2 and opp_pressure > 0.0:
                range_discount *= 1.0 - min(0.12, 0.10 * opp_pressure)
            p_win_call = p_win * range_discount
        ev_call = p_win_call * (pot + call_amount) - call_amount
    else:
        ev_call = p_win * pot  # free check: rough equity share of pot

    # P(opp_fold) = gamma[n,"FOLD"] = sum_b beta[n,b]*sigma[n,b,fold]  (Section 4)
    p_opp_fold = float(opp_action_probs.get("FOLD", 0.0)) if opp_action_probs else 0.0
    p_opp_raise = float(opp_action_probs.get("RAISE", 0.0)) if opp_action_probs else 0.0
    p_opp_continues = 1.0 - p_opp_fold

    # EV(raise r): opp folds -> win current pot; opp continues -> go to showdown
    # Continuation range after we raise is stronger than random, especially on
    # later streets and dangerous boards.
    p_win_raise = p_win
    if street >= 1:
        pressure = (call_amount / max(1.0, pot)) + max(0.0, min(1.0, float(board_threat))) * 0.8
        pressure += max(0.0, min(1.0, float(opp_pressure))) * 0.6
        raise_discount = min(0.35, 0.18 * pressure)
        p_win_raise = p_win * (1.0 - raise_discount)

    raise_r_base = _base_raise_amount_for_spot(
        winrate=winrate,
        min_raise=min_raise,
        max_raise=max_raise,
        street=street,
        board_threat=board_threat,
        call_amount=call_amount,
        pot=pot,
    )
    raise_r_choice_base = raise_r_base
    chips_in = call_amount + raise_r_base
    showdown_pot = pot + call_amount + 2 * raise_r_base
    ev_raise = (
        p_opp_fold * pot
        + p_opp_continues * (p_win_raise * showdown_pot - chips_in)
    )
    ev_raise += _made_hand_raise_bonus(street, hand_class, board_threat, pot)

    # Build EV map over valid actions
    evs = {}
    raise_threshold_by_street = {0: 60, 1: 68, 2: 72, 3: 80}
    raise_threshold = raise_threshold_by_street.get(street, 75)

    # Opponent-specific raise discipline: reduce bluffing into sticky opponents,
    # increase pressure against overfolders.
    if street >= 1:
        if p_opp_fold <= 0.18:
            raise_threshold += 4
        elif p_opp_fold >= 0.36:
            raise_threshold -= 3
        raise_threshold += int(round(max(0.0, min(1.0, opp_pressure)) * 4))
        if aggression_scale < 1.0:
            raise_threshold += int(round((1.0 - aggression_scale) * 8))

    allow_raise = can_raise and winrate >= raise_threshold

    if street == 0 and allow_raise:
        # More consistent preflop policy: avoid thin opens/3-bets that created
        # unstable outcomes across 3.16 CSV files.
        if call_amount == 0:
            if winrate < 58:
                allow_raise = False
            elif winrate < 66 and p_opp_fold < 0.34:
                allow_raise = False
        else:
            if winrate < 63 and p_opp_fold < 0.40:
                allow_raise = False

    if street >= 2 and allow_raise:
        # On turn/river, only raise when EV edge is meaningful unless we are very strong.
        ev_edge = ev_raise - ev_call
        min_edge = 0.08 * pot
        if winrate < 85 and ev_edge < min_edge:
            allow_raise = False
        # Late-street value raises should come mostly from made hands.
        if hand_class > 7 and p_opp_fold < 0.42:
            allow_raise = False

    if street == 3 and allow_raise:
        river_edge = ev_raise - ev_call
        if winrate < 90:
            if board_threat >= 0.45 and p_opp_fold < 0.30:
                allow_raise = False
            elif p_opp_raise >= 0.40 and river_edge < 0.14 * pot:
                allow_raise = False
            elif hand_class > 7:
                allow_raise = False

    # Fold-equity semi-bluff lane on flop/turn in high-pressure spots.
    allow_pressure_raise = (
        can_raise
        and street in (1, 2)
        and call_amount > 0
        and (raise_threshold - 12) <= winrate < raise_threshold
        and p_opp_fold >= 0.48
        and board_threat >= 0.55
        and opp_pressure <= 0.55
    )

    if allow_raise:
        evs["RAISE"] = ev_raise
        raise_r_choice_base = raise_r_base
    elif allow_pressure_raise:
        bluff_raise = _clamp_int(max(min_raise, int(max_raise * 0.14)), min_raise, max_raise)
        bluff_chips_in = call_amount + bluff_raise
        bluff_showdown_pot = pot + call_amount + 2 * bluff_raise
        # Semi-bluffs realize less equity when called.
        p_win_bluff = p_win_raise * 0.88
        ev_pressure = p_opp_fold * pot + p_opp_continues * (p_win_bluff * bluff_showdown_pot - bluff_chips_in)
        if ev_pressure >= ev_call + 0.03 * pot:
            evs["RAISE"] = ev_pressure
            raise_r_choice_base = bluff_raise

    # Preflop steal lane: small open into fold-prone opponents.
    allow_preflop_steal = (
        can_raise
        and street == 0
        and call_amount == 0
        and aggression_scale >= 0.85
        and 50 <= winrate < 58
        and p_opp_fold >= 0.42
    )
    if allow_preflop_steal and "RAISE" not in evs:
        steal_raise = _clamp_int(max(min_raise, int(max_raise * 0.10)), min_raise, max_raise)
        steal_showdown_pot = pot + 2 * steal_raise
        ev_steal = p_opp_fold * pot + p_opp_continues * (p_win * steal_showdown_pot - steal_raise)
        if ev_steal >= ev_call + 0.02 * pot:
            evs["RAISE"] = ev_steal
            raise_r_choice_base = steal_raise

    # Probe-bet lane: when checked to us on flop/turn with medium equity,
    # apply pressure using a small sizing to realize fold equity.
    allow_probe_raise = (
        can_raise
        and call_amount == 0
        and street in (1, 2)
        and 46 <= winrate < raise_threshold
        and p_opp_fold >= 0.36
        and opp_pressure <= 0.65
    )
    if allow_probe_raise and street == 1:
        # Dry flops can sustain lighter probes; wet flops should probe mostly
        # with at least pair-level hands to avoid spewy semi-bluffs.
        if board_threat >= 0.55:
            allow_probe_raise = hand_class <= 8 and winrate >= 54
        elif board_threat <= 0.30:
            allow_probe_raise = winrate >= 44

    if allow_probe_raise and "RAISE" not in evs:
        probe_raise = _clamp_int(max(min_raise, int(max_raise * 0.11)), min_raise, max_raise)
        probe_showdown_pot = pot + 2 * probe_raise
        p_win_probe = p_win_raise * 0.92
        ev_probe = p_opp_fold * pot + p_opp_continues * (p_win_probe * probe_showdown_pot - probe_raise)
        if ev_probe >= ev_call + 0.02 * pot:
            evs["RAISE"] = ev_probe
            raise_r_choice_base = probe_raise

    # Delayed c-bet on turn after checked pots: attack passive lines observed in CSV.
    allow_delayed_turn_cbet = (
        can_raise
        and street == 2
        and call_amount == 0
        and aggression_scale >= 0.90
        and pot <= 30
        and p_opp_fold >= 0.32
        and opp_pressure <= 0.45
        and winrate >= 38
    )
    if allow_delayed_turn_cbet and "RAISE" not in evs:
        delayed_raise = _clamp_int(max(min_raise, int(max_raise * 0.12)), min_raise, max_raise)
        delayed_showdown_pot = pot + 2 * delayed_raise
        p_win_delayed = p_win_raise * 0.90
        ev_delayed = p_opp_fold * pot + p_opp_continues * (p_win_delayed * delayed_showdown_pot - delayed_raise)
        if ev_delayed >= ev_call + 0.02 * pot:
            evs["RAISE"] = ev_delayed
            raise_r_choice_base = delayed_raise

    # River thin value stab in passive checked pots.
    allow_river_thin_value = (
        can_raise
        and street == 3
        and call_amount == 0
        and aggression_scale >= 0.95
        and hand_class <= 8
        and winrate >= 60
        and p_opp_raise <= 0.28
        and opp_pressure <= 0.35
    )
    if allow_river_thin_value and "RAISE" not in evs:
        thin_raise = _clamp_int(max(min_raise, int(max_raise * 0.10)), min_raise, max_raise)
        thin_showdown_pot = pot + 2 * thin_raise
        ev_thin = p_opp_fold * pot + p_opp_continues * (p_win_raise * thin_showdown_pot - thin_raise)
        if ev_thin >= ev_call + 0.015 * pot:
            evs["RAISE"] = ev_thin
            raise_r_choice_base = thin_raise
    if can_call:
        evs["CALL"] = ev_call
    if can_check:
        evs["CHECK"] = ev_call  # same formula (call_amount == 0 when CHECK is legal)
    if can_fold:
        evs["FOLD"] = ev_fold

    # Late streets vs aggressive bets: require stronger equity than raw pot-odds
    # to avoid thin bluff-catching against ProbabilityAgent's strong raise range.
    if call_amount > 0 and street >= 2 and "CALL" in evs:
        pot_odds_pct = (100.0 * call_amount) / max(1.0, (pot + call_amount))
        bet_ratio = call_amount / max(1.0, pot)
        if street == 2:
            safety_buffer = 10 + min(12.0, 12.0 * bet_ratio)
        else:
            safety_buffer = 14 + min(18.0, 18.0 * bet_ratio)
        # Connected/paired/suited boards reduce bluff frequency from most bots.
        safety_buffer += max(0.0, min(1.0, float(board_threat))) * 10.0
        # If villain raises frequently on this node, tighten bluff-catching.
        safety_buffer += max(0.0, (p_opp_raise - 0.35)) * 20.0
        safety_buffer += max(0.0, min(1.0, float(opp_pressure))) * 8.0

        # One-pair / high-card hands should not hero-call large late pressure often.
        if hand_class >= 8:
            if street == 2:
                safety_buffer += 7.0 + 8.0 * bet_ratio
            else:
                safety_buffer += 11.0 + 12.0 * bet_ratio

        min_call_winrate = pot_odds_pct + safety_buffer
        # River near-all-in bets are heavily value-weighted in this opponent;
        # enforce a stronger bluff-catch floor even when pot odds look good.
        if street == 3 and opp_bet >= int(max_raise * 0.8):
            min_call_winrate = max(min_call_winrate, 45.0)
        elif street == 3 and opp_bet >= int(max_raise * 0.5):
            min_call_winrate = max(min_call_winrate, 38.0)

        # Additional turn/river pressure folds for weak made hands.
        if hand_class >= 8:
            if street == 2 and (bet_ratio >= 0.45 or opp_pressure >= 0.55):
                min_call_winrate = max(min_call_winrate, 52.0)
            if street == 3 and (bet_ratio >= 0.30 or opp_pressure >= 0.40):
                min_call_winrate = max(min_call_winrate, 60.0)

        if winrate < min_call_winrate:
            evs.pop("CALL", None)

    if not evs:
        return action_types.FOLD.value, 0, 0, 0

    # CHECK always dominates FOLD -- never fold when checking is free
    if can_check and "FOLD" in evs:
        evs.pop("FOLD")

    evs = _apply_action_biases(evs, action_biases)

    best, candidates = select_action_with_exploration(
        evs,
        pot=pot,
        exploration=exploration,
        rng=rng,
    )

    # Near breakeven calls are sensitive to simulation/model noise.
    # Only apply anti-overfold in early/small-pot situations.
    if best == "FOLD" and can_call:
        noise_margin = 0.05 * pot
        small_call = call_amount <= int(0.10 * max(1, pot))
        if street <= 1 and small_call and ev_call >= -noise_margin:
            best = "CALL"

    raise_r_final = randomized_raise_amount(
        raise_r_choice_base,
        min_raise=min_raise,
        max_raise=max_raise,
        exploration=exploration,
        raise_sizing=raise_sizing,
        pot=pot,
        call_amount=call_amount,
        street=street,
        rng=rng,
    )

    _debug_log(
        f"[ev] street={street} winrate={winrate} call={call_amount} pot={pot} "
        f"p_opp_fold={p_opp_fold:.2f} "
        f"opp_pressure={opp_pressure:.2f} "
        f"hand_class={hand_class} "
        f"EV(fold)={ev_fold:.1f} EV(call/chk)={ev_call:.1f} EV(raise)={ev_raise:.1f} "
        f"biases={action_biases or {}} sizing={(raise_sizing or {}).get('weights', {}) if raise_sizing else {}} "
        f"mix_p={(exploration.mix_probability if exploration else 0.0):.2f} "
        f"candidates={[(name, round(ev, 2)) for name, ev in candidates]} -> {best}"
    )

    if best == "RAISE":
        return action_types.RAISE.value, raise_r_final, 0, 0
    if best == "CALL":
        return action_types.CALL.value, 0, 0, 0
    if best == "CHECK":
        return action_types.CHECK.value, 0, 0, 0
    return action_types.FOLD.value, 0, 0, 0


def preflop_action(my_cards, remaining_card_pool, valid_actions, min_raise, max_raise,
                   pot_size=0, my_bet=0, opp_bet=0, opp_action_probs=None,
                   opp_pressure: float = 0.0,
                   aggression_scale: float = 1.0,
                   action_biases: dict[str, float] | None = None,
                   raise_sizing: dict | None = None,
                   exploration: ExplorationSettings | None = None, rng=None):
    valid_hole = [c for c in my_cards if isinstance(c, int) and c >= 0]
    _refresh_remaining_pool(remaining_card_pool, valid_hole)

    trials = _winrate_trials_for_street(0, pot_size, my_bet, opp_bet)
    if len(valid_hole) == 2:
        winrate = predict_hand_winrate(valid_hole, remaining_card_pool, [], trials=trials)
    else:
        winrates = []
        for combo in combinations(valid_hole, 2):
            wr = predict_hand_winrate(list(combo), remaining_card_pool, [], trials=trials)
            winrates.append(wr)
        winrate = max(winrates) if winrates else 0
    _debug_log(f"[preflop_action] winrate={winrate}")
    return ev_action_decision(winrate, valid_actions, min_raise, max_raise,
                              pot_size, my_bet, opp_bet, opp_action_probs, street=0,
                              opp_pressure=opp_pressure,
                              hand_class=9,
                              aggression_scale=aggression_scale,
                              action_biases=action_biases,
                              raise_sizing=raise_sizing,
                              exploration=exploration, rng=rng)


def flop_action(my_cards, community_cards, remaining_card_pool, valid_actions, min_raise, max_raise,
                dead_cards=None, pot_size=0, my_bet=0, opp_bet=0, opp_action_probs=None,
                opp_pressure: float = 0.0,
                aggression_scale: float = 1.0,
                action_biases: dict[str, float] | None = None,
                raise_sizing: dict | None = None,
                exploration: ExplorationSettings | None = None, rng=None):
    valid_hole = [c for c in my_cards if isinstance(c, int) and c >= 0]
    community = [c for c in community_cards if isinstance(c, int) and c >= 0]
    _refresh_remaining_pool(remaining_card_pool, valid_hole, community, extra_dead=dead_cards)

    trials = _winrate_trials_for_street(1, pot_size, my_bet, opp_bet, board_threat=board_completion_threat(community))
    winrate = predict_hand_winrate(valid_hole, remaining_card_pool, community, trials=trials)
    hand_class = hand_rank_class(valid_hole, community)
    _debug_log(f"[flop_action] winrate={winrate}")
    return ev_action_decision(winrate, valid_actions, min_raise, max_raise,
                              pot_size, my_bet, opp_bet, opp_action_probs, street=1,
                              opp_pressure=opp_pressure,
                              hand_class=hand_class,
                              aggression_scale=aggression_scale,
                              action_biases=action_biases,
                              raise_sizing=raise_sizing,
                              exploration=exploration, rng=rng)


def discard_action(my_cards, community_cards, remaining_card_pool, dead_cards=None):
    valid_hole = [c for c in my_cards if isinstance(c, int) and c >= 0]
    valid_community = [c for c in community_cards if isinstance(c, int) and c >= 0]
    _refresh_remaining_pool(remaining_card_pool, valid_hole, valid_community, extra_dead=dead_cards)

    best_score = -1.0
    best_indices = (0, 1)
    board_threat = board_completion_threat(valid_community)
    for combo in combinations(range(len(valid_hole)), 2):
        hand = [valid_hole[i] for i in combo]
        wr = predict_hand_winrate(hand, remaining_card_pool, valid_community, trials=140)
        hclass = hand_rank_class(hand, valid_community)
        made_bonus = 0.0
        if hclass <= 7:
            made_bonus = 6.0
        elif hclass == 8:
            made_bonus = 3.0 + 2.0 * board_threat
        score = wr + made_bonus
        if score > best_score:
            best_score = score
            best_indices = combo
    _debug_log(f"[discard_action] score={best_score:.1f}")
    action_types = PokerEnv.ActionType
    return action_types.DISCARD.value, 0, best_indices[0], best_indices[1]


def turn_action(my_cards, community_cards, remaining_card_pool, valid_actions, min_raise, max_raise,
                dead_cards=None, pot_size=0, my_bet=0, opp_bet=0, opp_action_probs=None,
                opp_pressure: float = 0.0,
                aggression_scale: float = 1.0,
                action_biases: dict[str, float] | None = None,
                raise_sizing: dict | None = None,
                exploration: ExplorationSettings | None = None, rng=None):
    valid_hole = [c for c in my_cards if isinstance(c, int) and c >= 0]
    community = [c for c in community_cards if isinstance(c, int) and c >= 0]
    _refresh_remaining_pool(remaining_card_pool, valid_hole, community, extra_dead=dead_cards)

    threat = board_completion_threat(community)
    trials = _winrate_trials_for_street(2, pot_size, my_bet, opp_bet, board_threat=threat)
    raw_winrate = predict_hand_winrate(valid_hole, remaining_card_pool, community, trials=trials)
    hand_class = hand_rank_class(valid_hole, community)
    opp_raise = max(0, int(opp_bet) - int(my_bet))
    pot_before = max(1, int(pot_size) - opp_raise)
    winrate = adjust_winrate_for_opp_bet(raw_winrate, opp_raise, pot_before, community)
    _debug_log(f"[turn_action] winrate={winrate} raw={raw_winrate} opp_raise={opp_raise} threat={threat:.2f}")
    return ev_action_decision(winrate, valid_actions, min_raise, max_raise,
                              pot_size, my_bet, opp_bet, opp_action_probs, street=2, board_threat=threat,
                              opp_pressure=opp_pressure,
                              hand_class=hand_class,
                              aggression_scale=aggression_scale,
                              action_biases=action_biases,
                              raise_sizing=raise_sizing,
                              exploration=exploration, rng=rng)


def river_action(my_cards, community_cards, remaining_card_pool, valid_actions, min_raise, max_raise,
                 dead_cards=None, pot_size=0, my_bet=0, opp_bet=0, opp_action_probs=None,
                 opp_pressure: float = 0.0,
                 aggression_scale: float = 1.0,
                 action_biases: dict[str, float] | None = None,
                 raise_sizing: dict | None = None,
                 exploration: ExplorationSettings | None = None, rng=None):
    valid_hole = [c for c in my_cards if isinstance(c, int) and c >= 0]
    community = [c for c in community_cards if isinstance(c, int) and c >= 0]
    _refresh_remaining_pool(remaining_card_pool, valid_hole, community, extra_dead=dead_cards)

    threat = board_completion_threat(community)
    trials = _winrate_trials_for_street(3, pot_size, my_bet, opp_bet, board_threat=threat)
    raw_winrate = predict_hand_winrate(valid_hole, remaining_card_pool, community, trials=trials)
    hand_class = hand_rank_class(valid_hole, community)
    opp_raise = max(0, int(opp_bet) - int(my_bet))
    pot_before = max(1, int(pot_size) - opp_raise)
    winrate = adjust_winrate_for_opp_bet(raw_winrate, opp_raise, pot_before, community)
    _debug_log(f"[river_action] winrate={winrate} raw={raw_winrate} opp_raise={opp_raise} threat={threat:.2f}")
    return ev_action_decision(winrate, valid_actions, min_raise, max_raise,
                              pot_size, my_bet, opp_bet, opp_action_probs, street=3, board_threat=threat,
                              opp_pressure=opp_pressure,
                              hand_class=hand_class,
                              aggression_scale=aggression_scale,
                              action_biases=action_biases,
                              raise_sizing=raise_sizing,
                              exploration=exploration, rng=rng)


def call_function(call_amount, winrate, my_bet, max_raise):
    action_types = PokerEnv.ActionType
    if call_amount == 0:
        return action_types.CALL.value, 0, 0, 0
    if winrate > 80:
        return action_types.CALL.value, 0, 0, 0
    elif winrate > 50 and (my_bet + call_amount) <= max_raise * 3 / 20:
        return action_types.CALL.value, 0, 0, 0
    elif winrate > 30 and (my_bet + call_amount) <= max_raise / 10:
        return action_types.CALL.value, 0, 0, 0
    else:
        return action_types.FOLD.value, 0, 0, 0
