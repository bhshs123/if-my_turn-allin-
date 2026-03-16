import random
from dataclasses import dataclass
from itertools import combinations

from gym_env import PokerEnv
from submission.strategies.basic import (
    predict_hand_winrate,
    update_pool,
    board_completion_threat,
    adjust_winrate_for_opp_bet,
)


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


def _base_raise_amount(winrate: int, min_raise: int, max_raise: int) -> int:
    target = int(max_raise * _raise_fraction_for_winrate(winrate))
    return _clamp_int(max(min_raise, target), min_raise, max_raise)


def randomized_raise_amount(
    base_raise: int,
    min_raise: int,
    max_raise: int,
    exploration: ExplorationSettings | None = None,
    rng=None,
) -> int:
    """Add a small amount of bounded raise-size variation around the base size."""
    if max_raise < min_raise:
        return base_raise

    settings = exploration or ExplorationSettings()
    jitter_pct = max(0.0, float(settings.raise_jitter_pct))
    base_raise = _clamp_int(base_raise, min_raise, max_raise)
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
            p_win_call = p_win * range_discount
        ev_call = p_win_call * (pot + call_amount) - call_amount
    else:
        ev_call = p_win * pot  # free check: rough equity share of pot

    # P(opp_fold) = gamma[n,"FOLD"] = sum_b beta[n,b]*sigma[n,b,fold]  (Section 4)
    p_opp_fold = float(opp_action_probs.get("FOLD", 0.0)) if opp_action_probs else 0.0
    p_opp_continues = 1.0 - p_opp_fold

    # EV(raise r): opp folds -> win current pot; opp continues -> go to showdown
    # Continuation range after we raise is stronger than random, especially on
    # later streets and dangerous boards.
    p_win_raise = p_win
    if street >= 1:
        pressure = (call_amount / max(1.0, pot)) + max(0.0, min(1.0, float(board_threat))) * 0.8
        raise_discount = min(0.35, 0.18 * pressure)
        p_win_raise = p_win * (1.0 - raise_discount)

    raise_r_base = _base_raise_amount(winrate, min_raise, max_raise)
    chips_in = call_amount + raise_r_base
    showdown_pot = pot + call_amount + 2 * raise_r_base
    ev_raise = (
        p_opp_fold * pot
        + p_opp_continues * (p_win_raise * showdown_pot - chips_in)
    )

    # Build EV map over valid actions
    evs = {}
    raise_threshold_by_street = {0: 60, 1: 68, 2: 72, 3: 80}
    raise_threshold = raise_threshold_by_street.get(street, 75)
    allow_raise = can_raise and winrate >= raise_threshold

    if allow_raise:
        evs["RAISE"] = ev_raise
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
        min_call_winrate = pot_odds_pct + safety_buffer
        # River near-all-in bets are heavily value-weighted in this opponent;
        # enforce a stronger bluff-catch floor even when pot odds look good.
        if street == 3 and opp_bet >= int(max_raise * 0.8):
            min_call_winrate = max(min_call_winrate, 45.0)
        elif street == 3 and opp_bet >= int(max_raise * 0.5):
            min_call_winrate = max(min_call_winrate, 38.0)
        if winrate < min_call_winrate:
            evs.pop("CALL", None)

    if not evs:
        return action_types.FOLD.value, 0, 0, 0

    # CHECK always dominates FOLD -- never fold when checking is free
    if can_check and "FOLD" in evs:
        evs.pop("FOLD")

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
        raise_r_base,
        min_raise=min_raise,
        max_raise=max_raise,
        exploration=exploration,
        rng=rng,
    )

    print(
        f"[ev] street={street} winrate={winrate} call={call_amount} pot={pot} "
        f"p_opp_fold={p_opp_fold:.2f} "
        f"EV(fold)={ev_fold:.1f} EV(call/chk)={ev_call:.1f} EV(raise)={ev_raise:.1f} "
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
                   exploration: ExplorationSettings | None = None, rng=None):
    valid_hole = [c for c in my_cards if isinstance(c, int) and c >= 0]
    _refresh_remaining_pool(remaining_card_pool, valid_hole)

    if len(valid_hole) == 2:
        winrate = predict_hand_winrate(valid_hole, remaining_card_pool, [])
    else:
        winrates = []
        for combo in combinations(valid_hole, 2):
            wr = predict_hand_winrate(list(combo), remaining_card_pool, [])
            winrates.append(wr)
        winrate = max(winrates) if winrates else 0
    print(f"[preflop_action] winrate={winrate}")
    return ev_action_decision(winrate, valid_actions, min_raise, max_raise,
                              pot_size, my_bet, opp_bet, opp_action_probs, street=0,
                              exploration=exploration, rng=rng)


def flop_action(my_cards, community_cards, remaining_card_pool, valid_actions, min_raise, max_raise,
                dead_cards=None, pot_size=0, my_bet=0, opp_bet=0, opp_action_probs=None,
                exploration: ExplorationSettings | None = None, rng=None):
    valid_hole = [c for c in my_cards if isinstance(c, int) and c >= 0]
    community = [c for c in community_cards if isinstance(c, int) and c >= 0]
    _refresh_remaining_pool(remaining_card_pool, valid_hole, community, extra_dead=dead_cards)

    winrate = predict_hand_winrate(valid_hole, remaining_card_pool, community)
    print(f"[flop_action] winrate={winrate}")
    return ev_action_decision(winrate, valid_actions, min_raise, max_raise,
                              pot_size, my_bet, opp_bet, opp_action_probs, street=1,
                              exploration=exploration, rng=rng)


def discard_action(my_cards, community_cards, remaining_card_pool, dead_cards=None):
    valid_hole = [c for c in my_cards if isinstance(c, int) and c >= 0]
    valid_community = [c for c in community_cards if isinstance(c, int) and c >= 0]
    _refresh_remaining_pool(remaining_card_pool, valid_hole, valid_community, extra_dead=dead_cards)

    best_winrate = 0
    best_indices = (0, 1)
    for combo in combinations(range(len(valid_hole)), 2):
        hand = [valid_hole[i] for i in combo]
        wr = predict_hand_winrate(hand, remaining_card_pool, valid_community)
        if wr > best_winrate:
            best_winrate = wr
            best_indices = combo
    print(f"[discard_action] winrate={best_winrate}")
    action_types = PokerEnv.ActionType
    return action_types.DISCARD.value, 0, best_indices[0], best_indices[1]


def turn_action(my_cards, community_cards, remaining_card_pool, valid_actions, min_raise, max_raise,
                dead_cards=None, pot_size=0, my_bet=0, opp_bet=0, opp_action_probs=None,
                exploration: ExplorationSettings | None = None, rng=None):
    valid_hole = [c for c in my_cards if isinstance(c, int) and c >= 0]
    community = [c for c in community_cards if isinstance(c, int) and c >= 0]
    _refresh_remaining_pool(remaining_card_pool, valid_hole, community, extra_dead=dead_cards)

    raw_winrate = predict_hand_winrate(valid_hole, remaining_card_pool, community)
    opp_raise = max(0, int(opp_bet) - int(my_bet))
    pot_before = max(1, int(pot_size) - opp_raise)
    winrate = adjust_winrate_for_opp_bet(raw_winrate, opp_raise, pot_before, community)
    threat = board_completion_threat(community)
    print(f"[turn_action] winrate={winrate} raw={raw_winrate} opp_raise={opp_raise} threat={threat:.2f}")
    return ev_action_decision(winrate, valid_actions, min_raise, max_raise,
                              pot_size, my_bet, opp_bet, opp_action_probs, street=2, board_threat=threat,
                              exploration=exploration, rng=rng)


def river_action(my_cards, community_cards, remaining_card_pool, valid_actions, min_raise, max_raise,
                 dead_cards=None, pot_size=0, my_bet=0, opp_bet=0, opp_action_probs=None,
                 exploration: ExplorationSettings | None = None, rng=None):
    valid_hole = [c for c in my_cards if isinstance(c, int) and c >= 0]
    community = [c for c in community_cards if isinstance(c, int) and c >= 0]
    _refresh_remaining_pool(remaining_card_pool, valid_hole, community, extra_dead=dead_cards)

    raw_winrate = predict_hand_winrate(valid_hole, remaining_card_pool, community)
    opp_raise = max(0, int(opp_bet) - int(my_bet))
    pot_before = max(1, int(pot_size) - opp_raise)
    winrate = adjust_winrate_for_opp_bet(raw_winrate, opp_raise, pot_before, community)
    threat = board_completion_threat(community)
    print(f"[river_action] winrate={winrate} raw={raw_winrate} opp_raise={opp_raise} threat={threat:.2f}")
    return ev_action_decision(winrate, valid_actions, min_raise, max_raise,
                              pot_size, my_bet, opp_bet, opp_action_probs, street=3, board_threat=threat,
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
