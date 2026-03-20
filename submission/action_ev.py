from gym_env import PokerEnv

from submission.action_shared import (
    ExplorationSettings,
    _base_raise_amount_for_spot,
    _clamp_float,
    _clamp_int,
    _made_hand_raise_bonus,
    debug_log,
    randomized_raise_amount,
    select_action_with_exploration,
)


def _apply_action_biases(evs: dict[str, float], action_biases: dict[str, float] | None) -> dict[str, float]:
    if not action_biases:
        return evs

    adjusted = dict(evs)
    for action_name, bias in action_biases.items():
        if action_name in adjusted:
            adjusted[action_name] += float(bias)
    return adjusted


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
    """EV-based fold/call/raise/check decision (game tree spec, Sections 4-5)."""
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

    ev_fold = 0.0

    if call_amount > 0:
        p_win_call = p_win
        if street >= 1:
            bet_ratio = call_amount / max(1, pot)
            range_discount = 1.0 - min(0.25, bet_ratio * 0.55)
            if street >= 2 and opp_pressure > 0.0:
                range_discount *= 1.0 - min(0.12, 0.10 * opp_pressure)
            p_win_call = p_win * range_discount
        ev_call = p_win_call * (pot + call_amount) - call_amount
    else:
        ev_call = p_win * pot

    p_opp_fold = float(opp_action_probs.get("FOLD", 0.0)) if opp_action_probs else 0.0
    p_opp_raise = float(opp_action_probs.get("RAISE", 0.0)) if opp_action_probs else 0.0
    p_opp_continues = 1.0 - p_opp_fold

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

    evs = {}
    raise_threshold_by_street = {0: 55, 1: 68, 2: 72, 3: 80}
    raise_threshold = raise_threshold_by_street.get(street, 75)

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
        if call_amount == 0:
            if winrate < 54:
                allow_raise = False
            elif winrate < 64 and p_opp_fold < 0.28:
                allow_raise = False
        else:
            if winrate < 60 and p_opp_fold < 0.34:
                allow_raise = False

    if street >= 2 and allow_raise:
        ev_edge = ev_raise - ev_call
        min_edge = 0.08 * pot
        if winrate < 85 and ev_edge < min_edge:
            allow_raise = False
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
        p_win_bluff = p_win_raise * 0.88
        ev_pressure = p_opp_fold * pot + p_opp_continues * (p_win_bluff * bluff_showdown_pot - bluff_chips_in)
        if ev_pressure >= ev_call + 0.03 * pot:
            evs["RAISE"] = ev_pressure
            raise_r_choice_base = bluff_raise

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

    allow_probe_raise = (
        can_raise
        and call_amount == 0
        and street in (1, 2)
        and 46 <= winrate < raise_threshold
        and p_opp_fold >= 0.36
        and opp_pressure <= 0.65
    )
    if allow_probe_raise and street == 1:
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
        evs["CHECK"] = ev_call
    if can_fold:
        evs["FOLD"] = ev_fold

    min_call_winrate = None
    if call_amount > 0 and street >= 2 and "CALL" in evs:
        pot_odds_pct = (100.0 * call_amount) / max(1.0, (pot + call_amount))
        bet_ratio = call_amount / max(1.0, pot)
        if street == 2:
            safety_buffer = 13 + min(16.0, 16.0 * bet_ratio)
        else:
            safety_buffer = 18 + min(24.0, 24.0 * bet_ratio)
        safety_buffer += max(0.0, min(1.0, float(board_threat))) * 12.0
        safety_buffer += max(0.0, (p_opp_raise - 0.35)) * 24.0
        safety_buffer += max(0.0, min(1.0, float(opp_pressure))) * 10.0

        if hand_class >= 8:
            if street == 2:
                safety_buffer += 8.0 + 10.0 * bet_ratio
            else:
                safety_buffer += 14.0 + 16.0 * bet_ratio

        min_call_winrate = pot_odds_pct + safety_buffer
        if street == 3 and opp_bet >= int(max_raise * 0.8):
            min_call_winrate = max(min_call_winrate, 72.0)
        elif street == 3 and opp_bet >= int(max_raise * 0.5):
            min_call_winrate = max(min_call_winrate, 66.0)
        elif street == 3 and opp_bet > 0:
            min_call_winrate = max(min_call_winrate, 62.0)

        if hand_class >= 8:
            if street == 2 and (bet_ratio >= 0.40 or opp_pressure >= 0.50):
                min_call_winrate = max(min_call_winrate, 56.0)
            if street == 3 and (bet_ratio >= 0.25 or opp_pressure >= 0.35):
                min_call_winrate = max(min_call_winrate, 66.0)

        if winrate < min_call_winrate:
            evs.pop("CALL", None)

    if not evs:
        return action_types.FOLD.value, 0, 0, 0

    if can_check and "FOLD" in evs:
        evs.pop("FOLD")

    strong_call_bias_guard = (
        call_amount > 0
        and street >= 2
        and "CALL" in evs
        and "FOLD" in evs
        and ev_call >= max(1.0, 0.10 * pot)
        and (min_call_winrate is None or winrate >= (min_call_winrate + 10.0))
    )

    evs = _apply_action_biases(evs, action_biases)

    best, candidates = select_action_with_exploration(
        evs,
        pot=pot,
        exploration=exploration,
        rng=rng,
    )

    # Anti-predict biases are only meant to shape marginal turn/river calls.
    # Do not let them turn a clearly profitable bluff-catcher into a fold.
    if strong_call_bias_guard and best == "FOLD":
        best = "CALL"

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

    debug_log(
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
