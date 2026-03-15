from gym_env import PokerEnv
from submission.strategies.basic import (
    predict_hand_winrate,
    update_pool,
    board_completion_threat,
    adjust_winrate_for_opp_bet,
)
from itertools import combinations


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
    raise_r = max(min_raise, int(max_raise * 0.12))
    chips_in = call_amount + raise_r
    showdown_pot = pot + call_amount + 2 * raise_r
    ev_raise = (
        p_opp_fold * pot
        + p_opp_continues * (p_win * showdown_pot - chips_in)
    )

    # Final raise sizing: scale bet with hand strength
    if winrate > 80:
        raise_r_final = max(min_raise, int(max_raise * 0.20))
    elif winrate > 70:
        raise_r_final = max(min_raise, int(max_raise * 0.15))
    else:
        raise_r_final = max(min_raise, int(max_raise * 0.12))

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

    best = max(evs, key=lambda a: evs[a])

    # Near breakeven calls are sensitive to simulation/model noise.
    # Avoid over-folding: if call EV is only slightly negative, prefer CALL.
    if best == "FOLD" and can_call:
        noise_margin = 0.05 * pot
        if ev_call >= -noise_margin:
            best = "CALL"

    print(
        f"[ev] street={street} winrate={winrate} call={call_amount} pot={pot} "
        f"p_opp_fold={p_opp_fold:.2f} "
        f"EV(fold)={ev_fold:.1f} EV(call/chk)={ev_call:.1f} EV(raise)={ev_raise:.1f} -> {best}"
    )

    if best == "RAISE":
        return action_types.RAISE.value, raise_r_final, 0, 0
    if best == "CALL":
        return action_types.CALL.value, 0, 0, 0
    if best == "CHECK":
        return action_types.CHECK.value, 0, 0, 0
    return action_types.FOLD.value, 0, 0, 0


def preflop_action(my_cards, remaining_card_pool, valid_actions, min_raise, max_raise,
                   pot_size=0, my_bet=0, opp_bet=0, opp_action_probs=None):
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
                              pot_size, my_bet, opp_bet, opp_action_probs, street=0)


def flop_action(my_cards, community_cards, remaining_card_pool, valid_actions, min_raise, max_raise,
                dead_cards=None, pot_size=0, my_bet=0, opp_bet=0, opp_action_probs=None):
    valid_hole = [c for c in my_cards if isinstance(c, int) and c >= 0]
    community = [c for c in community_cards if isinstance(c, int) and c >= 0]
    _refresh_remaining_pool(remaining_card_pool, valid_hole, community, extra_dead=dead_cards)

    winrate = predict_hand_winrate(valid_hole, remaining_card_pool, community)
    print(f"[flop_action] winrate={winrate}")
    return ev_action_decision(winrate, valid_actions, min_raise, max_raise,
                              pot_size, my_bet, opp_bet, opp_action_probs, street=1)


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
                dead_cards=None, pot_size=0, my_bet=0, opp_bet=0, opp_action_probs=None):
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
                              pot_size, my_bet, opp_bet, opp_action_probs, street=2, board_threat=threat)


def river_action(my_cards, community_cards, remaining_card_pool, valid_actions, min_raise, max_raise,
                 dead_cards=None, pot_size=0, my_bet=0, opp_bet=0, opp_action_probs=None):
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
                              pot_size, my_bet, opp_bet, opp_action_probs, street=3, board_threat=threat)


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
