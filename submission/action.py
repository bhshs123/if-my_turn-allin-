from __future__ import annotations

import numbers
from itertools import combinations
from typing import Iterable

from gym_env import PokerEnv
from submission.action_ev import ev_action_decision
from submission.action_shared import (
    ExplorationSettings,
    _winrate_trials_for_street,
    exploration_candidates,
    randomized_raise_amount,
    select_action_with_exploration,
)
from submission.strategies.basic import (
    adjust_winrate_for_opp_bet,
    board_completion_threat,
    hand_rank_class,
    predict_hand_winrate,
)


def _filtered_pool(pool: set[int], my_cards: Iterable[int], community_cards: Iterable[int], dead_cards: Iterable[int]) -> set[int]:
    blocked = {int(c) for c in my_cards if isinstance(c, numbers.Integral) and c >= 0}
    blocked.update(int(c) for c in community_cards if isinstance(c, numbers.Integral) and c >= 0)
    blocked.update(int(c) for c in dead_cards if isinstance(c, numbers.Integral) and c >= 0)
    return {int(c) for c in pool if isinstance(c, numbers.Integral) and c >= 0 and c not in blocked}


def _estimate_winrate(
    my_cards: list[int],
    community_cards: list[int],
    card_pool: set[int],
    street: int,
    pot_size: int,
    my_bet: int,
    opp_bet: int,
    board_threat: float,
) -> int:
    trials = _winrate_trials_for_street(street, pot_size, my_bet, opp_bet, board_threat)
    winrate = predict_hand_winrate(
        my_cards,
        card_pool,
        community_cards=community_cards if community_cards else None,
        trials=trials,
    )

    if street >= 2:
        opp_street_raise = max(0, int(opp_bet) - int(my_bet))
        winrate = adjust_winrate_for_opp_bet(
            winrate=winrate,
            opp_street_raise=opp_street_raise,
            pot_before=max(1, int(pot_size)),
            community_cards=community_cards,
            street=street,
        )
    return int(max(0, min(100, winrate)))


def discard_action(
    my_cards: list[int],
    community_cards: list[int],
    card_pool: set[int],
    dead_cards: list[int] | None = None,
) -> tuple[int, int, int, int]:
    """Return (DISCARD, 0, keep_idx_1, keep_idx_2) where indices are positions in my_cards (0-4)."""
    dead_cards = dead_cards or []
    action_types = PokerEnv.ActionType

    # Build index-to-card mapping for valid cards only.
    indexed = [(i, int(c)) for i, c in enumerate(my_cards) if isinstance(c, numbers.Integral) and c >= 0]
    valid_board = [int(c) for c in community_cards if isinstance(c, numbers.Integral) and c >= 0]

    if len(indexed) < 2:
        # Fall back to keeping first two positional slots.
        return action_types.DISCARD.value, 0, 0, 1

    if len(indexed) == 2:
        return action_types.DISCARD.value, 0, indexed[0][0], indexed[1][0]

    all_card_vals = [card for _, card in indexed]
    local_pool = _filtered_pool(card_pool, all_card_vals, valid_board, dead_cards)

    best_indices = (indexed[0][0], indexed[1][0])
    best_score = -1
    for (i1, c1), (i2, c2) in combinations(indexed, 2):
        score = predict_hand_winrate(
            [c1, c2],
            local_pool,
            community_cards=valid_board if valid_board else None,
            trials=140,
        )
        if score > best_score:
            best_score = score
            best_indices = (i1, i2)

    return action_types.DISCARD.value, 0, best_indices[0], best_indices[1]


def preflop_action(
    my_cards: list[int],
    card_pool: set[int],
    valid_actions: list,
    min_raise: int,
    max_raise: int,
    pot_size: int = 0,
    my_bet: int = 0,
    opp_bet: int = 0,
    opp_action_probs: dict | None = None,
    opp_pressure: float = 0.0,
    aggression_scale: float = 1.0,
    action_biases: dict[str, float] | None = None,
    raise_sizing: dict | None = None,
    exploration: ExplorationSettings | None = None,
    rng=None,
) -> tuple[int, int, int, int]:
    usable_cards = [int(c) for c in my_cards if isinstance(c, numbers.Integral) and c >= 0][:2]
    local_pool = _filtered_pool(card_pool, usable_cards, [], [])
    winrate = _estimate_winrate(
        usable_cards,
        [],
        local_pool,
        street=0,
        pot_size=pot_size,
        my_bet=my_bet,
        opp_bet=opp_bet,
        board_threat=0.0,
    )

    return ev_action_decision(
        winrate=winrate,
        valid_actions=valid_actions,
        min_raise=min_raise,
        max_raise=max_raise,
        pot_size=pot_size,
        my_bet=my_bet,
        opp_bet=opp_bet,
        opp_action_probs=opp_action_probs or {},
        street=0,
        board_threat=0.0,
        opp_pressure=opp_pressure,
        hand_class=9,
        aggression_scale=aggression_scale,
        action_biases=action_biases,
        raise_sizing=raise_sizing,
        exploration=exploration,
        rng=rng,
    )


def flop_action(
    my_cards: list[int],
    community_cards: list[int],
    card_pool: set[int],
    valid_actions: list,
    min_raise: int,
    max_raise: int,
    dead_cards: list[int] | None = None,
    pot_size: int = 0,
    my_bet: int = 0,
    opp_bet: int = 0,
    opp_action_probs: dict | None = None,
    opp_pressure: float = 0.0,
    aggression_scale: float = 1.0,
    action_biases: dict[str, float] | None = None,
    raise_sizing: dict | None = None,
    exploration: ExplorationSettings | None = None,
    rng=None,
) -> tuple[int, int, int, int]:
    dead_cards = dead_cards or []
    usable_cards = [int(c) for c in my_cards if isinstance(c, numbers.Integral) and c >= 0][:2]
    board = [int(c) for c in community_cards if isinstance(c, numbers.Integral) and c >= 0]
    local_pool = _filtered_pool(card_pool, usable_cards, board, dead_cards)

    board_threat = board_completion_threat(board)
    winrate = _estimate_winrate(
        usable_cards,
        board,
        local_pool,
        street=1,
        pot_size=pot_size,
        my_bet=my_bet,
        opp_bet=opp_bet,
        board_threat=board_threat,
    )
    hand_class = hand_rank_class(usable_cards, board)

    return ev_action_decision(
        winrate=winrate,
        valid_actions=valid_actions,
        min_raise=min_raise,
        max_raise=max_raise,
        pot_size=pot_size,
        my_bet=my_bet,
        opp_bet=opp_bet,
        opp_action_probs=opp_action_probs or {},
        street=1,
        board_threat=board_threat,
        opp_pressure=opp_pressure,
        hand_class=hand_class,
        aggression_scale=aggression_scale,
        action_biases=action_biases,
        raise_sizing=raise_sizing,
        exploration=exploration,
        rng=rng,
    )


def turn_action(
    my_cards: list[int],
    community_cards: list[int],
    card_pool: set[int],
    valid_actions: list,
    min_raise: int,
    max_raise: int,
    dead_cards: list[int] | None = None,
    pot_size: int = 0,
    my_bet: int = 0,
    opp_bet: int = 0,
    opp_action_probs: dict | None = None,
    opp_pressure: float = 0.0,
    aggression_scale: float = 1.0,
    action_biases: dict[str, float] | None = None,
    raise_sizing: dict | None = None,
    exploration: ExplorationSettings | None = None,
    rng=None,
) -> tuple[int, int, int, int]:
    dead_cards = dead_cards or []
    usable_cards = [int(c) for c in my_cards if isinstance(c, numbers.Integral) and c >= 0][:2]
    board = [int(c) for c in community_cards if isinstance(c, numbers.Integral) and c >= 0]
    local_pool = _filtered_pool(card_pool, usable_cards, board, dead_cards)

    board_threat = board_completion_threat(board)
    winrate = _estimate_winrate(
        usable_cards,
        board,
        local_pool,
        street=2,
        pot_size=pot_size,
        my_bet=my_bet,
        opp_bet=opp_bet,
        board_threat=board_threat,
    )
    hand_class = hand_rank_class(usable_cards, board)

    return ev_action_decision(
        winrate=winrate,
        valid_actions=valid_actions,
        min_raise=min_raise,
        max_raise=max_raise,
        pot_size=pot_size,
        my_bet=my_bet,
        opp_bet=opp_bet,
        opp_action_probs=opp_action_probs or {},
        street=2,
        board_threat=board_threat,
        opp_pressure=opp_pressure,
        hand_class=hand_class,
        aggression_scale=aggression_scale,
        action_biases=action_biases,
        raise_sizing=raise_sizing,
        exploration=exploration,
        rng=rng,
    )


def river_action(
    my_cards: list[int],
    community_cards: list[int],
    card_pool: set[int],
    valid_actions: list,
    min_raise: int,
    max_raise: int,
    dead_cards: list[int] | None = None,
    pot_size: int = 0,
    my_bet: int = 0,
    opp_bet: int = 0,
    opp_action_probs: dict | None = None,
    opp_pressure: float = 0.0,
    aggression_scale: float = 1.0,
    action_biases: dict[str, float] | None = None,
    raise_sizing: dict | None = None,
    exploration: ExplorationSettings | None = None,
    rng=None,
) -> tuple[int, int, int, int]:
    dead_cards = dead_cards or []
    usable_cards = [int(c) for c in my_cards if isinstance(c, numbers.Integral) and c >= 0][:2]
    board = [int(c) for c in community_cards if isinstance(c, numbers.Integral) and c >= 0]
    local_pool = _filtered_pool(card_pool, usable_cards, board, dead_cards)

    board_threat = board_completion_threat(board)
    winrate = _estimate_winrate(
        usable_cards,
        board,
        local_pool,
        street=3,
        pot_size=pot_size,
        my_bet=my_bet,
        opp_bet=opp_bet,
        board_threat=board_threat,
    )
    hand_class = hand_rank_class(usable_cards, board)

    return ev_action_decision(
        winrate=winrate,
        valid_actions=valid_actions,
        min_raise=min_raise,
        max_raise=max_raise,
        pot_size=pot_size,
        my_bet=my_bet,
        opp_bet=opp_bet,
        opp_action_probs=opp_action_probs or {},
        street=3,
        board_threat=board_threat,
        opp_pressure=opp_pressure,
        hand_class=hand_class,
        aggression_scale=aggression_scale,
        action_biases=action_biases,
        raise_sizing=raise_sizing,
        exploration=exploration,
        rng=rng,
    )


__all__ = [
    "ExplorationSettings",
    "ev_action_decision",
    "exploration_candidates",
    "select_action_with_exploration",
    "randomized_raise_amount",
    "discard_action",
    "preflop_action",
    "flop_action",
    "turn_action",
    "river_action",
]
