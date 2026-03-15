from gym_env import PokerEnv
from submission.strategies.basic import predict_hand_winrate
from itertools import combinations


def action_from_winrate(
    winrate: int,
    valid_actions: list[bool],
    min_raise: int,
    max_raise: int,
) -> tuple[int, int, int, int]:
    """Convert a winrate score into a pre-flop action.

    Args:
        winrate: winrate score (0-100) from predict_hand_winrate.
        valid_actions: boolean list of valid actions from environment.
        min_raise: minimum raise amount (from env observation).
        max_raise: maximum raise amount (from env observation).

    Returns:
        Action tuple for the gym environment.
    """

    action_types = PokerEnv.ActionType

    if winrate < 15:
        return action_types.FOLD.value, 0, 0, 0

    if winrate > 90 and valid_actions[action_types.RAISE.value]:
        amt = max(min_raise, int(max_raise * 0.2))
        return action_types.RAISE.value, amt, 0, 0

    if winrate > 70 and valid_actions[action_types.RAISE.value]:
        amt = max(min_raise, int(max_raise * 0.15))
        return action_types.RAISE.value, amt, 0, 0

    if winrate > 50 and valid_actions[action_types.RAISE.value]:
        amt = max(min_raise, int(max_raise * 0.1))
        return action_types.RAISE.value, amt, 0, 0

    if winrate > 15 and valid_actions[action_types.CHECK.value]:
        return action_types.CHECK.value, 0, 0, 0

    if valid_actions[action_types.CALL.value]:
        return action_types.CALL.value, 0, 0, 0

    return action_types.FOLD.value, 0, 0, 0


def preflop_action(my_cards, remaining_card_pool, valid_actions, min_raise, max_raise):
    valid_hole = [c for c in my_cards if c != -1]
    if len(valid_hole) == 2:
        winrate = predict_hand_winrate(valid_hole, remaining_card_pool, [])
    else:
        winrates = []
        for combo in combinations(valid_hole, 2):
            winrate = predict_hand_winrate(list(combo), remaining_card_pool, [])
            winrates.append(winrate)
        winrate = max(winrates) if winrates else 0
    return action_from_winrate(winrate, valid_actions, min_raise, max_raise)


def flop_action(my_cards, community_cards, remaining_card_pool, valid_actions, min_raise, max_raise):
    valid_hole = [c for c in my_cards if c != -1]
    community = [c for c in community_cards if c != -1]
    winrate = predict_hand_winrate(valid_hole, remaining_card_pool, community)
    # Similar logic to preflop, but perhaps adjust thresholds if needed
    return action_from_winrate(winrate, valid_actions, min_raise, max_raise)


def discard_action(my_cards, community_cards, remaining_card_pool):
    # my_cards has 5 cards, community_cards has 3, find best 2 to keep
    best_winrate = 0
    best_indices = (0, 1)  # default
    for combo in combinations(range(5), 2):  # indices 0,1,2,3,4
        hand = [my_cards[i] for i in combo]
        winrate = predict_hand_winrate(hand, remaining_card_pool, community_cards)
        if winrate > best_winrate:
            best_winrate = winrate
            best_indices = combo
    # Action: DISCARD, 0, keep_card_1, keep_card_2
    action_types = PokerEnv.ActionType
    return action_types.DISCARD.value, 0, best_indices[0], best_indices[1]


def turn_action(my_cards, community_cards, remaining_card_pool, valid_actions, min_raise, max_raise):
    valid_hole = [c for c in my_cards if c != -1]
    community = [c for c in community_cards if c != -1]
    winrate = predict_hand_winrate(valid_hole, remaining_card_pool, community)
    return action_from_winrate(winrate, valid_actions, min_raise, max_raise)


def river_action(my_cards, community_cards, remaining_card_pool, valid_actions, min_raise, max_raise):
    valid_hole = [c for c in my_cards if c != -1]
    community = [c for c in community_cards if c != -1]
    winrate = predict_hand_winrate(valid_hole, remaining_card_pool, community)
    return action_from_winrate(winrate, valid_actions, min_raise, max_raise)


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
