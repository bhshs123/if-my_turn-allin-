from __future__ import annotations

from typing import Iterable, List, Optional, Set
import random
from treys import Card, Evaluator

# Pool of cards that have not yet been seen (not in our hand or revealed/discarded).
# This pool should be updated as we observe cards (our own, opponent's discards, etc.).
remaining_card_pool: Set[int] = set(range(27))


def reset_pool() -> None:
    """Reset the card pool to contain all cards (0-26)."""
    global remaining_card_pool
    remaining_card_pool = set(range(27))


def update_pool(seen_cards: Iterable[int]) -> None:
    """Remove seen cards from the remaining card pool.

    Args:
        seen_cards: List of card integers observed (e.g., your hole cards, discarded cards).
    """

    global remaining_card_pool
    for c in seen_cards:
        if c is None:
            continue
        if isinstance(c, int) and 0 <= c < 27:
            remaining_card_pool.discard(c)


def predict_hand_winrate(
    my_cards: List[int],
    pool: Set[int],
    community_cards: Optional[List[int]] = None,
    trials: int = 50,
) -> int:
    """Estimate win rate of our hand against three random opponents.

    This function treats the current community cards as fixed (if provided),
    and fills up to 5 community cards by drawing from the remaining pool.
    Then it deals 2 cards each to three opponents from the remaining deck.

    Each trial is independent: we keep our own cards fixed, and redraw the rest.

    We use the provided WrappedEval evaluator from the gym environment to
    compare poker hands. The function returns the number of trials (out of
    `trials`) that we win outright (ties count as non-wins).
    """

    # Ensure cards are in valid range
    my_cards = [c % 27 for c in my_cards if isinstance(c, int)]
    if community_cards is not None:
        community_cards = [c % 27 for c in community_cards if isinstance(c, int)]
    pool = {c % 27 for c in pool if isinstance(c, int)}

    # Only evaluate valid hold'em shape: 2 hole cards + up to 5 community cards.
    if len(my_cards) != 2:
        return 0
    if community_cards is not None and len(set(community_cards)) != len(community_cards):
        return 0
    if set(my_cards) & set(community_cards or []):
        return 0

    # Need at least 5 community + 6 opponent cards = 11 cards.
    if len(pool) < 11:
        return 0

    evaluator = Evaluator()

    def evaluate_best(cards: List[int]) -> int:
        def int_to_card(card_int: int) -> str:
            ranks = "23456789A"
            suits = "dhs"
            normalized = card_int % 27
            rank_index = normalized % 9
            suit_index = normalized // 9
            return f"{ranks[rank_index]}{suits[suit_index]}"

        hand_str = [int_to_card(c) for c in cards[:2]]
        board_str = [int_to_card(c) for c in cards[2:]]

        hand = [Card.new(c) for c in hand_str]
        board = [Card.new(c) for c in board_str]
        reg_score = evaluator.evaluate(hand, board)

        # Tournament special rule: Ace can also be high in 6-7-8-9-A.
        # We simulate that by treating A as T and taking the better score.
        alt_hand = [Card.new(c.replace("A", "T")) for c in hand_str]
        alt_board = [Card.new(c.replace("A", "T")) for c in board_str]
        alt_score = evaluator.evaluate(alt_hand, alt_board)

        return min(reg_score, alt_score)

    wins = 0
    played_trials = 0
    dead_cards = set(my_cards) | set(community_cards or [])

    if len(dead_cards) != len(my_cards) + len(community_cards or []):
        return 0

    for _ in range(trials):
        # Build community board (5 cards): use known ones, then fill randomly.
        board = []
        if community_cards is not None:
            board = [c for c in community_cards]
        # Fill to 5 cards
        available = [c for c in pool if c not in dead_cards and c not in board]
        if len(board) < 5:
            needed = 5 - len(board)
            if len(available) < needed:
                break
            board += random.sample(available, needed)
            available = [c for c in available if c not in board]

        # Deal 3 opponents (2 cards each) from remaining available cards.
        if len(available) < 6:
            break
        opp_cards = random.sample(available, 6)
        played_trials += 1
        # Evaluate our hand.
        our_rank = evaluate_best(my_cards + board)

        best_opponent_rank = float("inf")
        for i in range(3):
            opp_hand = opp_cards[2 * i : 2 * i + 2]
            opp_rank = evaluate_best(opp_hand + board)
            if opp_rank < best_opponent_rank:
                best_opponent_rank = opp_rank

        # Lower rank is better
        if our_rank < best_opponent_rank:
            wins += 1

    if played_trials == 0:
        return 0

    return int(wins / played_trials * 100)


