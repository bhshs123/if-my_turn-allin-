from __future__ import annotations

from typing import Iterable, List, Optional, Set
import random

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
    trials: int = 500,
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

    # Need at least 5 community + 6 opponent cards = 11 cards.
    if len(pool) < 11:
        return 0

    from gym import WrappedEval

    evaluator = WrappedEval()

    def evaluate_best(cards: List[int]) -> int:
        # WrappedEval expects exactly 7 cards: 2 hole + 5 board.
        # We pass hole+board and let it evaluate.
        return evaluator.evaluate(cards[:2], cards[2:])

    wins = 0
    for _ in range(trials):
        # Build community board (5 cards): use known ones, then fill randomly.
        board = []
        if community_cards is not None:
            board = [c for c in community_cards if isinstance(c, int) and c >= 0]
        # Fill to 5 cards
        available = [c for c in pool if c not in board]
        if len(board) < 5:
            needed = 5 - len(board)
            board += random.sample(available, needed)
            available = [c for c in available if c not in board]

        # Deal 3 opponents (2 cards each) from remaining available cards.
        if len(available) < 6:
            break
        opp_cards = random.sample(available, 6)
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

    return int(wins / trials * 100)


