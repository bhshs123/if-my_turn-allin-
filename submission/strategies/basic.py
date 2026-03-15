from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Optional, Set
import random
from treys import Card, Evaluator

# Module-level evaluator singleton (shared across helpers to avoid re-init overhead).
_TREYS_EVAL = Evaluator()

# Pool of cards that have not yet been seen (not in our hand or revealed/discarded).
# This pool should be updated as we observe cards (our own, opponent's discards, etc.).
remaining_card_pool: Set[int] = set(range(27))
WINRATE_TRIALS = 100


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
    trials: int = WINRATE_TRIALS,
) -> int:
    """Estimate win rate of our hand against one random opponent.

    This function treats the current community cards as fixed (if provided),
    and fills up to 5 community cards by drawing from the remaining pool.
    Then it deals 2 cards to one opponent from the remaining deck.

    Each trial is independent: we keep our own cards fixed, and redraw the rest.

    We use the provided WrappedEval evaluator from the gym environment to
    compare poker hands. The function returns the number of trials (out of
    `trials`) that we win outright (ties count as non-wins).
    """

    # Ensure cards are in valid range
    my_cards = [c % 27 for c in my_cards if isinstance(c, int) and c >= 0]
    if community_cards is not None:
        community_cards = [c % 27 for c in community_cards if isinstance(c, int) and c >= 0]
    pool = {c % 27 for c in pool if isinstance(c, int) and c >= 0}

    # Only evaluate valid hold'em shape: 2 hole cards + up to 5 community cards.
    if len(my_cards) != 2:
        return 0
    if community_cards is not None and len(set(community_cards)) != len(community_cards):
        return 0
    if set(my_cards) & set(community_cards or []):
        return 0

    # Need at least 5 community + 2 opponent cards = 7 cards.
    if len(pool) < 7:
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

        # Deal one opponent (2 cards) from remaining available cards.
        if len(available) < 2:
            break
        opp_cards = random.sample(available, 2)
        played_trials += 1
        # Evaluate our hand.
        our_rank = evaluate_best(my_cards + board)
        opp_rank = evaluate_best(opp_cards + board)

        # Lower rank is better
        if our_rank < opp_rank:
            wins += 1

    if played_trials == 0:
        return 0

    return int(wins / played_trials * 100)


def board_completion_threat(community_cards: List[int]) -> float:
    """Return 0.0–1.0 indicating flush/straight/trips threat on the current board.

    Higher values mean the board is more dangerous — the new card on turn/river
    is more likely to have completed a strong hand for a raising opponent.
    """
    valid = [c for c in community_cards if isinstance(c, int) and 0 <= c < 27]
    if len(valid) < 3:
        return 0.0

    # Flush threat: how many cards share the most common suit
    max_suit_count = max(sum(1 for c in valid if c // 9 == s) for s in {c // 9 for c in valid})
    flush_threat = max(0.0, (max_suit_count - 2) / 3.0)  # 0 at ≤2 cards, 1.0 at 5

    # Straight threat: longest consecutive rank run on the board
    ranks_list = [c % 9 for c in valid]
    ranks = sorted(set(ranks_list))
    if 8 in ranks:  # Ace can act as low (rank -1, below 2)
        ranks = [-1] + ranks
    best_run = cur_run = 1
    for i in range(1, len(ranks)):
        cur_run = (cur_run + 1) if ranks[i] == ranks[i - 1] + 1 else 1
        best_run = max(best_run, cur_run)
    straight_threat = max(0.0, (best_run - 2) / 3.0)  # 0 at ≤2, 1.0 at 5

    # Three-of-a-kind threat: board contains a pair → opponent may hold the 3rd card.
    # One pair on board → 0.5; two pairs or trips on board → 1.0 (full house threat too).
    rank_counts = Counter(ranks_list)
    max_rank_count = max(rank_counts.values())
    paired_ranks = sum(1 for cnt in rank_counts.values() if cnt >= 2)
    if max_rank_count >= 3:
        trips_threat = 1.0        # board already has trips → opponent could have quads or full house
    elif paired_ranks >= 2:
        trips_threat = 1.0        # two-paired board → full house danger
    elif paired_ranks == 1:
        trips_threat = 0.5        # one pair → three-of-a-kind danger
    else:
        trips_threat = 0.0

    return min(1.0, max(flush_threat, straight_threat, trips_threat))


def adjust_winrate_for_opp_bet(
    winrate: int,
    opp_street_raise: int,
    pot_before: int,
    community_cards: Optional[List[int]] = None,
) -> int:
    """Adjust our winrate estimate downward based on opponent aggression + board texture.

    When the opponent raises big on turn/river, especially on a connected/suited board,
    that indicates they may have completed a strong hand.  We scale our winrate down
    proportionally to discourage bluff-calling into likely strong holdings.

    Args:
        opp_street_raise: Extra chips the opponent raised this street (beyond the call amount).
        pot_before:       Pot size before the opponent's raise (used for pot-odds scaling).
        community_cards:  Current visible community cards for board texture check.
    """
    if opp_street_raise <= 0:
        return winrate

    # Scale raise relative to pot: a pot-sized raise scores 1.0
    aggression = min(1.0, opp_street_raise / max(1, pot_before))
    danger = board_completion_threat(community_cards or [])

    # Base 20% down at max aggression; board danger adds up to 15% more
    penalty = aggression * (0.20 + 0.15 * danger)
    return max(0, int(winrate * (1.0 - penalty)))


# ---------------------------------------------------------------------------
# Opponent bet model
# ---------------------------------------------------------------------------

def parse_card_str(s: str) -> int:
    """Convert card string like '7d' or 'Ah' to our integer encoding (0-26)."""
    rank_idx = "23456789A".index(s[0].upper())
    suit_idx = "dhs".index(s[1].lower())
    return suit_idx * 9 + rank_idx


def _eval_best_treys_score(hole_ints: List[int], board_ints: List[int]) -> int:
    """Return best treys rank score (lower = better) with the A-as-T high-straight rule."""
    def _to_str(c: int) -> str:
        return "23456789A"[c % 9] + "dhs"[c // 9]

    hand_s = [_to_str(c) for c in hole_ints]
    board_s = [_to_str(c) for c in board_ints]
    hand = [Card.new(c) for c in hand_s]
    board = [Card.new(c) for c in board_s]
    reg = _TREYS_EVAL.evaluate(hand, board)
    alt_h = [Card.new(c.replace("A", "T")) for c in hand_s]
    alt_b = [Card.new(c.replace("A", "T")) for c in board_s]
    alt = _TREYS_EVAL.evaluate(alt_h, alt_b)
    return min(reg, alt)


def opp_hand_strength_at_showdown(
    opp_card_strs: List[str],
    comm_strs: List[str],
) -> Optional[float]:
    """Compute opponent's absolute hand strength (0-100) from showdown card strings.

    Uses the treys rank score mapped to a 0-100 percentile:
      score=1 (royal flush) → 100.0,  score=7462 (worst) → 0.0.
    Returns None if the inputs are invalid or missing.
    """
    try:
        opp_ints = [parse_card_str(s) for s in opp_card_strs]
        comm_ints = [parse_card_str(s) for s in comm_strs]
        if len(opp_ints) != 2 or len(comm_ints) != 5:
            return None
        score = _eval_best_treys_score(opp_ints, comm_ints)
        # treys: 1 (best) … 7462 (worst).  Map to 0-100 where 100 = strongest.
        strength = (1.0 - (score - 1) / 7461.0) * 100.0
        return max(0.0, min(100.0, strength))
    except Exception:
        return None


class OppBetModel:
    """Bayesian bucketed regression: bet_fraction → opponent hand strength (0-100).

    Intuition:
      - We divide bet_fraction [0, 1) into NUM_BUCKETS equal bins.
      - Each bin maintains a running average of observed hand strengths.
      - Prior is a linear ramp (small bet ≈ weak hand, large bet ≈ strong hand).
      - At prediction time we apply isotonic (non-decreasing) smoothing so the
        model always says 'bigger bet ≈ at least as strong' even if some bins are
        noisy from limited data.

    Training samples come from showdowns: `observe(bet_fraction, hand_strength)`.
    Prediction: `predict(bet_fraction)` returns estimated strength 0-100.
    """

    NUM_BUCKETS: int = 10       # 10 bins of 10% width each
    PRIOR_WEIGHT: float = 3.0  # Each bin starts as if it already has 3 observations

    def __init__(self) -> None:
        # Initialise each bucket with a ramp prior:
        #   bucket 0 centre = 5%  → prior strength 5
        #   bucket 9 centre = 95% → prior strength 95
        prior = [(i + 0.5) / self.NUM_BUCKETS * 100.0 for i in range(self.NUM_BUCKETS)]
        self._bucket_sum:   List[float] = [p * self.PRIOR_WEIGHT for p in prior]
        self._bucket_count: List[float] = [self.PRIOR_WEIGHT] * self.NUM_BUCKETS
        self._n_obs: int = 0

    # ------------------------------------------------------------------
    def _bucket_idx(self, bet_fraction: float) -> int:
        return min(self.NUM_BUCKETS - 1, int(bet_fraction * self.NUM_BUCKETS))

    # ------------------------------------------------------------------
    def observe(self, bet_fraction: float, hand_strength: float) -> None:
        """Record a training observation from a completed showdown.

        Args:
            bet_fraction:  opp_peak_bet / MAX_PLAYER_BET  (0.0 – 1.0)
            hand_strength: absolute hand strength 0-100 from opp_hand_strength_at_showdown
        """
        b = self._bucket_idx(bet_fraction)
        self._bucket_sum[b] += hand_strength
        self._bucket_count[b] += 1.0
        self._n_obs += 1

    # ------------------------------------------------------------------
    def predict(self, bet_fraction: float) -> float:
        """Predict opponent's hand strength (0-100) for the given bet fraction.

        Steps:
          1. Compute raw average for each bucket.
          2. Apply isotonic smoothing (enforce non-decreasing order left→right).
          3. Return the smoothed value for bet_fraction's bucket.
        """
        raw = [self._bucket_sum[i] / self._bucket_count[i] for i in range(self.NUM_BUCKETS)]

        # Isotonic smoothing: higher bet should imply at least as strong a hand.
        smoothed = list(raw)
        for i in range(1, self.NUM_BUCKETS):
            if smoothed[i] < smoothed[i - 1]:
                smoothed[i] = smoothed[i - 1]

        return smoothed[self._bucket_idx(bet_fraction)]

    # ------------------------------------------------------------------
    @property
    def n_observations(self) -> int:
        """Total number of showdown observations recorded so far."""
        return self._n_obs


def predict_opp_winrate(opp_bet: int, max_player_bet: int, model: OppBetModel) -> int:
    """Estimate the opponent's hand strength (0-100) from their current bet size.

    Args:
        opp_bet:        Opponent's total bet so far this hand.
        max_player_bet: Maximum possible bet (PokerEnv.MAX_PLAYER_BET = 100).
        model:          The per-session OppBetModel instance.

    Returns:
        Predicted opponent hand strength as an integer percentage 0-100.
    """
    bet_fraction = min(1.0, max(0.0, opp_bet / max(1, max_player_bet)))
    return int(model.predict(bet_fraction))


