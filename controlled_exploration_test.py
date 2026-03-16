import math

from gym_env import PokerEnv
from submission.action import (
    ExplorationSettings,
    ev_action_decision,
    exploration_candidates,
    randomized_raise_amount,
    select_action_with_exploration,
)
from submission.player import RecentPerformanceTracker


class FixedRng:
    def __init__(self, random_values=None, randint_values=None):
        self._random_values = list(random_values or [0.0])
        self._randint_values = list(randint_values or [])

    def random(self):
        if not self._random_values:
            return 0.0
        return self._random_values.pop(0)

    def randint(self, low, high):
        if not self._randint_values:
            return low
        value = self._randint_values.pop(0)
        return max(low, min(high, value))


def test_exploration_candidates_only_keep_near_ev_actions():
    settings = ExplorationSettings(mix_probability=0.10, ev_margin_pct=0.04, ev_margin_floor=1.0)
    evs = {"RAISE": 10.0, "CALL": 8.8, "CHECK": 5.5, "FOLD": 0.0}

    candidates = exploration_candidates(evs, pot=40, exploration=settings)

    assert candidates == [("RAISE", 10.0), ("CALL", 8.8)]


def test_select_action_with_exploration_never_overrides_best_fold():
    settings = ExplorationSettings(mix_probability=1.0, ev_margin_pct=0.10, ev_margin_floor=5.0)
    rng = FixedRng(random_values=[0.0, 0.99])

    action, candidates = select_action_with_exploration(
        {"FOLD": 0.0, "CALL": -2.0},
        pot=25,
        exploration=settings,
        rng=rng,
    )

    assert action == "FOLD"
    assert candidates == [("FOLD", 0.0)]


def test_ev_action_decision_never_turns_free_check_into_fold():
    valid_actions = [1, 0, 1, 0, 0]
    action = ev_action_decision(
        winrate=35,
        valid_actions=valid_actions,
        min_raise=2,
        max_raise=10,
        pot_size=20,
        my_bet=0,
        opp_bet=0,
        opp_action_probs={"FOLD": 0.1},
        street=1,
        exploration=ExplorationSettings(mix_probability=1.0),
        rng=FixedRng(random_values=[0.0, 0.99]),
    )

    assert action[0] == PokerEnv.ActionType.CHECK.value


def test_randomized_raise_amount_stays_within_bounds():
    amount = randomized_raise_amount(
        base_raise=20,
        min_raise=10,
        max_raise=22,
        exploration=ExplorationSettings(raise_jitter_pct=0.20),
        rng=FixedRng(randint_values=[99]),
    )

    assert amount == 22


def test_recent_performance_tracker_triggers_decay_and_cooldown():
    tracker = RecentPerformanceTracker(
        recent_window=5,
        min_baseline_hands=8,
        explore_duration=3,
        cooldown_hands=2,
        significance_z=0.0,
        min_mean_drop=5.0,
    )

    for _ in range(8):
        assert tracker.record_hand_result(20.0) is None

    event = None
    for _ in range(5):
        event = tracker.record_hand_result(-20.0)
        if event == "triggered":
            break

    assert event == "triggered"
    assert tracker.snapshot()["active"] is True
    assert math.isclose(tracker.current_mix_probability(), 0.10, rel_tol=1e-9, abs_tol=1e-9)

    tracker.record_hand_result(-20.0)
    mix_mid = tracker.current_mix_probability()
    tracker.record_hand_result(-20.0)
    mix_late = tracker.current_mix_probability()
    end_event = tracker.record_hand_result(-20.0)

    assert mix_mid < 0.10
    assert mix_late < mix_mid
    assert mix_late >= 0.05
    assert end_event == "ended"
    assert tracker.cooldown_hands_left == 2

    tracker.record_hand_result(-20.0)
    tracker.record_hand_result(-20.0)
    assert tracker.exploration_hands_left == 0
    assert tracker.cooldown_hands_left == 0
