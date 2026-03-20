import importlib.util
import math
import sys
import types
from collections import deque


def _install_optional_dependency_stubs() -> None:
    if importlib.util.find_spec("numpy") is None:
        sys.modules.setdefault("numpy", types.ModuleType("numpy"))

    if importlib.util.find_spec("gym") is None:
        gym = types.ModuleType("gym")

        class Env:
            pass

        class _Space:
            def __init__(self, *args, **kwargs):
                pass

        gym.Env = Env
        gym.spaces = types.SimpleNamespace(
            Tuple=_Space,
            Discrete=_Space,
            Dict=_Space,
            Box=_Space,
        )
        sys.modules.setdefault("gym", gym)

    if importlib.util.find_spec("treys") is None:
        treys = types.ModuleType("treys")

        class Card:
            @staticmethod
            def new(card_str):
                return 0

            @staticmethod
            def int_to_str(card_int):
                return "2d"

        class Evaluator:
            def evaluate(self, hand, board):
                return 0

        treys.Card = Card
        treys.Evaluator = Evaluator
        sys.modules.setdefault("treys", treys)

    if importlib.util.find_spec("fastapi") is None:
        fastapi = types.ModuleType("fastapi")

        class FastAPI:
            def get(self, *args, **kwargs):
                return lambda func: func

            def post(self, *args, **kwargs):
                return lambda func: func

        class HTTPException(Exception):
            def __init__(self, status_code=None, detail=None):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        fastapi.FastAPI = FastAPI
        fastapi.HTTPException = HTTPException
        sys.modules.setdefault("fastapi", fastapi)

    if importlib.util.find_spec("pydantic") is None:
        pydantic = types.ModuleType("pydantic")

        class BaseModel:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        pydantic.BaseModel = BaseModel
        sys.modules.setdefault("pydantic", pydantic)

    if importlib.util.find_spec("uvicorn") is None:
        uvicorn = types.ModuleType("uvicorn")
        uvicorn.run = lambda *args, **kwargs: None
        sys.modules.setdefault("uvicorn", uvicorn)


_install_optional_dependency_stubs()

import pytest

from gym_env import PokerEnv
import submission.action as action_module
import submission.action_ev as action_ev_module
from submission.action import ExplorationSettings, ev_action_decision
from submission.action_shared import select_action_with_exploration
from submission.anti_predict import AntiPredictTracker
from submission.exploration import RecentPerformanceTracker
import submission.player as player_module
from submission.player import PlayerAgent


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


def _valid_actions(*legal_action_names: str) -> list[int]:
    valid_actions = [0] * len(PokerEnv.ActionType)
    for action_name in legal_action_names:
        valid_actions[PokerEnv.ActionType[action_name].value] = 1
    return valid_actions


def _action_name(action_tuple) -> str:
    return PokerEnv.ActionType(action_tuple[0]).name


def _make_player_agent() -> PlayerAgent:
    agent = PlayerAgent.__new__(PlayerAgent)
    agent.action_types = PokerEnv.ActionType
    agent._dbbr = types.SimpleNamespace(config=types.SimpleNamespace(warmup_iters=0))
    agent._hand_counter = 999
    return agent


def _call_spot_kwargs(
    *,
    street: int,
    winrate: int,
    hand_class: int,
    bet_ratio: float,
    opp_pressure: float,
    action_biases: dict[str, float] | None = None,
    exploration: ExplorationSettings | None = None,
) -> dict:
    pot_size = 100
    call_amount = int(round(pot_size * bet_ratio))
    opp_action_probs = {"FOLD": 0.05, "RAISE": 0.35, "CALL": 0.60}
    return {
        "winrate": winrate,
        "valid_actions": _valid_actions("FOLD", "CALL"),
        "min_raise": 2,
        "max_raise": 100,
        "pot_size": pot_size,
        "my_bet": 0,
        "opp_bet": call_amount,
        "opp_action_probs": opp_action_probs,
        "street": street,
        "board_threat": 0.0,
        "opp_pressure": opp_pressure,
        "hand_class": hand_class,
        "action_biases": action_biases,
        "exploration": exploration or ExplorationSettings(mix_probability=0.0),
    }


def _explicit_call_spot_kwargs(
    *,
    street: int,
    winrate: int,
    hand_class: int,
    pot_size: int,
    opp_bet: int,
    opp_pressure: float,
    max_raise: int,
    action_biases: dict[str, float] | None = None,
    exploration: ExplorationSettings | None = None,
) -> dict:
    return {
        "winrate": winrate,
        "valid_actions": _valid_actions("FOLD", "CALL"),
        "min_raise": 2,
        "max_raise": max_raise,
        "pot_size": pot_size,
        "my_bet": 0,
        "opp_bet": opp_bet,
        "opp_action_probs": {"FOLD": 0.05, "RAISE": 0.35, "CALL": 0.60},
        "street": street,
        "board_threat": 0.0,
        "opp_pressure": opp_pressure,
        "hand_class": hand_class,
        "action_biases": action_biases,
        "exploration": exploration or ExplorationSettings(mix_probability=0.0),
    }


def _expected_min_call_winrate(
    *,
    street: int,
    hand_class: int,
    bet_ratio: float,
    opp_pressure: float,
    max_raise: int = 100,
) -> float:
    pot = 100.0
    call_amount = pot * bet_ratio
    opp_bet = call_amount
    p_opp_raise = 0.35
    pot_odds_pct = (100.0 * call_amount) / max(1.0, (pot + call_amount))

    if street == 2:
        safety_buffer = 13.0 + min(16.0, 16.0 * bet_ratio)
    else:
        safety_buffer = 18.0 + min(24.0, 24.0 * bet_ratio)
    safety_buffer += max(0.0, min(1.0, float(opp_pressure))) * 10.0
    safety_buffer += max(0.0, (p_opp_raise - 0.35)) * 24.0

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

    return min_call_winrate


def _expected_min_call_winrate_explicit(
    *,
    street: int,
    hand_class: int,
    pot_size: int,
    opp_bet: int,
    opp_pressure: float,
    max_raise: int,
) -> float:
    return _expected_min_call_winrate(
        street=street,
        hand_class=hand_class,
        bet_ratio=opp_bet / max(1.0, float(pot_size)),
        opp_pressure=opp_pressure,
        max_raise=max_raise,
    )


def _smallest_calling_winrate(*, street: int, hand_class: int, bet_ratio: float, opp_pressure: float):
    for winrate in range(101):
        action = ev_action_decision(
            **_call_spot_kwargs(
                street=street,
                winrate=winrate,
                hand_class=hand_class,
                bet_ratio=bet_ratio,
                opp_pressure=opp_pressure,
            )
        )
        if _action_name(action) == "CALL":
            return winrate
    return None


def _smallest_calling_winrate_explicit(
    *,
    street: int,
    hand_class: int,
    pot_size: int,
    opp_bet: int,
    opp_pressure: float,
    max_raise: int,
):
    for winrate in range(101):
        action = ev_action_decision(
            **_explicit_call_spot_kwargs(
                street=street,
                winrate=winrate,
                hand_class=hand_class,
                pot_size=pot_size,
                opp_bet=opp_bet,
                opp_pressure=opp_pressure,
                max_raise=max_raise,
            )
        )
        if _action_name(action) == "CALL":
            return winrate
    return None


def _call_actions_in_range(
    *,
    street: int,
    hand_class: int,
    bet_ratio: float,
    opp_pressure: float,
    winrates: range,
    action_biases: dict[str, float] | None = None,
) -> dict[int, str]:
    actions = {}
    for winrate in winrates:
        action = ev_action_decision(
            **_call_spot_kwargs(
                street=street,
                winrate=winrate,
                hand_class=hand_class,
                bet_ratio=bet_ratio,
                opp_pressure=opp_pressure,
                action_biases=action_biases,
            )
        )
        actions[winrate] = _action_name(action)
    return actions


def _make_observation(
    *,
    street: int,
    legal_action_names: tuple[str, ...],
    my_bet: int = 0,
    opp_bet: int = 0,
    pot_size: int = 0,
    opp_last_action: str = "None",
    blind_position: int = 1,
) -> dict:
    board_by_street = {
        0: [],
        1: [10, 11, 12],
        2: [10, 11, 12, 13],
        3: [10, 11, 12, 13, 14],
    }
    return {
        "street": street,
        "acting_agent": 0,
        "my_cards": [0, 1, 2, 3, 4],
        "community_cards": board_by_street.get(street, []),
        "my_bet": my_bet,
        "my_discarded_cards": [],
        "opp_bet": opp_bet,
        "opp_discarded_cards": [],
        "min_raise": 2,
        "max_raise": 100,
        "valid_actions": _valid_actions(*legal_action_names),
        "time_used": 0.0,
        "time_left": 100.0,
        "opp_last_action": opp_last_action,
        "blind_position": blind_position,
        "pot_size": pot_size,
    }


def _run_player_act_case(
    monkeypatch,
    *,
    street: int,
    legal_action_names: tuple[str, ...],
    base_action: tuple[int, int, int, int],
    anti_context: dict | None,
    dbbr_action: str | None,
    my_bet: int,
    opp_bet: int,
    pot_size: int,
    opp_probs: dict[str, float] | None = None,
    opp_pressure: float = 0.4,
    use_real_anti_predict: bool = False,
):
    agent = PlayerAgent(stream=False)
    agent._hand_counter = 999
    captured = {}
    recorded = {}

    def fake_street_action(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return base_action

    street_attr = {
        0: "preflop_action",
        1: "flop_action",
        2: "turn_action",
        3: "river_action",
    }[street]
    monkeypatch.setattr(player_module, street_attr, fake_street_action)
    monkeypatch.setattr(agent, "_update_from_opp_last_action", lambda observation: None)
    monkeypatch.setattr(agent, "_maybe_rotate_style", lambda hand_number: None)
    monkeypatch.setattr(agent, "_opp_action_probs", lambda observation, legal: opp_probs or {"FOLD": 0.35, "CALL": 0.65})
    monkeypatch.setattr(agent, "_opp_pressure", lambda observation: opp_pressure)
    monkeypatch.setattr(agent, "_current_aggression_scale", lambda pressure: 1.0)
    monkeypatch.setattr(agent, "_current_exploration_settings", lambda: ExplorationSettings(mix_probability=0.0))
    monkeypatch.setattr(agent._exploration, "snapshot", lambda: {"active": False, "mix_probability": 0.0})
    if not use_real_anti_predict:
        monkeypatch.setattr(agent._anti_predict, "build_context", lambda observation, legal, pot: anti_context)
    monkeypatch.setattr(agent._anti_predict, "record_action", lambda node, action_name: recorded.update({"node": node, "action_name": action_name}))
    monkeypatch.setattr(agent._dbbr, "select_action", lambda observation, legal: dbbr_action)

    observation = _make_observation(
        street=street,
        legal_action_names=legal_action_names,
        my_bet=my_bet,
        opp_bet=opp_bet,
        pot_size=pot_size,
        opp_last_action="RAISE" if opp_bet > my_bet else "CHECK",
        blind_position=0,
    )
    action = agent.act(observation, 0.0, False, False, {"hand_number": 17})
    return action, captured, recorded, agent, observation


def _seed_history(tracker: AntiPredictTracker, node_key: str, action_counts: dict[str, int]) -> None:
    for action_name, count in action_counts.items():
        for _ in range(count):
            tracker.record_action(node_key, action_name)


def _capture_evs_before_selection(monkeypatch):
    captured = {}

    def fake_select_action_with_exploration(evs, pot, exploration=None, rng=None):
        captured["evs"] = dict(evs)
        ranked = sorted(evs.items(), key=lambda item: item[1], reverse=True)
        return ranked[0][0], ranked

    monkeypatch.setattr(action_ev_module, "select_action_with_exploration", fake_select_action_with_exploration)
    return captured


def test_ev_decision_never_returns_fold_when_check_is_legal():
    action = ev_action_decision(
        winrate=35,
        valid_actions=_valid_actions("FOLD", "CHECK"),
        min_raise=2,
        max_raise=20,
        pot_size=20,
        my_bet=0,
        opp_bet=0,
        opp_action_probs={"FOLD": 1.0},
        street=2,
        action_biases={"CHECK": -100.0, "FOLD": 100.0},
        exploration=ExplorationSettings(mix_probability=0.0),
    )

    assert action[0] == PokerEnv.ActionType.CHECK.value


def test_dbbr_override_does_not_turn_legal_check_into_fold():
    agent = _make_player_agent()
    base_action = (PokerEnv.ActionType.CHECK.value, 0, 0, 0)

    final_action = agent._apply_action_override(
        base_action=base_action,
        chosen_action_name="FOLD",
        min_raise=2,
        max_raise=20,
        observation={"valid_actions": _valid_actions("FOLD", "CHECK")},
    )

    assert final_action == base_action


def test_dbbr_override_does_not_turn_legal_call_into_fold():
    agent = _make_player_agent()
    base_action = (PokerEnv.ActionType.CALL.value, 0, 0, 0)

    final_action = agent._apply_action_override(
        base_action=base_action,
        chosen_action_name="FOLD",
        min_raise=2,
        max_raise=100,
        observation={"valid_actions": _valid_actions("FOLD", "CALL")},
    )

    assert final_action == base_action


@pytest.mark.parametrize(
    ("street", "winrate"),
    [
        (2, 90),
        (3, 95),
    ],
)
def test_antipredict_biases_do_not_force_fold_in_strong_turn_or_river_calls(street, winrate):
    action = ev_action_decision(
        winrate=winrate,
        valid_actions=_valid_actions("FOLD", "CALL"),
        min_raise=2,
        max_raise=100,
        pot_size=40,
        my_bet=20,
        opp_bet=30,
        opp_action_probs={"FOLD": 0.05, "RAISE": 0.10, "CALL": 0.85},
        street=street,
        board_threat=0.0,
        opp_pressure=0.0,
        hand_class=2,
        action_biases={"CALL": -100.0, "FOLD": 100.0},
        exploration=ExplorationSettings(mix_probability=0.0),
    )

    assert action[0] == PokerEnv.ActionType.CALL.value


@pytest.mark.parametrize(
    ("street", "hand_class", "bet_ratio", "opp_pressure"),
    [
        (2, 2, 0.25, 0.30),
        (2, 8, 0.25, 0.30),
        (2, 2, 0.50, 0.50),
        (2, 8, 0.50, 0.50),
        (3, 2, 0.25, 0.30),
        (3, 8, 0.25, 0.30),
        (3, 2, 0.50, 0.50),
        (3, 8, 0.50, 0.50),
    ],
)
def test_call_pruning_boundary_matches_smallest_calling_winrate(street, hand_class, bet_ratio, opp_pressure):
    threshold = _smallest_calling_winrate(
        street=street,
        hand_class=hand_class,
        bet_ratio=bet_ratio,
        opp_pressure=opp_pressure,
    )
    min_call_winrate = _expected_min_call_winrate(
        street=street,
        hand_class=hand_class,
        bet_ratio=bet_ratio,
        opp_pressure=opp_pressure,
    )
    expected_floor = math.ceil(min_call_winrate)

    assert threshold == expected_floor, (
        f"street={street} hand_class={hand_class} bet_ratio={bet_ratio} "
        f"opp_pressure={opp_pressure} expected_min_call_winrate={min_call_winrate:.2f} "
        f"threshold={threshold}"
    )

    below = ev_action_decision(
        **_call_spot_kwargs(
            street=street,
            winrate=max(0, threshold - 1),
            hand_class=hand_class,
            bet_ratio=bet_ratio,
            opp_pressure=opp_pressure,
        )
    )
    at_threshold = ev_action_decision(
        **_call_spot_kwargs(
            street=street,
            winrate=threshold,
            hand_class=hand_class,
            bet_ratio=bet_ratio,
            opp_pressure=opp_pressure,
        )
    )

    assert _action_name(below) != "CALL"
    assert _action_name(at_threshold) == "CALL"


@pytest.mark.parametrize(
    ("street", "bet_ratio", "opp_pressure"),
    [
        (2, 0.25, 0.30),
        (2, 0.50, 0.50),
        (3, 0.25, 0.30),
        (3, 0.50, 0.50),
    ],
)
def test_weaker_hand_class_requires_at_least_as_much_winrate_to_call(street, bet_ratio, opp_pressure):
    strong_threshold = _smallest_calling_winrate(
        street=street,
        hand_class=2,
        bet_ratio=bet_ratio,
        opp_pressure=opp_pressure,
    )
    weak_threshold = _smallest_calling_winrate(
        street=street,
        hand_class=8,
        bet_ratio=bet_ratio,
        opp_pressure=opp_pressure,
    )

    assert weak_threshold >= strong_threshold


@pytest.mark.parametrize("street", [2, 3])
@pytest.mark.parametrize("hand_class", [2, 8])
def test_call_threshold_rises_with_bet_ratio_and_opp_pressure(street, hand_class):
    low = _smallest_calling_winrate(street=street, hand_class=hand_class, bet_ratio=0.25, opp_pressure=0.30)
    mid = _smallest_calling_winrate(street=street, hand_class=hand_class, bet_ratio=0.50, opp_pressure=0.50)
    high = _smallest_calling_winrate(street=street, hand_class=hand_class, bet_ratio=0.80, opp_pressure=0.70)

    assert low <= mid
    if high is None:
        assert street == 3 and hand_class == 8
    else:
        assert mid <= high


@pytest.mark.parametrize(
    ("label", "street", "hand_class", "opp_pressure", "opp_bets", "pot_size", "max_raise"),
    [
        ("river_bet_ratio_0_25", 3, 8, 0.30, (249, 250, 251), 1000, 1000),
        ("turn_bet_ratio_0_40", 2, 8, 0.30, (399, 400, 401), 1000, 1000),
        ("turn_bet_ratio_0_50", 2, 2, 0.30, (499, 500, 501), 1000, 1000),
        ("turn_bet_ratio_0_80", 2, 2, 0.30, (799, 800, 801), 1000, 1000),
    ],
)
def test_exact_bet_ratio_threshold_boundaries(label, street, hand_class, opp_pressure, opp_bets, pot_size, max_raise):
    thresholds = []
    for opp_bet in opp_bets:
        expected = _expected_min_call_winrate_explicit(
            street=street,
            hand_class=hand_class,
            pot_size=pot_size,
            opp_bet=opp_bet,
            opp_pressure=opp_pressure,
            max_raise=max_raise,
        )
        observed = _smallest_calling_winrate_explicit(
            street=street,
            hand_class=hand_class,
            pot_size=pot_size,
            opp_bet=opp_bet,
            opp_pressure=opp_pressure,
            max_raise=max_raise,
        )
        thresholds.append((opp_bet, expected, observed))
        assert observed == math.ceil(expected), (
            f"{label} opp_bet={opp_bet} expected_min_call_winrate={expected:.4f} "
            f"observed_threshold={observed}"
        )

    assert thresholds[0][2] <= thresholds[1][2] <= thresholds[2][2], thresholds


@pytest.mark.parametrize(
    ("label", "hand_class", "opp_pressure", "opp_bets"),
    [
        ("river_opp_bet_half_max_raise", 2, 0.30, (49, 50, 51)),
        ("river_opp_bet_eighty_pct_max_raise", 2, 0.30, (79, 80, 81)),
    ],
)
def test_exact_opp_bet_threshold_boundaries(label, hand_class, opp_pressure, opp_bets):
    thresholds = []
    for opp_bet in opp_bets:
        expected = _expected_min_call_winrate_explicit(
            street=3,
            hand_class=hand_class,
            pot_size=100,
            opp_bet=opp_bet,
            opp_pressure=opp_pressure,
            max_raise=100,
        )
        observed = _smallest_calling_winrate_explicit(
            street=3,
            hand_class=hand_class,
            pot_size=100,
            opp_bet=opp_bet,
            opp_pressure=opp_pressure,
            max_raise=100,
        )
        thresholds.append((opp_bet, expected, observed))
        assert observed == math.ceil(expected), (
            f"{label} opp_bet={opp_bet} expected_min_call_winrate={expected:.4f} "
            f"observed_threshold={observed}"
        )

    assert thresholds[0][2] <= thresholds[1][2] <= thresholds[2][2], thresholds


@pytest.mark.parametrize(
    ("label", "street", "hand_class", "opp_bet", "pot_size", "max_raise", "pressures"),
    [
        ("river_pressure_0_35", 3, 8, 250, 1000, 1000, (0.349, 0.350, 0.351)),
        ("turn_pressure_0_50", 2, 8, 400, 1000, 1000, (0.499, 0.500, 0.501)),
        ("river_pressure_0_70", 3, 7, 250, 1000, 1000, (0.699, 0.700, 0.701)),
    ],
)
def test_exact_opp_pressure_threshold_boundaries(label, street, hand_class, opp_bet, pot_size, max_raise, pressures):
    thresholds = []
    for opp_pressure in pressures:
        expected = _expected_min_call_winrate_explicit(
            street=street,
            hand_class=hand_class,
            pot_size=pot_size,
            opp_bet=opp_bet,
            opp_pressure=opp_pressure,
            max_raise=max_raise,
        )
        observed = _smallest_calling_winrate_explicit(
            street=street,
            hand_class=hand_class,
            pot_size=pot_size,
            opp_bet=opp_bet,
            opp_pressure=opp_pressure,
            max_raise=max_raise,
        )
        thresholds.append((opp_pressure, expected, observed))
        assert observed == math.ceil(expected), (
            f"{label} opp_pressure={opp_pressure} expected_min_call_winrate={expected:.4f} "
            f"observed_threshold={observed}"
        )

    assert thresholds[0][2] <= thresholds[1][2] <= thresholds[2][2], thresholds


def test_ev_action_decision_returns_fold_when_only_fold_is_legal():
    action = ev_action_decision(
        winrate=75,
        valid_actions=_valid_actions("FOLD"),
        min_raise=2,
        max_raise=100,
        pot_size=40,
        my_bet=0,
        opp_bet=0,
        opp_action_probs={},
        street=2,
        exploration=ExplorationSettings(mix_probability=0.0),
    )

    assert _action_name(action) == "FOLD"


def test_ev_action_decision_keeps_fold_when_call_is_pruned_by_min_call_winrate():
    action = ev_action_decision(
        **_call_spot_kwargs(
            street=3,
            winrate=60,
            hand_class=8,
            bet_ratio=0.80,
            opp_pressure=0.70,
        )
    )

    assert _action_name(action) == "FOLD"


@pytest.mark.parametrize(
    ("hand_class", "bet_ratio", "opp_pressure"),
    [
        (7, 0.25, 0.30),
        (7, 0.25, 0.50),
        (7, 0.50, 0.30),
        (8, 0.25, 0.30),
    ],
)
def test_turn_bluffcatcher_range_keeps_some_calls_in_moderate_spots(hand_class, bet_ratio, opp_pressure):
    actions = _call_actions_in_range(
        street=2,
        hand_class=hand_class,
        bet_ratio=bet_ratio,
        opp_pressure=opp_pressure,
        winrates=range(48, 63),
    )
    assert "CALL" in actions.values(), (
        f"turn spot overfolded: hand_class={hand_class} bet_ratio={bet_ratio} "
        f"opp_pressure={opp_pressure} actions={actions}"
    )


@pytest.mark.parametrize(
    ("hand_class", "bet_ratio", "opp_pressure"),
    [
        (7, 0.25, 0.30),
        (7, 0.25, 0.50),
        (8, 0.25, 0.30),
        (8, 0.25, 0.50),
    ],
)
def test_river_bluffcatcher_range_keeps_some_calls_in_moderate_spots(hand_class, bet_ratio, opp_pressure):
    actions = _call_actions_in_range(
        street=3,
        hand_class=hand_class,
        bet_ratio=bet_ratio,
        opp_pressure=opp_pressure,
        winrates=range(52, 69),
    )
    assert "CALL" in actions.values(), (
        f"river spot overfolded: hand_class={hand_class} bet_ratio={bet_ratio} "
        f"opp_pressure={opp_pressure} actions={actions}"
    )


@pytest.mark.parametrize(
    ("street", "hand_class", "bet_ratio", "opp_pressure", "winrates"),
    [
        (2, 8, 0.80, 0.70, range(48, 63)),
        (3, 7, 0.80, 0.70, range(52, 69)),
        (3, 8, 0.50, 0.70, range(52, 69)),
    ],
)
def test_bluffcatcher_range_still_folds_extreme_pressure_spots(street, hand_class, bet_ratio, opp_pressure, winrates):
    actions = _call_actions_in_range(
        street=street,
        hand_class=hand_class,
        bet_ratio=bet_ratio,
        opp_pressure=opp_pressure,
        winrates=winrates,
    )
    assert set(actions.values()) == {"FOLD"}, actions


def test_facing_turn_raise_context_builds_expected_biases_from_call_heavy_history():
    tracker = AntiPredictTracker()
    _seed_history(tracker, "facing_turn_raise", {"CALL": 18, "FOLD": 6})
    observation = _make_observation(
        street=2,
        legal_action_names=("FOLD", "CALL"),
        my_bet=20,
        opp_bet=45,
        pot_size=40,
        opp_last_action="RAISE",
    )

    context = tracker.build_context(observation, ["FOLD", "CALL"], pot_size=40)

    assert context["node"] == "facing_turn_raise"
    assert context["observed"]["CALL"] == pytest.approx(0.75)
    assert context["biases"]["CALL"] == pytest.approx(-1.00368, abs=1e-4)
    assert context["biases"]["FOLD"] == pytest.approx(1.00368, abs=1e-4)


def test_facing_river_raise_context_builds_expected_biases_from_call_heavy_history():
    tracker = AntiPredictTracker()
    _seed_history(tracker, "facing_river_raise", {"CALL": 18, "FOLD": 6})
    observation = _make_observation(
        street=3,
        legal_action_names=("FOLD", "CALL"),
        my_bet=20,
        opp_bet=45,
        pot_size=40,
        opp_last_action="RAISE",
    )

    context = tracker.build_context(observation, ["FOLD", "CALL"], pot_size=40)

    assert context["node"] == "facing_river_raise"
    assert context["observed"]["CALL"] == pytest.approx(0.75)
    assert context["biases"]["CALL"] == pytest.approx(-1.8144, abs=1e-4)
    assert context["biases"]["FOLD"] == pytest.approx(1.8144, abs=1e-4)


def test_normalize_targets_remaps_call_to_check_when_call_is_illegal():
    tracker = AntiPredictTracker()
    observation = _make_observation(
        street=0,
        legal_action_names=("RAISE", "CHECK"),
        my_bet=2,
        opp_bet=2,
    )

    normalized = tracker._normalize_targets("preflop_unopened", ["RAISE", "CHECK"], observation)
    context = tracker.build_context(observation, ["RAISE", "CHECK"], pot_size=20)

    assert normalized == pytest.approx({"RAISE": 0.40, "CHECK": 0.60})
    assert context["targets"] == pytest.approx({"RAISE": 0.40, "CHECK": 0.60})


def test_normalize_targets_remaps_check_to_call_when_check_is_illegal_in_nonpaying_spot():
    tracker = AntiPredictTracker()
    observation = _make_observation(
        street=2,
        legal_action_names=("RAISE", "CALL"),
        my_bet=20,
        opp_bet=20,
        opp_last_action="CHECK",
        blind_position=0,
    )

    normalized = tracker._normalize_targets("turn_check_to_us", ["RAISE", "CALL"], observation)
    context = tracker.build_context(observation, ["RAISE", "CALL"], pot_size=20)

    assert normalized == pytest.approx({"RAISE": 0.42, "CALL": 0.58})
    assert context["targets"] == pytest.approx({"RAISE": 0.42, "CALL": 0.58})


def test_normalize_targets_drops_invalid_actions_and_renormalizes():
    tracker = AntiPredictTracker()
    observation = _make_observation(
        street=0,
        legal_action_names=("RAISE",),
        my_bet=2,
        opp_bet=2,
    )

    normalized = tracker._normalize_targets("preflop_unopened", ["RAISE"], observation)
    context = tracker.build_context(observation, ["RAISE"], pot_size=20)

    assert normalized == {"RAISE": 1.0}
    assert context["targets"] == {"RAISE": 1.0}


def test_antipredict_empty_history_observed_distribution_is_uniform():
    tracker = AntiPredictTracker()
    observed = tracker._observed_distribution("facing_turn_raise", ["CALL", "FOLD"])

    assert observed == pytest.approx({"CALL": 0.5, "FOLD": 0.5})


def test_antipredict_observed_distribution_updates_after_record_action():
    tracker = AntiPredictTracker()
    tracker.record_action("facing_turn_raise", "CALL")
    tracker.record_action("facing_turn_raise", "CALL")
    tracker.record_action("facing_turn_raise", "FOLD")

    observed = tracker._observed_distribution("facing_turn_raise", ["CALL", "FOLD"])

    assert observed == pytest.approx({"CALL": 2 / 3, "FOLD": 1 / 3})


def test_antipredict_history_discards_old_actions_after_maxlen():
    tracker = AntiPredictTracker()
    tracker._history["facing_turn_raise"] = deque(maxlen=3)

    for action_name in ("CALL", "FOLD", "CALL", "FOLD"):
        tracker.record_action("facing_turn_raise", action_name)

    observed = tracker._observed_distribution("facing_turn_raise", ["CALL", "FOLD"])

    assert list(tracker._history["facing_turn_raise"]) == ["FOLD", "CALL", "FOLD"]
    assert observed == pytest.approx({"CALL": 1 / 3, "FOLD": 2 / 3})


def test_antipredict_bias_direction_changes_as_history_shifts_from_call_heavy_to_fold_heavy():
    tracker = AntiPredictTracker()
    observation = _make_observation(
        street=2,
        legal_action_names=("FOLD", "CALL"),
        my_bet=20,
        opp_bet=45,
        pot_size=40,
        opp_last_action="RAISE",
    )

    _seed_history(tracker, "facing_turn_raise", {"CALL": 22, "FOLD": 2})
    call_heavy = tracker.build_context(observation, ["FOLD", "CALL"], pot_size=40)

    tracker._history["facing_turn_raise"].clear()
    _seed_history(tracker, "facing_turn_raise", {"CALL": 2, "FOLD": 22})
    fold_heavy = tracker.build_context(observation, ["FOLD", "CALL"], pot_size=40)

    assert call_heavy["biases"]["CALL"] < 0 < fold_heavy["biases"]["CALL"]
    assert call_heavy["biases"]["FOLD"] > 0 > fold_heavy["biases"]["FOLD"]
    assert fold_heavy["biases"]["CALL"] > call_heavy["biases"]["CALL"]


@pytest.mark.parametrize(
    ("street", "hand_class", "bet_ratio", "opp_pressure", "call_bias", "fold_bias", "winrates"),
    [
        (2, 7, 0.50, 0.50, -1.00368, 1.00368, range(48, 63)),
        (3, 8, 0.25, 0.50, -1.8144, 1.8144, range(52, 69)),
    ],
)
def test_realistic_facing_raise_biases_do_not_flip_representative_marginal_call_spots(
    street,
    hand_class,
    bet_ratio,
    opp_pressure,
    call_bias,
    fold_bias,
    winrates,
):
    flip_winrates = []
    for winrate in winrates:
        base_action = _action_name(
            ev_action_decision(
                **_call_spot_kwargs(
                    street=street,
                    winrate=winrate,
                    hand_class=hand_class,
                    bet_ratio=bet_ratio,
                    opp_pressure=opp_pressure,
                )
            )
        )
        biased_action = _action_name(
            ev_action_decision(
                **_call_spot_kwargs(
                    street=street,
                    winrate=winrate,
                    hand_class=hand_class,
                    bet_ratio=bet_ratio,
                    opp_pressure=opp_pressure,
                    action_biases={"CALL": call_bias, "FOLD": fold_bias},
                )
            )
        )
        if base_action == "CALL" and biased_action == "FOLD":
            flip_winrates.append(winrate)

    assert not flip_winrates, (
        f"realistic anti-predict biases unexpectedly flipped marginal calls on "
        f"street={street} hand_class={hand_class} bet_ratio={bet_ratio} "
        f"opp_pressure={opp_pressure}: {flip_winrates}"
    )


@pytest.mark.parametrize(
    ("winrate", "expect_call_present"),
    [
        (66, False),
        (67, True),
        (68, True),
    ],
)
def test_river_half_raise_floor_prunes_call_before_selection(monkeypatch, winrate, expect_call_present):
    captured = _capture_evs_before_selection(monkeypatch)

    action = ev_action_decision(
        **_call_spot_kwargs(
            street=3,
            winrate=winrate,
            hand_class=7,
            bet_ratio=0.50,
            opp_pressure=0.34,
        )
    )

    assert ("CALL" in captured["evs"]) is expect_call_present
    assert _action_name(action) == ("CALL" if expect_call_present else "FOLD")


@pytest.mark.parametrize(
    ("winrate", "expect_call_present"),
    [
        (85, False),
        (86, True),
        (87, True),
    ],
)
def test_river_eighty_pct_raise_floor_prunes_call_before_selection(monkeypatch, winrate, expect_call_present):
    captured = _capture_evs_before_selection(monkeypatch)

    action = ev_action_decision(
        **_call_spot_kwargs(
            street=3,
            winrate=winrate,
            hand_class=7,
            bet_ratio=0.80,
            opp_pressure=0.34,
        )
    )

    assert ("CALL" in captured["evs"]) is expect_call_present
    assert _action_name(action) == ("CALL" if expect_call_present else "FOLD")


@pytest.mark.parametrize(
    ("opp_bet", "opp_pressure", "winrate", "expected_name"),
    [
        (249, 0.30, 65, "CALL"),
        (250, 0.30, 65, "FOLD"),
        (240, 0.34, 65, "CALL"),
        (240, 0.35, 65, "FOLD"),
        (240, 0.36, 65, "FOLD"),
    ],
)
def test_river_bluffcatcher_cutoffs_near_ratio_and_pressure_thresholds(opp_bet, opp_pressure, winrate, expected_name):
    action = ev_action_decision(
        **_explicit_call_spot_kwargs(
            street=3,
            winrate=winrate,
            hand_class=8,
            pot_size=1000,
            opp_bet=opp_bet,
            opp_pressure=opp_pressure,
            max_raise=1000,
        )
    )

    assert _action_name(action) == expected_name


def test_realistic_facing_river_raise_biases_do_not_expand_river_fold_region():
    tracker = AntiPredictTracker()
    _seed_history(tracker, "facing_river_raise", {"CALL": 18, "FOLD": 6})
    context = tracker.build_context(
        _make_observation(
            street=3,
            legal_action_names=("FOLD", "CALL"),
            my_bet=20,
            opp_bet=45,
            pot_size=40,
            opp_last_action="RAISE",
        ),
        ["FOLD", "CALL"],
        pot_size=40,
    )

    without_bias = {
        winrate: _action_name(
            ev_action_decision(
                **_call_spot_kwargs(
                    street=3,
                    winrate=winrate,
                    hand_class=8,
                    bet_ratio=0.25,
                    opp_pressure=0.34,
                )
            )
        )
        for winrate in (65, 66, 67)
    }
    with_bias = {
        winrate: _action_name(
            ev_action_decision(
                **_call_spot_kwargs(
                    street=3,
                    winrate=winrate,
                    hand_class=8,
                    bet_ratio=0.25,
                    opp_pressure=0.34,
                    action_biases=context["biases"],
                )
            )
        )
        for winrate in (65, 66, 67)
    }

    assert context["biases"] == pytest.approx({"CALL": -1.8144, "FOLD": 1.8144})
    assert without_bias == {65: "FOLD", 66: "CALL", 67: "CALL"}
    assert with_bias == without_bias


def test_river_action_can_fold_after_river_aggression_discount_even_when_raw_ev_call_would_call(monkeypatch):
    monkeypatch.setattr(action_module, "predict_hand_winrate", lambda my_cards, pool, community_cards=None, trials=0: 68)
    monkeypatch.setattr(action_module, "hand_rank_class", lambda hole_cards, community_cards: 7)
    monkeypatch.setattr(action_module, "board_completion_threat", lambda community_cards: 0.0)

    direct_ev_action = ev_action_decision(
        **_call_spot_kwargs(
            street=3,
            winrate=68,
            hand_class=7,
            bet_ratio=0.50,
            opp_pressure=0.34,
        )
    )
    river_action_result = action_module.river_action(
        my_cards=[0, 1],
        community_cards=[2, 3, 4, 5, 6],
        card_pool=set(range(27)),
        valid_actions=_valid_actions("FOLD", "CALL"),
        min_raise=2,
        max_raise=100,
        pot_size=100,
        my_bet=0,
        opp_bet=50,
        opp_action_probs={"FOLD": 0.05, "RAISE": 0.35, "CALL": 0.60},
        opp_pressure=0.34,
        exploration=ExplorationSettings(mix_probability=0.0),
    )

    assert _action_name(direct_ev_action) == "CALL"
    assert _action_name(river_action_result) == "FOLD"


def test_apply_action_override_ignores_raise_when_raise_is_illegal():
    agent = _make_player_agent()
    base_action = (PokerEnv.ActionType.DISCARD.value, 0, 1, 2)

    final_action = agent._apply_action_override(
        base_action=base_action,
        chosen_action_name="RAISE",
        min_raise=2,
        max_raise=100,
        observation={"valid_actions": _valid_actions("FOLD", "CHECK")},
    )

    assert final_action == base_action


@pytest.mark.parametrize(
    ("min_raise", "max_raise", "expected_amount"),
    [
        (20, 30, 20),
        (2, 100, 12),
        (7, 7, 7),
    ],
)
def test_apply_action_override_raise_amount_respects_min_and_max_bounds(min_raise, max_raise, expected_amount):
    agent = _make_player_agent()
    base_action = (PokerEnv.ActionType.DISCARD.value, 0, 1, 2)

    final_action = agent._apply_action_override(
        base_action=base_action,
        chosen_action_name="RAISE",
        min_raise=min_raise,
        max_raise=max_raise,
        observation={"valid_actions": _valid_actions("FOLD", "RAISE", "CHECK")},
    )

    assert final_action == (PokerEnv.ActionType.RAISE.value, expected_amount, 0, 0)


@pytest.mark.parametrize("base_name", ["CHECK", "CALL"])
def test_apply_action_override_preserves_base_when_invalid_raise_is_proposed(base_name):
    agent = _make_player_agent()
    base_action = (PokerEnv.ActionType[base_name].value, 0, 0, 0)

    final_action = agent._apply_action_override(
        base_action=base_action,
        chosen_action_name="RAISE",
        min_raise=10,
        max_raise=5,
        observation={"valid_actions": _valid_actions("FOLD", base_name)},
    )

    assert final_action == base_action


@pytest.mark.parametrize(
    ("case_name", "street", "legal_action_names", "base_name", "anti_context", "dbbr_action", "my_bet", "opp_bet", "pot_size", "expected_name"),
    [
        (
            "free_check_spot",
            2,
            ("FOLD", "CHECK"),
            "CHECK",
            {"node": "turn_check_to_us", "biases": {"CHECK": -2.0, "FOLD": 2.0}, "sizing": None},
            "FOLD",
            20,
            20,
            60,
            "CHECK",
        ),
        (
            "legal_call_positive_ev",
            2,
            ("FOLD", "CALL"),
            "CALL",
            None,
            "FOLD",
            20,
            30,
            100,
            "CALL",
        ),
        (
            "turn_facing_raise_moderate_strength",
            2,
            ("FOLD", "CALL"),
            "CALL",
            {"node": "facing_turn_raise", "biases": {"CALL": -1.0, "FOLD": 1.0}, "sizing": None},
            "FOLD",
            20,
            45,
            100,
            "CALL",
        ),
        (
            "antipredict_and_dbbr_both_prefer_fold",
            3,
            ("FOLD", "CALL"),
            "CALL",
            {"node": "facing_river_raise", "biases": {"CALL": -1.8, "FOLD": 1.8}, "sizing": None},
            "FOLD",
            20,
            45,
            100,
            "CALL",
        ),
    ],
)
def test_player_act_returns_expected_final_action_across_fold_bug_cases(
    monkeypatch,
    case_name,
    street,
    legal_action_names,
    base_name,
    anti_context,
    dbbr_action,
    my_bet,
    opp_bet,
    pot_size,
    expected_name,
):
    base_action = (PokerEnv.ActionType[base_name].value, 0, 0, 0)
    action, captured, recorded, agent, _ = _run_player_act_case(
        monkeypatch,
        street=street,
        legal_action_names=legal_action_names,
        base_action=base_action,
        anti_context=anti_context,
        dbbr_action=dbbr_action,
        my_bet=my_bet,
        opp_bet=opp_bet,
        pot_size=pot_size,
    )

    assert _action_name(action) == expected_name, case_name
    assert captured["kwargs"]["action_biases"] == (anti_context or {}).get("biases")
    assert captured["kwargs"]["opp_action_probs"] == {"FOLD": 0.35, "CALL": 0.65}
    assert agent._last_decision["base"] == base_name
    assert agent._last_decision["override"] == dbbr_action
    assert agent._last_decision["final"] == expected_name
    assert recorded["action_name"] == expected_name


@pytest.mark.parametrize(
    ("street", "node_key"),
    [
        (2, "facing_turn_raise"),
        (3, "facing_river_raise"),
    ],
)
def test_player_act_with_real_antipredict_context_keeps_call_in_protected_spot(monkeypatch, street, node_key):
    agent = PlayerAgent(stream=False)
    agent._hand_counter = 999
    _seed_history(agent._anti_predict, node_key, {"CALL": 18, "FOLD": 6})

    captured = {}
    recorded = {}

    def fake_street_action(*args, **kwargs):
        captured["kwargs"] = kwargs
        return (PokerEnv.ActionType.CALL.value, 0, 0, 0)

    monkeypatch.setattr(player_module, "turn_action" if street == 2 else "river_action", fake_street_action)
    monkeypatch.setattr(agent, "_update_from_opp_last_action", lambda observation: None)
    monkeypatch.setattr(agent, "_maybe_rotate_style", lambda hand_number: None)
    monkeypatch.setattr(agent, "_opp_action_probs", lambda observation, legal: {"FOLD": 0.20, "RAISE": 0.30, "CALL": 0.50})
    monkeypatch.setattr(agent, "_opp_pressure", lambda observation: 0.50)
    monkeypatch.setattr(agent, "_current_aggression_scale", lambda pressure: 1.0)
    monkeypatch.setattr(agent, "_current_exploration_settings", lambda: ExplorationSettings(mix_probability=0.0))
    monkeypatch.setattr(agent._exploration, "snapshot", lambda: {"active": False, "mix_probability": 0.0})
    monkeypatch.setattr(agent._anti_predict, "record_action", lambda node, action_name: recorded.update({"node": node, "action_name": action_name}))
    monkeypatch.setattr(agent._dbbr, "select_action", lambda observation, legal: "FOLD")

    observation = _make_observation(
        street=street,
        legal_action_names=("FOLD", "CALL"),
        my_bet=20,
        opp_bet=45,
        pot_size=100,
        opp_last_action="RAISE",
        blind_position=0,
    )
    action = agent.act(observation, 0.0, False, False, {"hand_number": 21})

    assert _action_name(action) == "CALL"
    assert recorded["node"] == node_key
    assert recorded["action_name"] == "CALL"
    assert captured["kwargs"]["action_biases"]["CALL"] < 0
    assert captured["kwargs"]["action_biases"]["FOLD"] > 0


def test_player_act_river_keeps_call_in_realistic_facing_raise_protected_spot(monkeypatch):
    agent = PlayerAgent(stream=False)
    agent._hand_counter = 999
    _seed_history(agent._anti_predict, "facing_river_raise", {"CALL": 18, "FOLD": 6})

    monkeypatch.setattr(agent, "_update_from_opp_last_action", lambda observation: None)
    monkeypatch.setattr(agent, "_maybe_rotate_style", lambda hand_number: None)
    monkeypatch.setattr(agent, "_opp_action_probs", lambda observation, legal: {"FOLD": 0.20, "RAISE": 0.35, "CALL": 0.45})
    monkeypatch.setattr(agent, "_opp_pressure", lambda observation: 0.34)
    monkeypatch.setattr(agent, "_current_aggression_scale", lambda pressure: 1.0)
    monkeypatch.setattr(agent, "_current_exploration_settings", lambda: ExplorationSettings(mix_probability=0.0))
    monkeypatch.setattr(agent._exploration, "snapshot", lambda: {"active": False, "mix_probability": 0.0})
    monkeypatch.setattr(agent._dbbr, "select_action", lambda observation, legal: "FOLD")
    monkeypatch.setattr(action_module, "predict_hand_winrate", lambda my_cards, pool, community_cards=None, trials=0: 90)
    monkeypatch.setattr(action_module, "hand_rank_class", lambda hole_cards, community_cards: 7)
    monkeypatch.setattr(action_module, "board_completion_threat", lambda community_cards: 0.0)

    observation = _make_observation(
        street=3,
        legal_action_names=("FOLD", "CALL"),
        my_bet=0,
        opp_bet=10,
        pot_size=40,
        opp_last_action="RAISE",
        blind_position=0,
    )
    action = agent.act(observation, 0.0, False, False, {"hand_number": 31})

    assert _action_name(action) == "CALL"
    assert agent._last_decision["base"] == "CALL"
    assert agent._last_decision["override"] == "FOLD"
    assert agent._last_decision["final"] == "CALL"
    assert agent._last_decision["anti_predict"]["node"] == "facing_river_raise"


def test_player_act_river_fold_comes_from_base_river_action_when_call_is_pruned(monkeypatch):
    agent = PlayerAgent(stream=False)
    agent._hand_counter = 999

    monkeypatch.setattr(agent, "_update_from_opp_last_action", lambda observation: None)
    monkeypatch.setattr(agent, "_maybe_rotate_style", lambda hand_number: None)
    monkeypatch.setattr(agent, "_opp_action_probs", lambda observation, legal: {"FOLD": 0.05, "RAISE": 0.35, "CALL": 0.60})
    monkeypatch.setattr(agent, "_opp_pressure", lambda observation: 0.34)
    monkeypatch.setattr(agent, "_current_aggression_scale", lambda pressure: 1.0)
    monkeypatch.setattr(agent, "_current_exploration_settings", lambda: ExplorationSettings(mix_probability=0.0))
    monkeypatch.setattr(agent._exploration, "snapshot", lambda: {"active": False, "mix_probability": 0.0})
    monkeypatch.setattr(agent._dbbr, "select_action", lambda observation, legal: "CALL")
    monkeypatch.setattr(action_module, "predict_hand_winrate", lambda my_cards, pool, community_cards=None, trials=0: 68)
    monkeypatch.setattr(action_module, "hand_rank_class", lambda hole_cards, community_cards: 7)
    monkeypatch.setattr(action_module, "board_completion_threat", lambda community_cards: 0.0)

    observation = _make_observation(
        street=3,
        legal_action_names=("FOLD", "CALL"),
        my_bet=0,
        opp_bet=50,
        pot_size=100,
        opp_last_action="RAISE",
        blind_position=0,
    )
    action = agent.act(observation, 0.0, False, False, {"hand_number": 32})

    assert _action_name(action) == "FOLD"
    assert agent._last_decision["base"] == "FOLD"
    assert agent._last_decision["override"] == "CALL"
    assert agent._last_decision["final"] == "FOLD"


@pytest.mark.parametrize(
    ("action_name", "expected_index"),
    [
        ("FOLD", 0),
        ("RAISE", 1),
        ("CHECK", 2),
        ("CALL", 3),
        ("DISCARD", 4),
    ],
)
def test_action_type_enum_values_match_valid_action_indices(action_name, expected_index):
    assert PokerEnv.ActionType[action_name].value == expected_index
    valid_actions = _valid_actions(action_name)
    assert valid_actions[expected_index] == 1
    assert sum(valid_actions) == 1


@pytest.mark.parametrize("action_name", ["FOLD", "CHECK", "CALL"])
def test_action_name_round_trip_avoids_index_confusion(action_name):
    agent = _make_player_agent()
    action_tuple = (PokerEnv.ActionType[action_name].value, 0, 0, 0)

    assert agent._action_name_to_id(action_name) == PokerEnv.ActionType[action_name].value
    assert agent._action_tuple_to_name(action_tuple) == action_name


def test_recent_performance_tracker_uses_baseline_mix_probability_when_inactive():
    tracker = RecentPerformanceTracker(baseline_mix_probability=0.03)

    assert tracker.current_mix_probability() == pytest.approx(0.03)
    assert tracker.snapshot()["active"] is False


def test_recent_performance_tracker_triggers_after_significant_recent_underperformance():
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
    assert tracker.exploration_hands_left == 3


def test_recent_performance_tracker_mix_probability_decays_during_exploration_burst():
    tracker = RecentPerformanceTracker(
        recent_window=5,
        min_baseline_hands=8,
        explore_duration=4,
        cooldown_hands=2,
        significance_z=0.0,
        min_mean_drop=5.0,
    )

    for _ in range(8):
        tracker.record_hand_result(20.0)
    for _ in range(5):
        tracker.record_hand_result(-20.0)
        if tracker.exploration_hands_left > 0:
            break

    first = tracker.current_mix_probability()
    tracker.record_hand_result(-20.0)
    second = tracker.current_mix_probability()
    tracker.record_hand_result(-20.0)
    third = tracker.current_mix_probability()

    assert first > second > third >= tracker.baseline_mix_probability


def test_recent_performance_tracker_enters_and_exits_cooldown_after_burst():
    tracker = RecentPerformanceTracker(
        recent_window=5,
        min_baseline_hands=8,
        explore_duration=2,
        cooldown_hands=3,
        significance_z=0.0,
        min_mean_drop=5.0,
    )

    for _ in range(8):
        tracker.record_hand_result(20.0)
    for _ in range(5):
        tracker.record_hand_result(-20.0)
        if tracker.exploration_hands_left > 0:
            break

    assert tracker.record_hand_result(-20.0) is None
    end_event = tracker.record_hand_result(-20.0)

    assert end_event == "ended"
    assert tracker.cooldown_hands_left == 3
    assert tracker.current_mix_probability() == pytest.approx(tracker.baseline_mix_probability)

    tracker.record_hand_result(-20.0)
    tracker.record_hand_result(-20.0)
    tracker.record_hand_result(-20.0)
    assert tracker.cooldown_hands_left == 0


def test_recent_performance_tracker_risk_off_turns_on_in_sustained_drawdown():
    tracker = RecentPerformanceTracker(recent_window=12)

    for _ in range(12):
        tracker.record_hand_result(10.0)
    for _ in range(12):
        tracker.record_hand_result(-10.0)

    assert tracker.risk_off() is True


def test_select_action_with_exploration_does_not_sample_fold_when_non_fold_action_is_best():
    action, candidates = select_action_with_exploration(
        {"CHECK": 1.0, "FOLD": 0.0},
        pot=5,
        exploration=ExplorationSettings(mix_probability=1.0, ev_margin_pct=0.40, ev_margin_floor=2.0),
        rng=FixedRng(random_values=[0.0, 0.99]),
    )

    assert action == "CHECK"
    assert all(action_name != "FOLD" for action_name, _ in candidates)


def test_select_action_with_exploration_excludes_fold_from_check_call_mix():
    action, candidates = select_action_with_exploration(
        {"CHECK": 10.0, "FOLD": 9.95, "CALL": 9.90},
        pot=50,
        exploration=ExplorationSettings(mix_probability=1.0, ev_margin_pct=0.10, ev_margin_floor=6.0),
        rng=FixedRng(random_values=[0.0, 0.99]),
    )

    assert [action_name for action_name, _ in candidates] == ["CHECK", "CALL"]
    assert action == "CALL"


def test_select_action_with_exploration_excludes_fold_from_raise_call_mix():
    action, candidates = select_action_with_exploration(
        {"RAISE": 10.0, "FOLD": 9.95, "CALL": 9.90},
        pot=50,
        exploration=ExplorationSettings(mix_probability=1.0, ev_margin_pct=0.10, ev_margin_floor=6.0),
        rng=FixedRng(random_values=[0.0, 0.99]),
    )

    assert [action_name for action_name, _ in candidates] == ["RAISE", "CALL"]
    assert action == "CALL"


def test_ev_action_decision_with_mix_never_samples_fold_when_check_is_legal():
    action = ev_action_decision(
        winrate=40,
        valid_actions=_valid_actions("FOLD", "CHECK"),
        min_raise=2,
        max_raise=20,
        pot_size=5,
        my_bet=0,
        opp_bet=0,
        opp_action_probs={"FOLD": 0.5},
        street=2,
        action_biases={"CHECK": 0.0, "FOLD": 0.0},
        exploration=ExplorationSettings(mix_probability=1.0, ev_margin_pct=0.40, ev_margin_floor=2.0),
        rng=FixedRng(random_values=[0.0, 0.99]),
    )

    assert _action_name(action) == "CHECK"


def test_ev_action_decision_with_mix_never_samples_fold_when_call_is_clearly_profitable():
    action = ev_action_decision(
        **_call_spot_kwargs(
            street=2,
            winrate=88,
            hand_class=2,
            bet_ratio=0.25,
            opp_pressure=0.30,
            action_biases={"CALL": -0.2, "FOLD": 0.0},
            exploration=ExplorationSettings(mix_probability=1.0, ev_margin_pct=0.40, ev_margin_floor=10.0),
        ),
        rng=FixedRng(random_values=[0.0, 0.99]),
    )

    assert _action_name(action) == "CALL"
