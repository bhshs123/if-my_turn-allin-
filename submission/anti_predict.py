from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class NodeTargetProfile:
    targets: dict[str, float]
    strength: float
    window: int = 72


@dataclass(frozen=True)
class NodeSizingProfile:
    targets: dict[str, float]
    strength: float


class AntiPredictTracker:
    """Keep key public nodes near target action frequencies.

    This is a lightweight anti-exploit layer: it does not replace the EV policy,
    it only nudges near-marginal spots back toward a mixed strategy.
    """

    def __init__(self) -> None:
        self._profiles = {
            "preflop_unopened": NodeTargetProfile({"RAISE": 0.40, "CALL": 0.60}, strength=1.05),
            "flop_check_to_us": NodeTargetProfile({"RAISE": 0.56, "CHECK": 0.44}, strength=1.20),
            "turn_check_to_us": NodeTargetProfile({"RAISE": 0.42, "CHECK": 0.58}, strength=0.75),
            "river_check_to_us": NodeTargetProfile({"RAISE": 0.28, "CHECK": 0.72}, strength=0.55),
            "facing_turn_raise": NodeTargetProfile({"CALL": 0.24, "FOLD": 0.76}, strength=0.82),
            "facing_river_raise": NodeTargetProfile({"CALL": 0.12, "FOLD": 0.88}, strength=1.20),
        }
        self._sizing_profiles = {
            "preflop_unopened": NodeSizingProfile({"small": 0.42, "standard": 0.35, "pressure": 0.23}, strength=0.55),
            "flop_check_to_us": NodeSizingProfile({"small": 0.24, "standard": 0.50, "pressure": 0.26}, strength=0.75),
            "turn_check_to_us": NodeSizingProfile({"small": 0.18, "standard": 0.50, "pressure": 0.32}, strength=0.70),
            "river_check_to_us": NodeSizingProfile({"small": 0.44, "standard": 0.40, "pressure": 0.16}, strength=0.45),
        }
        self._history = {
            key: deque(maxlen=profile.window) for key, profile in self._profiles.items()
        }

    def identify_node(self, observation, legal_actions) -> Optional[str]:
        street = int(observation.get("street", -1))
        my_bet = int(observation.get("my_bet", 0))
        opp_bet = int(observation.get("opp_bet", 0))
        opp_last_action = str(observation.get("opp_last_action", "None"))
        blind_position = int(observation.get("blind_position", -1))
        legal = set(legal_actions)

        # Fallback when `opp_last_action` is not populated by the local env path:
        # post-flop, SB acts second; if bets are equal and we are SB with raise legal,
        # the line is effectively check-to-us.
        checked_to_us = opp_last_action == "CHECK" or (
            street >= 1 and opp_bet == my_bet and blind_position == 0
        )

        if street == 0 and "RAISE" in legal and opp_bet <= my_bet:
            return "preflop_unopened"

        if street == 1 and opp_bet == my_bet and checked_to_us and "RAISE" in legal:
            return "flop_check_to_us"
        if street == 2 and opp_bet == my_bet and checked_to_us and "RAISE" in legal:
            return "turn_check_to_us"
        if street == 3 and opp_bet == my_bet and checked_to_us and "RAISE" in legal:
            return "river_check_to_us"
        if street == 2 and opp_bet > my_bet and "CALL" in legal and "FOLD" in legal:
            return "facing_turn_raise"
        if street == 3 and opp_bet > my_bet and "CALL" in legal and "FOLD" in legal:
            return "facing_river_raise"
        return None

    def _normalize_targets(self, node_key: str, legal_actions, observation) -> dict[str, float]:
        profile = self._profiles[node_key]
        legal = set(legal_actions)
        mapped: dict[str, float] = {}
        for action_name, prob in profile.targets.items():
            resolved = action_name
            if action_name == "CALL" and "CALL" not in legal and "CHECK" in legal:
                resolved = "CHECK"
            elif (
                action_name == "CHECK"
                and "CHECK" not in legal
                and "CALL" in legal
                and int(observation.get("opp_bet", 0)) <= int(observation.get("my_bet", 0))
            ):
                resolved = "CALL"
            if resolved not in legal:
                continue
            mapped[resolved] = mapped.get(resolved, 0.0) + prob

        total = sum(mapped.values())
        if total <= 0.0:
            return {}
        return {action_name: prob / total for action_name, prob in mapped.items()}

    def _observed_distribution(self, node_key: str, actions: list[str]) -> dict[str, float]:
        history = self._history[node_key]
        if not history:
            uniform = 1.0 / max(1, len(actions))
            return {action_name: uniform for action_name in actions}

        counts = {action_name: 0 for action_name in actions}
        hits = 0
        for action_name in history:
            if action_name in counts:
                counts[action_name] += 1
                hits += 1
        if hits == 0:
            uniform = 1.0 / max(1, len(actions))
            return {action_name: uniform for action_name in actions}
        return {action_name: counts[action_name] / hits for action_name in actions}

    def build_context(self, observation, legal_actions, pot_size: int) -> Optional[dict]:
        node_key = self.identify_node(observation, legal_actions)
        if not node_key:
            return None

        targets = self._normalize_targets(node_key, legal_actions, observation)
        if not targets:
            return None

        observed = self._observed_distribution(node_key, list(targets.keys()))
        profile = self._profiles[node_key]
        scale = max(1.2, 0.06 * max(1, int(pot_size)))
        cap = max(1.0, 0.12 * max(1, int(pot_size)))
        biases: dict[str, float] = {}
        for action_name, target in targets.items():
            gap = target - observed.get(action_name, 0.0)
            bias = max(-cap, min(cap, gap * profile.strength * scale))
            if abs(bias) >= 0.08:
                biases[action_name] = bias

        sizing = None
        sizing_profile = self._sizing_profiles.get(node_key)
        if sizing_profile and "RAISE" in set(legal_actions):
            sizing = {
                "weights": dict(sizing_profile.targets),
                "strength": sizing_profile.strength,
            }

        return {
            "node": node_key,
            "targets": targets,
            "observed": observed,
            "biases": biases,
            "sizing": sizing,
        }

    def record_action(self, node_key: Optional[str], action_name: str) -> None:
        if not node_key or node_key not in self._history:
            return
        self._history[node_key].append(action_name)