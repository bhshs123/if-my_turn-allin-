import os
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from agents.agent import Agent
from gym_env import PokerEnv
from submission.strategies.basic import remaining_card_pool
from submission.opponent_modeling import DBBRConfig, DBBROpponentModel
from submission.action import (
    ExplorationSettings,
    preflop_action,
    flop_action,
    discard_action,
    turn_action,
    river_action,
)


@dataclass
class RecentPerformanceTracker:
    """Track match-level drift and open a short exploration window when needed."""

    recent_window: int = 50
    min_baseline_hands: int = 50
    explore_duration: int = 50
    cooldown_hands: int = 25
    max_mix_probability: float = 0.10
    min_mix_probability: float = 0.05
    baseline_mix_probability: float = 0.025
    significance_z: float = 1.35
    min_mean_drop: float = 8.0
    recent_rewards: deque = field(init=False, repr=False)
    total_hands: int = 0
    total_reward: float = 0.0
    total_reward_sq: float = 0.0
    exploration_hands_left: int = 0
    cooldown_hands_left: int = 0

    def __post_init__(self) -> None:
        self.recent_rewards = deque(maxlen=self.recent_window)

    def _recent_mean(self) -> float:
        if not self.recent_rewards:
            return 0.0
        return sum(self.recent_rewards) / len(self.recent_rewards)

    def _recent_std(self) -> float:
        if len(self.recent_rewards) < 2:
            return 0.0
        mean = self._recent_mean()
        variance = sum((reward - mean) ** 2 for reward in self.recent_rewards) / len(self.recent_rewards)
        return math.sqrt(max(0.0, variance))

    def long_mean(self) -> float:
        if self.total_hands <= 0:
            return 0.0
        return self.total_reward / self.total_hands

    def long_std(self) -> float:
        if self.total_hands < 2:
            return 0.0
        mean = self.long_mean()
        variance = (self.total_reward_sq / self.total_hands) - (mean * mean)
        return math.sqrt(max(0.0, variance))

    def _meaningful_drop_threshold(self) -> float:
        window_size = max(1, len(self.recent_rewards))
        stderr = max(self.long_std(), self._recent_std()) / math.sqrt(window_size)
        return max(self.min_mean_drop, self.significance_z * stderr)

    def should_trigger(self) -> bool:
        if self.exploration_hands_left > 0 or self.cooldown_hands_left > 0:
            return False
        if self.total_hands < self.min_baseline_hands:
            return False
        if len(self.recent_rewards) < self.recent_window:
            return False

        recent_mean = self._recent_mean()
        long_mean = self.long_mean()
        if recent_mean >= long_mean:
            return False

        return (long_mean - recent_mean) >= self._meaningful_drop_threshold()

    def record_hand_result(self, reward: float) -> str | None:
        """Update rolling stats once per hand and manage window/cooldown state."""
        reward = float(reward)
        self.total_hands += 1
        self.total_reward += reward
        self.total_reward_sq += reward * reward
        self.recent_rewards.append(reward)

        if self.exploration_hands_left > 0:
            self.exploration_hands_left -= 1
            if self.exploration_hands_left == 0:
                self.cooldown_hands_left = self.cooldown_hands
                return "ended"
            return None

        if self.cooldown_hands_left > 0:
            self.cooldown_hands_left -= 1
            return None

        if self.should_trigger():
            self.exploration_hands_left = self.explore_duration
            return "triggered"
        return None

    def current_mix_probability(self) -> float:
        if self.exploration_hands_left <= 0:
            return self.baseline_mix_probability
        if self.explore_duration <= 1:
            return max(self.min_mix_probability, self.baseline_mix_probability)

        progress = (self.exploration_hands_left - 1) / max(1, self.explore_duration - 1)
        burst = self.min_mix_probability + progress * (self.max_mix_probability - self.min_mix_probability)
        return max(self.baseline_mix_probability, burst)

    def snapshot(self) -> dict:
        return {
            "active": self.exploration_hands_left > 0,
            "mix_probability": round(self.current_mix_probability(), 4),
            "recent_mean": round(self._recent_mean(), 2),
            "long_mean": round(self.long_mean(), 2),
            "remaining": self.exploration_hands_left,
            "cooldown": self.cooldown_hands_left,
            "trigger_gap": round(self._meaningful_drop_threshold(), 2),
        }

    def risk_off(self) -> bool:
        """Tighten aggression when recent performance is meaningfully below baseline."""
        if self.total_hands < 24 or len(self.recent_rewards) < min(12, self.recent_window):
            return False
        recent = self._recent_mean()
        long = self.long_mean()
        return (recent < -4.0) or ((long - recent) >= max(6.0, self._meaningful_drop_threshold() * 0.8))


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

    This is a lightweight first-phase anti-exploit layer: it does not replace the
    EV policy, it only nudges near-marginal spots back toward a mixed strategy.
    """

    def __init__(self) -> None:
        self._profiles = {
            "preflop_unopened": NodeTargetProfile({"RAISE": 0.38, "CALL": 0.62}, strength=0.85),
            "flop_check_to_us": NodeTargetProfile({"RAISE": 0.56, "CHECK": 0.44}, strength=0.90),
            "turn_check_to_us": NodeTargetProfile({"RAISE": 0.42, "CHECK": 0.58}, strength=0.75),
            "river_check_to_us": NodeTargetProfile({"RAISE": 0.28, "CHECK": 0.72}, strength=0.55),
            "facing_turn_raise": NodeTargetProfile({"CALL": 0.24, "FOLD": 0.76}, strength=0.70),
            "facing_river_raise": NodeTargetProfile({"CALL": 0.12, "FOLD": 0.88}, strength=0.60),
        }
        self._sizing_profiles = {
            "preflop_unopened": NodeSizingProfile({"small": 0.52, "standard": 0.34, "pressure": 0.14}, strength=0.55),
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
        legal = set(legal_actions)

        if street == 0 and "RAISE" in legal and opp_bet <= my_bet:
            return "preflop_unopened"

        if street == 1 and opp_bet == my_bet and opp_last_action == "CHECK" and "RAISE" in legal:
            return "flop_check_to_us"
        if street == 2 and opp_bet == my_bet and opp_last_action == "CHECK" and "RAISE" in legal:
            return "turn_check_to_us"
        if street == 3 and opp_bet == my_bet and opp_last_action == "CHECK" and "RAISE" in legal:
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
            elif action_name == "CHECK" and "CHECK" not in legal and "CALL" in legal and int(observation.get("opp_bet", 0)) <= int(observation.get("my_bet", 0)):
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


class PlayerAgent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self._verbose = os.getenv("BOT_DEBUG_PLAYER", "0") == "1"
        self.action_types = PokerEnv.ActionType
        raw_player_id = str(os.getenv("PLAYER_ID", "0"))
        if raw_player_id.lower().startswith("bot") and raw_player_id[-1].isdigit():
            self._my_player_id = int(raw_player_id[-1])
        elif raw_player_id.isdigit():
            self._my_player_id = int(raw_player_id)
        else:
            self._my_player_id = 0
        self._hand_counter = 0
        self._last_seen_hand_number = None
        self._last_decision: dict | None = None
        self._rng = random.Random()
        self._exploration = RecentPerformanceTracker()
        self._anti_predict = AntiPredictTracker()
        self._opp_raise_ema = 0.0
        self._opp_raise_streak = 0
        self._style_mode = "balanced"
        self._style_hands_left = 0

        # Pending public state where opponent is about to act; confirmed on next opp_last_action.
        self._pending_opp_state: dict | None = None
        self._pending_opp_parent_key: str | None = None
        self._last_opp_public_key: str | None = None

        self._dbbr = DBBROpponentModel(
            config=DBBRConfig(
                warmup_iters=80,
                update_interval=20,
                n_prior=8.0,
                tolerance=1e-6,
                use_exploit_threshold=True,
                exploit_threshold=0.02,
                min_obs_per_node=2,
                deterministic_action_selection=False,
                debug_logging=(os.getenv("BOT_DEBUG_DBBR", "0") == "1"),
                enable_exploitation=True,
            )
        )

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(msg)

    def __name__(self):
        return "PlayerAgent"

    def _legal_action_names(self, valid_actions):
        names = []
        for i, is_valid in enumerate(valid_actions):
            if not is_valid:
                continue
            try:
                names.append(PokerEnv.ActionType(i).name)
            except ValueError:
                continue
        return names

    def _action_name_to_id(self, action_name: str) -> int:
        return PokerEnv.ActionType[action_name].value

    def _fmt_cards(self, cards):
        out = []
        for card in cards or []:
            if isinstance(card, int) and card >= 0:
                try:
                    out.append(PokerEnv.int_to_card(card))
                except Exception:
                    out.append(str(card))
        return out

    def _action_tuple_to_name(self, action_tuple) -> str:
        try:
            return PokerEnv.ActionType(action_tuple[0]).name
        except Exception:
            return str(action_tuple[0])

    def _current_exploration_settings(self) -> ExplorationSettings:
        """Build the current exploration profile for the EV action layer."""
        mix_probability = self._exploration.current_mix_probability()
        if self._style_mode == "pressure":
            mix_probability = min(0.12, mix_probability + 0.01)
        elif self._style_mode == "trap":
            mix_probability = max(0.01, mix_probability - 0.005)

        if self._exploration.risk_off():
            # Keep some unpredictability, but avoid high-variance lines during drawdown.
            mix_probability = min(mix_probability, 0.03)
            return ExplorationSettings(
                mix_probability=max(0.01, mix_probability),
                max_candidate_actions=3,
                ev_margin_pct=0.05,
                ev_margin_floor=1.8,
                raise_jitter_pct=0.04,
            )

        if self._style_mode == "pressure":
            return ExplorationSettings(
                mix_probability=mix_probability,
                max_candidate_actions=4,
                ev_margin_pct=0.075,
                ev_margin_floor=2.2,
                raise_jitter_pct=0.10,
            )
        if self._style_mode == "trap":
            return ExplorationSettings(
                mix_probability=mix_probability,
                max_candidate_actions=3,
                ev_margin_pct=0.055,
                ev_margin_floor=1.8,
                raise_jitter_pct=0.06,
            )

        return ExplorationSettings(
            mix_probability=mix_probability,
            max_candidate_actions=4,
            ev_margin_pct=0.065,
            ev_margin_floor=2.0,
            raise_jitter_pct=0.08 if mix_probability > 0.0 else 0.05,
        )

    def _maybe_rotate_style(self, hand_number) -> None:
        if hand_number == self._last_seen_hand_number:
            return
        self._last_seen_hand_number = hand_number

        if self._style_hands_left > 0:
            self._style_hands_left -= 1
            return

        # Non-stationary style mix to reduce exploitability by memory-based bots.
        roll = self._rng.random()
        if roll < 0.50:
            self._style_mode = "balanced"
            self._style_hands_left = self._rng.randint(10, 22)
        elif roll < 0.80:
            self._style_mode = "pressure"
            self._style_hands_left = self._rng.randint(6, 14)
        else:
            self._style_mode = "trap"
            self._style_hands_left = self._rng.randint(6, 14)

    def _current_aggression_scale(self, opp_pressure: float) -> float:
        """Throttle offensive lines during drawdowns or heavy opponent pressure."""
        scale = 1.0
        if self._style_mode == "pressure":
            scale += 0.06
        elif self._style_mode == "trap":
            scale -= 0.05
        if self._exploration.risk_off():
            scale -= 0.18
        if opp_pressure >= 0.65:
            scale -= 0.10
        elif opp_pressure >= 0.45:
            scale -= 0.05
        return max(0.70, min(1.05, scale))

    def _apply_action_override(self, base_action, chosen_action_name, min_raise, max_raise, observation):
        if chosen_action_name is None:
            return base_action

        base_type, _, keep_1, keep_2 = base_action
        chosen_id = self._action_name_to_id(chosen_action_name)
        if chosen_id == base_type:
            return base_action

        # Warmup data is too noisy for action overrides; collect observations only.
        if self._hand_counter <= self._dbbr.config.warmup_iters:
            return base_action

        # Keep fold/call/check decisions EV-driven; DBBR exploit can tune aggression,
        # but should not force extra folds that look unintelligent.
        if chosen_action_name == "FOLD":
            return base_action

        # If EV says fold/call/check, do not turn that into a looser action.
        if base_type in (
            PokerEnv.ActionType.FOLD.value,
            PokerEnv.ActionType.CALL.value,
            PokerEnv.ActionType.CHECK.value,
        ):
            return base_action

        # Never fold when a free check exists (defensive safety invariant).
        valid_actions = observation.get("valid_actions", [])
        if chosen_action_name == "CHECK":
            check_id = PokerEnv.ActionType.CHECK.value
            if check_id >= len(valid_actions) or not valid_actions[check_id]:
                return base_action

        if chosen_action_name == "RAISE":
            amt = max(min_raise, int(max_raise * 0.12))
            return chosen_id, amt, 0, 0
        if chosen_action_name in ("FOLD", "CHECK", "CALL"):
            return chosen_id, 0, 0, 0
        if chosen_action_name == "DISCARD":
            return chosen_id, 0, keep_1, keep_2
        return base_action

    def _opp_action_probs(self, observation, legal_actions):
        """Return gamma[n,a] = sum_b beta[n,b]*sigma[n,b,a] for the opponent's current node.
        Falls back to baseline heuristic during warmup or for unseen nodes."""
        opp_key = self._dbbr.encoder.encode_public_state(observation, actor="opponent")
        probs = self._dbbr.gamma.get(opp_key)
        if not probs:
            probs = self._dbbr.baseline_policy.get_action_probs(opp_key, legal_actions=legal_actions)
        return probs

    def _update_from_opp_last_action(self, observation):
        action_name = str(observation.get("opp_last_action", "None"))
        if action_name == "None" or self._pending_opp_state is None:
            return
        if action_name not in {"FOLD", "RAISE", "CHECK", "CALL", "DISCARD"}:
            self._pending_opp_state = None
            return

        if action_name == "RAISE":
            self._opp_raise_streak += 1
            self._opp_raise_ema = 0.80 * self._opp_raise_ema + 0.20
        else:
            self._opp_raise_streak = 0
            self._opp_raise_ema = 0.85 * self._opp_raise_ema

        self._dbbr.observe_opponent_action(
            self._pending_opp_state,
            action_name,
            metadata={
                "parent_public_key": self._pending_opp_parent_key,
                "parent_action": action_name,
            },
        )
        self._last_opp_public_key = self._dbbr.encoder.encode_public_state(self._pending_opp_state, actor="opponent")
        self._pending_opp_state = None

    def _opp_pressure(self, observation) -> float:
        immediate = 1.0 if str(observation.get("opp_last_action", "None")) == "RAISE" else 0.0
        streak_bonus = min(0.35, 0.12 * self._opp_raise_streak)
        return min(1.0, max(immediate, self._opp_raise_ema + streak_bonus))

    def act(self, observation, reward, terminated, truncated, info):
        """
        The structure below outlines where to add decision logic for:
        - pre-flop (street 0)
        - flop discard round (street 1, discard action required)
        - flop betting (street 1 after discard)
        - turn (street 2)
        - river (street 3)

        For each stage, inspect `observation` and `valid_actions` and return a tuple:
          (action_type, raise_amount, keep_card_1, keep_card_2)

        """
        # Pull opponent action observation from the previous transition into DBBR counters.
        self._update_from_opp_last_action(observation)

        self._maybe_rotate_style(info.get("hand_number"))

        street = observation["street"]
        valid_actions = observation["valid_actions"]
        legal_action_names = self._legal_action_names(valid_actions)
        my_cards = observation.get("my_cards", [])
        community_cards = observation.get("community_cards", [])
        min_raise = observation.get("min_raise", 1)
        max_raise = observation.get("max_raise", 100)
        my_bet = observation.get("my_bet", 0)
        opp_bet = observation.get("opp_bet", 0)
        pot_size = observation.get("pot_size", 0)
        # Collect known-discarded cards so the pool excludes them from simulation.
        my_discarded = [c for c in observation.get("my_discarded_cards", []) if isinstance(c, int) and c >= 0]
        opp_discarded = [c for c in observation.get("opp_discarded_cards", []) if isinstance(c, int) and c >= 0]
        dead_cards = my_discarded + opp_discarded

        opp_action_probs = self._opp_action_probs(observation, legal_action_names)
        opp_pressure = self._opp_pressure(observation)
        aggression_scale = self._current_aggression_scale(opp_pressure)
        exploration = self._current_exploration_settings()
        exploration_status = self._exploration.snapshot()
        anti_predict = self._anti_predict.build_context(observation, legal_action_names, pot_size)

        # --- Pre-flop (street 0) ---
        if street == 0:
            base_action = preflop_action(
                my_cards, remaining_card_pool, valid_actions, min_raise, max_raise,
                pot_size=pot_size, my_bet=my_bet, opp_bet=opp_bet, opp_action_probs=opp_action_probs,
                opp_pressure=opp_pressure,
                aggression_scale=aggression_scale,
                action_biases=(anti_predict or {}).get("biases"),
                raise_sizing=(anti_predict or {}).get("sizing"),
                exploration=exploration, rng=self._rng,
            )
            chosen = self._dbbr.select_action(observation, legal_action_names)
            final_action = self._apply_action_override(base_action, chosen, min_raise, max_raise, observation)
            self._anti_predict.record_action((anti_predict or {}).get("node"), self._action_tuple_to_name(final_action))
            self._last_decision = {
                "hand": info.get("hand_number"),
                "street": street,
                "my_cards": self._fmt_cards(my_cards),
                "board": self._fmt_cards(community_cards),
                "my_bet": my_bet,
                "opp_bet": opp_bet,
                "pot": pot_size,
                "opp_probs": dict(opp_action_probs),
                "opp_pressure": round(opp_pressure, 3),
                "aggression_scale": round(aggression_scale, 3),
                "style_mode": self._style_mode,
                "exploration": exploration_status,
                "anti_predict": anti_predict,
                "base": self._action_tuple_to_name(base_action),
                "override": chosen,
                "final": self._action_tuple_to_name(final_action),
            }
            self._log(f"[decision] {self._last_decision}")
            return final_action

        # --- Flop discard round (street 1) ---
        if street == 1 and valid_actions[self.action_types.DISCARD.value]:
            return discard_action(my_cards, community_cards, remaining_card_pool, dead_cards=dead_cards)

        # --- Flop betting (street 1 after discard) ---
        if street == 1:
            base_action = flop_action(
                my_cards, community_cards, remaining_card_pool, valid_actions, min_raise, max_raise,
                dead_cards=dead_cards, pot_size=pot_size, my_bet=my_bet, opp_bet=opp_bet, opp_action_probs=opp_action_probs,
                opp_pressure=opp_pressure,
                aggression_scale=aggression_scale,
                action_biases=(anti_predict or {}).get("biases"),
                raise_sizing=(anti_predict or {}).get("sizing"),
                exploration=exploration, rng=self._rng,
            )
            chosen = self._dbbr.select_action(observation, legal_action_names)
            final_action = self._apply_action_override(base_action, chosen, min_raise, max_raise, observation)
            self._anti_predict.record_action((anti_predict or {}).get("node"), self._action_tuple_to_name(final_action))
            self._last_decision = {
                "hand": info.get("hand_number"),
                "street": street,
                "my_cards": self._fmt_cards(my_cards),
                "board": self._fmt_cards(community_cards),
                "my_bet": my_bet,
                "opp_bet": opp_bet,
                "pot": pot_size,
                "opp_probs": dict(opp_action_probs),
                "opp_pressure": round(opp_pressure, 3),
                "aggression_scale": round(aggression_scale, 3),
                "style_mode": self._style_mode,
                "exploration": exploration_status,
                "anti_predict": anti_predict,
                "base": self._action_tuple_to_name(base_action),
                "override": chosen,
                "final": self._action_tuple_to_name(final_action),
            }
            self._log(f"[decision] {self._last_decision}")
            return final_action

        # --- Turn (street 2) ---
        if street == 2:
            base_action = turn_action(
                my_cards, community_cards, remaining_card_pool, valid_actions, min_raise, max_raise,
                dead_cards=dead_cards, pot_size=pot_size, my_bet=my_bet, opp_bet=opp_bet, opp_action_probs=opp_action_probs,
                opp_pressure=opp_pressure,
                aggression_scale=aggression_scale,
                action_biases=(anti_predict or {}).get("biases"),
                raise_sizing=(anti_predict or {}).get("sizing"),
                exploration=exploration, rng=self._rng,
            )
            chosen = self._dbbr.select_action(observation, legal_action_names)
            final_action = self._apply_action_override(base_action, chosen, min_raise, max_raise, observation)
            self._anti_predict.record_action((anti_predict or {}).get("node"), self._action_tuple_to_name(final_action))
            self._last_decision = {
                "hand": info.get("hand_number"),
                "street": street,
                "my_cards": self._fmt_cards(my_cards),
                "board": self._fmt_cards(community_cards),
                "my_bet": my_bet,
                "opp_bet": opp_bet,
                "pot": pot_size,
                "opp_probs": dict(opp_action_probs),
                "opp_pressure": round(opp_pressure, 3),
                "aggression_scale": round(aggression_scale, 3),
                "style_mode": self._style_mode,
                "exploration": exploration_status,
                "anti_predict": anti_predict,
                "base": self._action_tuple_to_name(base_action),
                "override": chosen,
                "final": self._action_tuple_to_name(final_action),
            }
            self._log(f"[decision] {self._last_decision}")
            return final_action

        # --- River (street 3) ---
        if street == 3:
            base_action = river_action(
                my_cards, community_cards, remaining_card_pool, valid_actions, min_raise, max_raise,
                dead_cards=dead_cards, pot_size=pot_size, my_bet=my_bet, opp_bet=opp_bet, opp_action_probs=opp_action_probs,
                opp_pressure=opp_pressure,
                aggression_scale=aggression_scale,
                action_biases=(anti_predict or {}).get("biases"),
                raise_sizing=(anti_predict or {}).get("sizing"),
                exploration=exploration, rng=self._rng,
            )
            chosen = self._dbbr.select_action(observation, legal_action_names)
            final_action = self._apply_action_override(base_action, chosen, min_raise, max_raise, observation)
            self._anti_predict.record_action((anti_predict or {}).get("node"), self._action_tuple_to_name(final_action))
            self._last_decision = {
                "hand": info.get("hand_number"),
                "street": street,
                "my_cards": self._fmt_cards(my_cards),
                "board": self._fmt_cards(community_cards),
                "my_bet": my_bet,
                "opp_bet": opp_bet,
                "pot": pot_size,
                "opp_probs": dict(opp_action_probs),
                "opp_pressure": round(opp_pressure, 3),
                "aggression_scale": round(aggression_scale, 3),
                "style_mode": self._style_mode,
                "exploration": exploration_status,
                "anti_predict": anti_predict,
                "base": self._action_tuple_to_name(base_action),
                "override": chosen,
                "final": self._action_tuple_to_name(final_action),
            }
            self._log(f"[decision] {self._last_decision}")
            return final_action

        # Fallback
        return self.action_types.FOLD.value, 0, 0, 0

    def observe(self, observation, reward, terminated, truncated, info) -> None:
        _ = truncated
        # Before opponent acts, cache the public state; action label will arrive as opp_last_action.
        if int(observation.get("acting_agent", -1)) != self._my_player_id:
            self._pending_opp_state = dict(observation)
            self._pending_opp_parent_key = self._last_opp_public_key

        # Match-level DBBR schedule is hand-based; update on terminal observation.
        if terminated:
            self._opp_raise_streak = 0
            self._opp_raise_ema *= 0.5
            showdown = "player_0_cards" in info and "player_1_cards" in info
            result = "win" if reward > 0 else "loss" if reward < 0 else "tie"
            summary = {
                "hand": info.get("hand_number"),
                "result": result,
                "reward": reward,
                "showdown": showdown,
                "board": info.get("community_cards"),
                "p0": info.get("player_0_cards"),
                "p1": info.get("player_1_cards"),
                "last_decision": self._last_decision,
            }
            self._log(f"[hand_end] {summary}")
            self._hand_counter += 1
            self._dbbr.maybe_update_model(self._hand_counter)
            event = self._exploration.record_hand_result(reward)
            if event == "triggered":
                self._log(f"[exploration] triggered {self._exploration.snapshot()}")
            elif event == "ended":
                self._log(f"[exploration] ended {self._exploration.snapshot()}")

