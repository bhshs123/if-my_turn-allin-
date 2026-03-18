from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


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
