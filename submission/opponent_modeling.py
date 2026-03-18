"""DBBR-WS opponent modeling for the 2026 poker bot project.

Implemented
- Public-history node encoding over observable information only.
- Online opponent action counting c[n,a] at public nodes.
- Posterior action probabilities alpha[n,a] with Dirichlet-style prior:
    alpha[n,a] = (p_star[n,a] * N_prior + c[n,a]) / (N_prior + sum_a' c[n,a'])
- Bucket abstraction and posterior bucket probabilities beta[n,b] with a practical
  parent/path recursion interface.
- DBBR-WS style weight-shifting updater (no LP/QP optimization):
  starts from sigma_star and shifts bucket action mass to match alpha.
- Periodic model rebuild and exploit-policy recomputation after warmup.
- Safety fallbacks to baseline for sparse/unseen/failure cases.

Approximations
- No full equilibrium table is available in this codebase; BaselinePolicy is a
  node-level prior that can optionally support bucket-conditioned probabilities.
- No exact global best-response solver exists; BestResponseSolver uses a
  lightweight exploit-policy approximation from the built opponent model.
- beta recursion uses stored parent links and parent sequence probabilities where
  available; otherwise it falls back to bucket prior h[b].

Configuration
- Use DBBRConfig to set warmup_iters (T), update_interval (k), n_prior (N_prior),
  tolerance, sparsity and confidence controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import numbers
import random


Action = str
PublicKey = str
BucketId = str


def _normalize_probs(probs: Mapping[str, float], eps: float = 1e-12) -> Dict[str, float]:
    total = sum(max(0.0, v) for v in probs.values())
    if total <= eps:
        n = max(1, len(probs))
        return {k: 1.0 / n for k in probs}
    out = {k: max(0.0, v) / total for k, v in probs.items()}
    # Numerical cleanup
    z = sum(out.values())
    if z <= eps:
        n = max(1, len(out))
        return {k: 1.0 / n for k in out}
    return {k: v / z for k, v in out.items()}


def _bucketize_value(x: int, step: int = 10, max_value: int = 200) -> int:
    capped = max(0, min(max_value, int(x)))
    return (capped // step) * step


def _action_id_to_name(action_id: int) -> Action:
    mapping = {
        0: "FOLD",
        1: "RAISE",
        2: "CHECK",
        3: "CALL",
        4: "DISCARD",
    }
    return mapping.get(action_id, "UNKNOWN")


def _legal_action_names(valid_actions: Sequence[int | bool]) -> Tuple[Action, ...]:
    names: List[Action] = []
    for i, is_valid in enumerate(valid_actions):
        if bool(is_valid):
            names.append(_action_id_to_name(i))
    return tuple(names)


@dataclass
class DBBRConfig:
    warmup_iters: int = 1000
    update_interval: int = 50
    n_prior: float = 5.0
    tolerance: float = 1e-6
    use_exploit_threshold: bool = True
    exploit_threshold: float = 0.0
    min_obs_per_node: int = 0
    deterministic_action_selection: bool = False
    debug_logging: bool = True
    enable_exploitation: bool = True
    num_buckets: int = 6


class PublicStateEncoder:
    """Canonical encoder for public decision contexts.

    Key includes only observable information:
    - street
    - public board cards (revealed only)
    - pot bucket
    - opponent bet bucket
    - position/blind marker
    - legal action set
    - simple action-history abstraction (opp_last_action)
    - actor role tag (opponent/self), to separate decision contexts
    """

    def encode_public_state(self, observation: Mapping[str, Any], actor: str = "opponent") -> PublicKey:
        street = int(observation.get("street", -1))
        board = tuple(int(c) for c in observation.get("community_cards", []) if isinstance(c, numbers.Integral) and c >= 0)
        pot_bucket = _bucketize_value(int(observation.get("pot_size", 0)), step=10, max_value=200)
        opp_bet_bucket = _bucketize_value(int(observation.get("opp_bet", 0)), step=10, max_value=100)
        blind_pos = int(observation.get("blind_position", -1))
        legal = _legal_action_names(observation.get("valid_actions", []))
        last = str(observation.get("opp_last_action", "None"))
        # Deterministic serializable key.
        return (
            f"actor={actor}|street={street}|board={board}|pot={pot_bucket}|opp_bet={opp_bet_bucket}|"
            f"blind={blind_pos}|last={last}|legal={legal}"
        )


class OpponentStats:
    """Stores c[n,a], legal-actions per node, and parent transition metadata."""

    def __init__(self) -> None:
        self.counts: Dict[Tuple[PublicKey, Action], int] = {}
        self.legal_actions: Dict[PublicKey, Tuple[Action, ...]] = {}
        self.parent_info: Dict[PublicKey, Tuple[Optional[PublicKey], Optional[Action]]] = {}

    def register_node(
        self,
        public_key: PublicKey,
        legal_actions: Sequence[Action],
        parent_key: Optional[PublicKey] = None,
        parent_action: Optional[Action] = None,
    ) -> None:
        if public_key not in self.legal_actions:
            self.legal_actions[public_key] = tuple(legal_actions)
        if public_key not in self.parent_info:
            self.parent_info[public_key] = (parent_key, parent_action)

    def record_action(self, public_key: PublicKey, action: Action, legal_actions: Sequence[Action]) -> None:
        self.register_node(public_key, legal_actions)
        if action not in self.legal_actions.get(public_key, ()):  # keep robust on malformed labels
            return
        k = (public_key, action)
        self.counts[k] = self.counts.get(k, 0) + 1

    def node_action_count(self, public_key: PublicKey, action: Action) -> int:
        return self.counts.get((public_key, action), 0)

    def node_total_count(self, public_key: PublicKey) -> int:
        acts = self.legal_actions.get(public_key, ())
        return sum(self.counts.get((public_key, a), 0) for a in acts)

    def iter_nodes(self) -> List[PublicKey]:
        return list(self.legal_actions.keys())


class BaselinePolicy:
    """Interface for p_star[n,a] (or optional sigma_star[n,b,a])."""

    def get_action_probs(
        self,
        public_state_key: PublicKey,
        bucket_id: Optional[BucketId] = None,
        legal_actions: Optional[Sequence[Action]] = None,
    ) -> Dict[Action, float]:
        raise NotImplementedError


class HeuristicBaselinePolicy(BaselinePolicy):
    """Node-only baseline with optional bucket conditioning.

    Practical fallback when no equilibrium table exists.
    """

    def _street_from_key(self, key: PublicKey) -> int:
        # key format contains "street=<int>|"
        try:
            seg = [s for s in key.split("|") if s.startswith("street=")][0]
            return int(seg.split("=")[1])
        except Exception:
            return 0

    def get_action_probs(
        self,
        public_state_key: PublicKey,
        bucket_id: Optional[BucketId] = None,
        legal_actions: Optional[Sequence[Action]] = None,
    ) -> Dict[Action, float]:
        legal = list(legal_actions or ())
        if not legal:
            return {}

        street = self._street_from_key(public_state_key)
        # Node-only baseline prior; optionally perturb by bucket profile.
        prefs: Dict[Action, float] = {a: 0.05 for a in legal}

        if "CHECK" in prefs:
            prefs["CHECK"] += 0.30
        if "CALL" in prefs:
            prefs["CALL"] += 0.25
        if "RAISE" in prefs:
            prefs["RAISE"] += 0.20 + 0.05 * street
        if "FOLD" in prefs:
            prefs["FOLD"] += 0.15

        # Optional bucket-conditioned flavor.
        if bucket_id is not None and "RAISE" in prefs:
            if bucket_id in ("made_strong", "nuts"):
                prefs["RAISE"] += 0.25
            if bucket_id in ("very_weak",):
                prefs["RAISE"] = max(0.01, prefs["RAISE"] - 0.15)
        return _normalize_probs(prefs)


class Bucketizer:
    """Simplified bucket abstraction over hidden opponent private states.

    Buckets are hand-strength / draw classes estimated from public information.
    """

    DEFAULT_BUCKETS: Tuple[BucketId, ...] = (
        "very_weak",
        "weak_draw",
        "pairish",
        "strong_draw",
        "made_strong",
        "nuts",
    )

    def enumerate_consistent_buckets(self, public_state: Mapping[str, Any]) -> List[BucketId]:
        _ = public_state
        return list(self.DEFAULT_BUCKETS)

    def get_bucket_distribution(self, public_state: Mapping[str, Any], actor: str = "opponent") -> Dict[BucketId, float]:
        _ = actor
        street = int(public_state.get("street", 0))
        pot = int(public_state.get("pot_size", 0))
        opp_bet = int(public_state.get("opp_bet", 0))
        # Heuristic prior mass: later streets and larger bets move mass toward strong buckets.
        aggr = min(1.0, opp_bet / 100.0)
        depth = min(1.0, street / 3.0)
        pressure = min(1.0, pot / 100.0)

        base = {
            "very_weak": 0.24,
            "weak_draw": 0.22,
            "pairish": 0.22,
            "strong_draw": 0.16,
            "made_strong": 0.12,
            "nuts": 0.04,
        }
        # Shift based on context.
        base["very_weak"] -= 0.10 * aggr
        base["weak_draw"] -= 0.04 * aggr
        base["strong_draw"] += 0.05 * depth
        base["made_strong"] += 0.08 * aggr + 0.06 * pressure
        base["nuts"] += 0.03 * aggr + 0.03 * depth
        return _normalize_probs(base)


class WeightShiftingUpdater:
    """DBBR-WS updater that adjusts sigma_hat to match alpha targets.

    Symbols mapping:
    - sigma_init[n,b,a]  : initial baseline policy at node/bucket/action
    - beta[n,b]          : posterior bucket weights
    - alpha[n,a]         : posterior action frequencies from observations + prior
    - gamma[n,a]         : weighted action probs under evolving sigma_hat
    """

    def __init__(self, tolerance: float = 1e-6, eps: float = 1e-12) -> None:
        self.tolerance = tolerance
        self.eps = eps

    def _weighted_gamma(
        self,
        sigma: Mapping[BucketId, Mapping[Action, float]],
        beta: Mapping[BucketId, float],
        actions: Sequence[Action],
    ) -> Dict[Action, float]:
        gamma = {a: 0.0 for a in actions}
        for b, w in beta.items():
            sb = sigma.get(b, {})
            for a in actions:
                gamma[a] += w * float(sb.get(a, 0.0))
        return _normalize_probs(gamma)

    def _remove_mass_stable(
        self,
        bucket_probs: MutableMapping[Action, float],
        exclude_action: Action,
        amount: float,
    ) -> Dict[Action, float]:
        removed: Dict[Action, float] = {}
        others = [a for a in bucket_probs if a != exclude_action]
        # Stable deterministic order: lowest-prob actions first.
        others.sort(key=lambda a: (bucket_probs[a], a))
        remain = amount
        for a in others:
            if remain <= self.eps:
                break
            take = min(bucket_probs[a], remain)
            bucket_probs[a] -= take
            removed[a] = take
            remain -= take
        # If not enough mass due to numerical drift, trim from exclude action back.
        if remain > self.eps:
            bucket_probs[exclude_action] = max(0.0, bucket_probs[exclude_action] - remain)
        return removed

    def _add_mass_stable(
        self,
        bucket_probs: MutableMapping[Action, float],
        include_action: Action,
        amount: float,
        baseline_bucket: Mapping[Action, float],
    ) -> Dict[Action, float]:
        added: Dict[Action, float] = {}
        others = [a for a in bucket_probs if a != include_action]
        # Prefer adding to actions with higher baseline prob.
        others.sort(key=lambda a: (-baseline_bucket.get(a, 0.0), a))
        remain = amount
        for a in others:
            if remain <= self.eps:
                break
            cap = 1.0 - bucket_probs[a]
            inc = min(cap, remain)
            bucket_probs[a] += inc
            added[a] = inc
            remain -= inc
        if remain > self.eps and others:
            # Put any tiny remainder to first stable action.
            a0 = others[0]
            bucket_probs[a0] += remain
            added[a0] = added.get(a0, 0.0) + remain
        return added

    def adjust_node_model_to_match_alpha(
        self,
        sigma_init: Dict[BucketId, Dict[Action, float]],
        beta: Dict[BucketId, float],
        alpha_target: Dict[Action, float],
        tolerance: float = 1e-6,
    ) -> Dict[BucketId, Dict[Action, float]]:
        tol = max(tolerance, self.tolerance)
        # Defensive normalization.
        beta = _normalize_probs(beta)
        alpha_target = _normalize_probs(alpha_target)

        sigma: Dict[BucketId, Dict[Action, float]] = {
            b: _normalize_probs(dict(ap)) for b, ap in sigma_init.items()
        }
        actions = list(alpha_target.keys())

        # Ensure every bucket has all actions.
        for b in list(sigma.keys()):
            for a in actions:
                sigma[b].setdefault(a, 0.0)
            sigma[b] = _normalize_probs(sigma[b])

        gamma = self._weighted_gamma(sigma, beta, actions)

        # Multiple passes until close enough or capped iterations.
        for _ in range(8):
            diffs = {a: alpha_target[a] - gamma.get(a, 0.0) for a in actions}
            max_dev = max(abs(v) for v in diffs.values()) if diffs else 0.0
            if max_dev <= tol:
                break

            ordered_actions = sorted(actions, key=lambda a: abs(diffs[a]), reverse=True)

            for target_action in ordered_actions:
                need = alpha_target[target_action] - gamma.get(target_action, 0.0)
                if abs(need) <= tol:
                    continue

                if need > 0.0:
                    # Increase target_action: favor buckets already preferring it.
                    bucket_order = sorted(sigma.keys(), key=lambda b: sigma[b].get(target_action, 0.0), reverse=True)
                    for b in bucket_order:
                        w = beta.get(b, 0.0)
                        if w <= self.eps:
                            continue
                        req = alpha_target[target_action] - gamma[target_action]
                        if req <= tol:
                            break
                        max_inc = 1.0 - sigma[b][target_action]
                        delta = min(max_inc, req / max(w, self.eps))
                        if delta <= self.eps:
                            continue
                        sigma[b][target_action] += delta
                        removed = self._remove_mass_stable(sigma[b], target_action, delta)
                        sigma[b] = _normalize_probs(sigma[b])
                        # Update gamma incrementally.
                        gamma[target_action] += w * delta
                        for a_other, rm in removed.items():
                            gamma[a_other] -= w * rm
                        gamma = _normalize_probs(gamma)
                else:
                    # Decrease target_action: remove from buckets overplaying it most.
                    bucket_order = sorted(sigma.keys(), key=lambda b: sigma[b].get(target_action, 0.0), reverse=True)
                    for b in bucket_order:
                        w = beta.get(b, 0.0)
                        if w <= self.eps:
                            continue
                        req = gamma[target_action] - alpha_target[target_action]
                        if req <= tol:
                            break
                        max_dec = sigma[b][target_action]
                        delta = min(max_dec, req / max(w, self.eps))
                        if delta <= self.eps:
                            continue
                        sigma[b][target_action] -= delta
                        added = self._add_mass_stable(sigma[b], target_action, delta, sigma_init.get(b, {}))
                        sigma[b] = _normalize_probs(sigma[b])
                        gamma[target_action] -= w * delta
                        for a_other, inc in added.items():
                            gamma[a_other] += w * inc
                        gamma = _normalize_probs(gamma)

        # Final renorm and sanity checks.
        for b in sigma:
            sigma[b] = _normalize_probs(sigma[b])
            assert abs(sum(sigma[b].values()) - 1.0) < 1e-6 + tol

        alpha_target = _normalize_probs(alpha_target)
        gamma = self._weighted_gamma(sigma, beta, actions)
        assert abs(sum(alpha_target.values()) - 1.0) < 1e-6 + tol
        assert abs(sum(gamma.values()) - 1.0) < 1e-6 + tol

        return sigma


class ExploitPolicy:
    """Action-probability policy container with baseline fallback."""

    def __init__(self, mapping: Dict[PublicKey, Dict[Action, float]], baseline: BaselinePolicy) -> None:
        self._mapping = mapping
        self._baseline = baseline

    def get_action_probs(
        self,
        state_key: PublicKey,
        legal_actions: Sequence[Action],
    ) -> Dict[Action, float]:
        if state_key in self._mapping:
            probs = {a: self._mapping[state_key].get(a, 0.0) for a in legal_actions}
            return _normalize_probs(probs)
        return self._baseline.get_action_probs(state_key, legal_actions=legal_actions)


class BestResponseSolver:
    """Approximate exploit-policy computation against sigma_hat.

    Since no exact solver exists in this codebase, we use a conservative node-local
    response heuristic: move mass away from actions the opponent model heavily
    punishes and toward actions likely to exploit predicted folds/passivity.
    """

    def solve(self, opponent_model: "DBBROpponentModel") -> Tuple[Optional[ExploitPolicy], float]:
        mapping: Dict[PublicKey, Dict[Action, float]] = {}
        gain_proxy = 0.0

        for key, legal in opponent_model.stats.legal_actions.items():
            legal_actions = list(legal)
            if not legal_actions:
                continue

            base = opponent_model.baseline_policy.get_action_probs(key, legal_actions=legal_actions)
            gamma = opponent_model.gamma.get(key)
            if not gamma:
                mapping[key] = base
                continue

            # Heuristic exploit signal from predicted opponent action frequencies.
            p_fold = gamma.get("FOLD", 0.0)
            p_raise = gamma.get("RAISE", 0.0)
            p_call = gamma.get("CALL", 0.0)
            p_check = gamma.get("CHECK", 0.0)

            out = dict(base)
            if "RAISE" in out:
                out["RAISE"] += 0.35 * p_fold - 0.15 * p_raise
            if "CHECK" in out:
                out["CHECK"] += 0.20 * p_raise + 0.10 * p_call
            if "CALL" in out:
                out["CALL"] += 0.10 * p_check - 0.10 * p_raise
            if "FOLD" in out:
                out["FOLD"] += 0.25 * p_raise - 0.05 * p_fold

            out = _normalize_probs({a: out.get(a, 0.0) for a in legal_actions})
            mapping[key] = out

            # Proxy: L1 distance from baseline as a rough "deviation utility" signal.
            gain_proxy += 0.5 * sum(abs(out.get(a, 0.0) - base.get(a, 0.0)) for a in legal_actions)

        if not mapping:
            return None, 0.0
        return ExploitPolicy(mapping, opponent_model.baseline_policy), gain_proxy


class DBBROpponentModel:
    """DBBR high-level loop and model storage.

    Paper symbols represented by fields:
    - c[n,a]                    -> stats.counts
    - alpha[n,a]                -> alpha
    - beta[n,b]                 -> beta
    - sigma_star[n,b,a]         -> sigma_star
    - sigma_hat[n,b,a]          -> sigma_hat
    - gamma[n,a]                -> gamma
    - sequence_prob[parent,b,a] -> sequence_prob
    - T, k, N_prior             -> config.warmup_iters, config.update_interval, config.n_prior
    """

    def __init__(
        self,
        config: Optional[DBBRConfig] = None,
        encoder: Optional[PublicStateEncoder] = None,
        stats: Optional[OpponentStats] = None,
        bucketizer: Optional[Bucketizer] = None,
        baseline_policy: Optional[BaselinePolicy] = None,
        updater: Optional[WeightShiftingUpdater] = None,
        solver: Optional[BestResponseSolver] = None,
    ) -> None:
        self.config = config or DBBRConfig()
        self.encoder = encoder or PublicStateEncoder()
        self.stats = stats or OpponentStats()
        self.bucketizer = bucketizer or Bucketizer()
        self.baseline_policy = baseline_policy or HeuristicBaselinePolicy()
        self.updater = updater or WeightShiftingUpdater(tolerance=self.config.tolerance)
        self.solver = solver or BestResponseSolver()

        self.alpha: Dict[PublicKey, Dict[Action, float]] = {}
        self.beta: Dict[PublicKey, Dict[BucketId, float]] = {}
        self.sigma_star: Dict[PublicKey, Dict[BucketId, Dict[Action, float]]] = {}
        self.sigma_hat: Dict[PublicKey, Dict[BucketId, Dict[Action, float]]] = {}
        self.gamma: Dict[PublicKey, Dict[Action, float]] = {}
        self.sequence_prob: Dict[PublicKey, Dict[BucketId, Dict[Action, float]]] = {}

        self.cached_exploit_policy: Optional[ExploitPolicy] = None
        self.iteration_counter: int = 0
        self._public_states: Dict[PublicKey, Dict[str, Any]] = {}
        self._last_report: Dict[str, Any] = {}

    def _log(self, msg: str) -> None:
        if self.config.debug_logging:
            print(f"[DBBR] {msg}")

    # ------------------------------------------------------------------
    # Required public methods
    # ------------------------------------------------------------------
    def observe_opponent_action(
        self,
        public_state: Mapping[str, Any],
        action: Action,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        public_key = self.encoder.encode_public_state(public_state, actor="opponent")
        legal = _legal_action_names(public_state.get("valid_actions", []))
        parent_key = metadata.get("parent_public_key") if metadata else None
        parent_action = metadata.get("parent_action") if metadata else None

        self.stats.register_node(public_key, legal, parent_key=parent_key, parent_action=parent_action)
        self.stats.record_action(public_key, action, legal)

        # Keep the latest public state snapshot for rebuilding model.
        self._public_states[public_key] = dict(public_state)

        self._log(f"observe: key={public_key} action={action} total={self.stats.node_total_count(public_key)}")

    def maybe_update_model(self, iteration_idx: int) -> None:
        self.iteration_counter = int(iteration_idx)
        if self.iteration_counter <= self.config.warmup_iters:
            self.cached_exploit_policy = None
            self._log(f"warmup iteration={self.iteration_counter}: baseline only")
            return
        if (self.iteration_counter - self.config.warmup_iters) % max(1, self.config.update_interval) != 0:
            return

        self._log(f"rebuild model at iteration={self.iteration_counter}")
        self.build_full_opponent_model()
        self.compute_exploit_policy()

    def compute_posterior_action_probs(self, public_key: PublicKey) -> Dict[Action, float]:
        legal = self.stats.legal_actions.get(public_key, ())
        if not legal:
            return {}

        p_star = self.baseline_policy.get_action_probs(public_key, legal_actions=legal)
        total_count = self.stats.node_total_count(public_key)
        denom = self.config.n_prior + total_count
        out: Dict[Action, float] = {}
        for a in legal:
            c = self.stats.node_action_count(public_key, a)
            out[a] = (p_star.get(a, 0.0) * self.config.n_prior + c) / max(1e-12, denom)

        out = _normalize_probs(out)
        self.alpha[public_key] = out
        return out

    def compute_posterior_bucket_probs(
        self,
        public_state: Mapping[str, Any],
        public_key: PublicKey,
    ) -> Dict[BucketId, float]:
        h = self.bucketizer.get_bucket_distribution(public_state, actor="opponent")
        h = _normalize_probs(h)

        parent_key, parent_action = self.stats.parent_info.get(public_key, (None, None))
        if parent_key is None or parent_action is None or parent_key not in self.sequence_prob:
            self.beta[public_key] = h
            return h

        parent_seq = self.sequence_prob[parent_key]
        weighted: Dict[BucketId, float] = {}
        for b, hb in h.items():
            s = parent_seq.get(b, {}).get(parent_action, 0.0)
            weighted[b] = hb * s
        weighted = _normalize_probs(weighted)
        self.beta[public_key] = weighted
        return weighted

    def build_opponent_model_for_node(
        self,
        public_state: Mapping[str, Any],
        public_key: PublicKey,
    ) -> None:
        legal = self.stats.legal_actions.get(public_key, ())
        if not legal:
            return

        # Sparse node safety: stay near baseline.
        total_obs = self.stats.node_total_count(public_key)
        alpha = self.compute_posterior_action_probs(public_key)
        beta = self.compute_posterior_bucket_probs(public_state, public_key)

        buckets = self.bucketizer.enumerate_consistent_buckets(public_state)
        sigma_init: Dict[BucketId, Dict[Action, float]] = {}
        for b in buckets:
            p_b = self.baseline_policy.get_action_probs(public_key, bucket_id=b, legal_actions=legal)
            sigma_init[b] = _normalize_probs({a: p_b.get(a, 0.0) for a in legal})

        self.sigma_star[public_key] = {b: dict(ap) for b, ap in sigma_init.items()}

        if total_obs < self.config.min_obs_per_node:
            sigma_hat = sigma_init
        else:
            sigma_hat = self.updater.adjust_node_model_to_match_alpha(
                sigma_init=sigma_init,
                beta=beta,
                alpha_target=alpha,
                tolerance=self.config.tolerance,
            )

        self.sigma_hat[public_key] = sigma_hat

        # gamma[n,a] = sum_b beta[n,b] * sigma_hat[n,b,a]
        gamma = {a: 0.0 for a in legal}
        for b, bw in beta.items():
            for a in legal:
                gamma[a] += bw * sigma_hat.get(b, {}).get(a, 0.0)
        self.gamma[public_key] = _normalize_probs(gamma)

        # sequence_prob store for child beta recursion.
        self.sequence_prob[public_key] = {
            b: {a: sigma_hat[b].get(a, 0.0) for a in legal}
            for b in sigma_hat
        }

        max_dev = max(abs(alpha[a] - self.gamma[public_key].get(a, 0.0)) for a in legal) if legal else 0.0
        self._log(
            f"node={public_key} obs={total_obs} alpha={alpha} beta={beta} "
            f"gamma={self.gamma[public_key]} max|a-g|={max_dev:.5f}"
        )

    def build_full_opponent_model(self) -> None:
        for key in self.stats.iter_nodes():
            state = self._public_states.get(key)
            if state is None:
                continue
            self.build_opponent_model_for_node(state, key)

    def compute_exploit_policy(self) -> None:
        if not self.config.enable_exploitation:
            self.cached_exploit_policy = None
            self._log("exploitation disabled by config")
            return

        policy, gain = self.solver.solve(self)
        if policy is None:
            self.cached_exploit_policy = None
            self._log("exploit policy unavailable; fallback baseline")
            return

        if self.config.use_exploit_threshold and gain < self.config.exploit_threshold:
            self.cached_exploit_policy = None
            self._log(f"gain {gain:.4f} < threshold {self.config.exploit_threshold:.4f}; fallback baseline")
            return

        self.cached_exploit_policy = policy
        self._last_report = {
            "iteration": self.iteration_counter,
            "nodes": len(self.stats.iter_nodes()),
            "gain_proxy": gain,
        }
        self._log(f"exploit policy recomputed: {self._last_report}")

    def select_action(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Action],
    ) -> Optional[Action]:
        key_self = self.encoder.encode_public_state(observation, actor="self")
        key_opp = self.encoder.encode_public_state(observation, actor="opponent")
        if not legal_actions:
            return None

        if self.iteration_counter <= self.config.warmup_iters:
            probs = self.baseline_policy.get_action_probs(key_self, legal_actions=legal_actions)
            self._log("select_action: baseline (warmup)")
        elif self.cached_exploit_policy is None:
            probs = self.baseline_policy.get_action_probs(key_self, legal_actions=legal_actions)
            self._log("select_action: baseline (no exploit policy)")
        else:
            # Opponent statistics are built on actor=opponent public nodes. For action-time
            # selection we try both keys and use the one available in exploit mapping.
            if key_self in self.cached_exploit_policy._mapping:
                probs = self.cached_exploit_policy.get_action_probs(key_self, legal_actions=legal_actions)
            elif key_opp in self.cached_exploit_policy._mapping:
                probs = self.cached_exploit_policy.get_action_probs(key_opp, legal_actions=legal_actions)
            else:
                probs = self.baseline_policy.get_action_probs(key_self, legal_actions=legal_actions)
            self._log("select_action: exploit policy")

        probs = _normalize_probs({a: probs.get(a, 0.0) for a in legal_actions})
        if self.config.deterministic_action_selection:
            return max(probs, key=probs.get)

        r = random.random()
        cdf = 0.0
        for a, p in probs.items():
            cdf += p
            if r <= cdf:
                return a
        return legal_actions[-1]

    def report_top_nodes(self, top_k: int = 10) -> Dict[str, Any]:
        nodes = []
        for key in self.stats.iter_nodes():
            total = self.stats.node_total_count(key)
            alpha = self.alpha.get(key, {})
            gamma = self.gamma.get(key, {})
            dev = max((abs(alpha.get(a, 0.0) - gamma.get(a, 0.0)) for a in alpha), default=0.0)
            nodes.append((total, dev, key))
        nodes.sort(reverse=True)
        return {
            "summary": self._last_report,
            "top_nodes": [
                {"total_obs": t, "max_dev": d, "key": k}
                for t, d, k in nodes[:top_k]
            ],
        }


class DBBRAgentMixin:
    """Optional mixin helper for agents wanting direct DBBR orchestration."""

    def __init__(self, dbbr_model: Optional[DBBROpponentModel] = None) -> None:
        self.dbbr_model = dbbr_model or DBBROpponentModel()

    def observe_opponent_action(self, public_state: Mapping[str, Any], action: Action, metadata: Optional[Mapping[str, Any]] = None) -> None:
        self.dbbr_model.observe_opponent_action(public_state, action, metadata=metadata)

    def maybe_update_model(self, iteration_idx: int) -> None:
        self.dbbr_model.maybe_update_model(iteration_idx)

    def compute_exploit_policy(self) -> None:
        self.dbbr_model.compute_exploit_policy()

    def select_action(self, state: Mapping[str, Any], legal_actions: Sequence[Action]) -> Optional[Action]:
        return self.dbbr_model.select_action(state, legal_actions)
