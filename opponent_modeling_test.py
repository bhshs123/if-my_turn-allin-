import math

from submission.opponent_modeling import (
    DBBRConfig,
    DBBROpponentModel,
    WeightShiftingUpdater,
    _normalize_probs,
)


def _obs(street=1, pot=20, opp_bet=10, legal=(1, 1, 1, 1, 0), last="None"):
    return {
        "street": street,
        "community_cards": [0, 9, 18, -1, -1],
        "pot_size": pot,
        "opp_bet": opp_bet,
        "blind_position": 0,
        "valid_actions": list(legal),
        "opp_last_action": last,
    }


def _assert_normalized(probs, tol=1e-6):
    s = sum(probs.values())
    assert abs(s - 1.0) < tol
    for v in probs.values():
        assert -tol <= v <= 1.0 + tol


def test_posterior_formula_matches_dirichlet_prior():
    cfg = DBBRConfig(n_prior=5.0, debug_logging=False)
    model = DBBROpponentModel(config=cfg)

    obs = _obs(legal=(1, 1, 0, 1, 0))  # FOLD, RAISE, CALL
    key = model.encoder.encode_public_state(obs, actor="opponent")
    legal = ("FOLD", "RAISE", "CALL")
    model.stats.register_node(key, legal)

    # Add counts: RAISE x3, CALL x1
    for _ in range(3):
        model.stats.record_action(key, "RAISE", legal)
    model.stats.record_action(key, "CALL", legal)

    p_star = model.baseline_policy.get_action_probs(key, legal_actions=legal)
    alpha = model.compute_posterior_action_probs(key)

    total = 4
    expected_raise = (p_star["RAISE"] * cfg.n_prior + 3) / (cfg.n_prior + total)
    expected_call = (p_star["CALL"] * cfg.n_prior + 1) / (cfg.n_prior + total)
    expected_fold = (p_star["FOLD"] * cfg.n_prior + 0) / (cfg.n_prior + total)
    expected = _normalize_probs({"FOLD": expected_fold, "RAISE": expected_raise, "CALL": expected_call})

    assert abs(alpha["RAISE"] - expected["RAISE"]) < 1e-9
    assert abs(alpha["CALL"] - expected["CALL"]) < 1e-9
    assert abs(alpha["FOLD"] - expected["FOLD"]) < 1e-9


def test_alpha_beta_sigma_gamma_are_normalized():
    model = DBBROpponentModel(config=DBBRConfig(debug_logging=False, min_obs_per_node=0))
    obs = _obs()

    for _ in range(5):
        model.observe_opponent_action(obs, "RAISE")
    for _ in range(2):
        model.observe_opponent_action(obs, "CALL")

    key = model.encoder.encode_public_state(obs, actor="opponent")
    model.build_opponent_model_for_node(obs, key)

    _assert_normalized(model.alpha[key])
    _assert_normalized(model.beta[key])
    _assert_normalized(model.gamma[key])
    for bucket_probs in model.sigma_hat[key].values():
        _assert_normalized(bucket_probs)


def test_weight_shifting_increases_target_action_when_alpha_above_gamma():
    updater = WeightShiftingUpdater()
    sigma_init = {
        "b0": {"RAISE": 0.1, "CALL": 0.9},
        "b1": {"RAISE": 0.2, "CALL": 0.8},
    }
    beta = {"b0": 0.5, "b1": 0.5}
    alpha = {"RAISE": 0.7, "CALL": 0.3}

    sigma_hat = updater.adjust_node_model_to_match_alpha(sigma_init, beta, alpha)
    gamma_raise = 0.5 * sigma_hat["b0"]["RAISE"] + 0.5 * sigma_hat["b1"]["RAISE"]
    assert gamma_raise > 0.15


def test_weight_shifting_decreases_target_action_when_alpha_below_gamma():
    updater = WeightShiftingUpdater()
    sigma_init = {
        "b0": {"RAISE": 0.8, "CALL": 0.2},
        "b1": {"RAISE": 0.9, "CALL": 0.1},
    }
    beta = {"b0": 0.4, "b1": 0.6}
    alpha = {"RAISE": 0.2, "CALL": 0.8}

    sigma_hat = updater.adjust_node_model_to_match_alpha(sigma_init, beta, alpha)
    gamma_raise = 0.4 * sigma_hat["b0"]["RAISE"] + 0.6 * sigma_hat["b1"]["RAISE"]
    assert gamma_raise < 0.86


def test_weight_shifting_exact_match_in_toy_2x2_case():
    updater = WeightShiftingUpdater()
    sigma_init = {
        "b0": {"A": 0.0, "B": 1.0},
        "b1": {"A": 1.0, "B": 0.0},
    }
    beta = {"b0": 0.5, "b1": 0.5}
    alpha = {"A": 0.75, "B": 0.25}

    sigma_hat = updater.adjust_node_model_to_match_alpha(sigma_init, beta, alpha, tolerance=1e-8)
    gamma_a = 0.5 * sigma_hat["b0"]["A"] + 0.5 * sigma_hat["b1"]["A"]
    gamma_b = 0.5 * sigma_hat["b0"]["B"] + 0.5 * sigma_hat["b1"]["B"]
    assert abs(gamma_a - 0.75) < 1e-6
    assert abs(gamma_b - 0.25) < 1e-6


def test_unseen_node_does_not_crash_and_falls_back_baseline():
    model = DBBROpponentModel(config=DBBRConfig(debug_logging=False, warmup_iters=0, update_interval=1))
    obs = _obs()
    legal = ["FOLD", "RAISE", "CHECK", "CALL"]
    action = model.select_action(obs, legal)
    assert action in legal


def test_warmup_and_update_schedule():
    cfg = DBBRConfig(debug_logging=False, warmup_iters=2, update_interval=1, min_obs_per_node=0)
    model = DBBROpponentModel(config=cfg)
    obs = _obs()

    # Seed at least one node with observations.
    for _ in range(3):
        model.observe_opponent_action(obs, "CALL")

    model.maybe_update_model(1)
    assert model.cached_exploit_policy is None  # warmup

    model.maybe_update_model(3)
    assert model.cached_exploit_policy is not None


def test_fallback_to_baseline_when_solver_fails():
    class FailingSolver:
        def solve(self, opponent_model):
            _ = opponent_model
            return None, 0.0

    cfg = DBBRConfig(debug_logging=False, warmup_iters=0, update_interval=1)
    model = DBBROpponentModel(config=cfg, solver=FailingSolver())
    obs = _obs()
    for _ in range(2):
        model.observe_opponent_action(obs, "RAISE")

    model.maybe_update_model(1)
    assert model.cached_exploit_policy is None


def test_integration_toy_over_raise_shifts_exploit_direction():
    cfg = DBBRConfig(debug_logging=False, warmup_iters=0, update_interval=1, min_obs_per_node=0)
    model = DBBROpponentModel(config=cfg)
    obs = _obs(legal=(1, 1, 1, 1, 0))
    key = model.encoder.encode_public_state(obs, actor="opponent")
    legal = ("FOLD", "RAISE", "CHECK", "CALL")

    # Opponent over-raises at this node.
    for _ in range(50):
        model.observe_opponent_action(obs, "RAISE")
    for _ in range(5):
        model.observe_opponent_action(obs, "CALL")

    model.maybe_update_model(1)
    assert model.cached_exploit_policy is not None

    base = model.baseline_policy.get_action_probs(key, legal_actions=legal)
    exploit = model.cached_exploit_policy.get_action_probs(key, legal)

    # Against an over-raising opponent, exploit policy should become more cautious.
    assert exploit["FOLD"] >= base["FOLD"] or exploit["CHECK"] >= base["CHECK"]
