import os

from agents.agent import Agent
from gym_env import PokerEnv
from submission.strategies.basic import remaining_card_pool
from submission.opponent_modeling import DBBRConfig, DBBROpponentModel
from submission.action import preflop_action, flop_action, discard_action, turn_action, river_action


class PlayerAgent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.action_types = PokerEnv.ActionType
        raw_player_id = str(os.getenv("PLAYER_ID", "0"))
        if raw_player_id.lower().startswith("bot") and raw_player_id[-1].isdigit():
            self._my_player_id = int(raw_player_id[-1])
        elif raw_player_id.isdigit():
            self._my_player_id = int(raw_player_id)
        else:
            self._my_player_id = 0
        self._hand_counter = 0

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
                debug_logging=True,
                enable_exploitation=True,
            )
        )

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

    def _apply_action_override(self, base_action, chosen_action_name, min_raise, max_raise, observation):
        if chosen_action_name is None:
            return base_action

        base_type, _, keep_1, keep_2 = base_action
        chosen_id = self._action_name_to_id(chosen_action_name)
        if chosen_id == base_type:
            return base_action

        # Keep fold/call/check decisions EV-driven; DBBR exploit can tune aggression,
        # but should not force extra folds that look unintelligent.
        if chosen_action_name == "FOLD":
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

        # --- Pre-flop (street 0) ---
        if street == 0:
            base_action = preflop_action(
                my_cards, remaining_card_pool, valid_actions, min_raise, max_raise,
                pot_size=pot_size, my_bet=my_bet, opp_bet=opp_bet, opp_action_probs=opp_action_probs,
            )
            chosen = self._dbbr.select_action(observation, legal_action_names)
            return self._apply_action_override(base_action, chosen, min_raise, max_raise, observation)

        # --- Flop discard round (street 1) ---
        if street == 1 and valid_actions[self.action_types.DISCARD.value]:
            return discard_action(my_cards, community_cards, remaining_card_pool, dead_cards=dead_cards)

        # --- Flop betting (street 1 after discard) ---
        if street == 1:
            base_action = flop_action(
                my_cards, community_cards, remaining_card_pool, valid_actions, min_raise, max_raise,
                dead_cards=dead_cards, pot_size=pot_size, my_bet=my_bet, opp_bet=opp_bet, opp_action_probs=opp_action_probs,
            )
            chosen = self._dbbr.select_action(observation, legal_action_names)
            return self._apply_action_override(base_action, chosen, min_raise, max_raise, observation)

        # --- Turn (street 2) ---
        if street == 2:
            base_action = turn_action(
                my_cards, community_cards, remaining_card_pool, valid_actions, min_raise, max_raise,
                dead_cards=dead_cards, pot_size=pot_size, my_bet=my_bet, opp_bet=opp_bet, opp_action_probs=opp_action_probs,
            )
            chosen = self._dbbr.select_action(observation, legal_action_names)
            return self._apply_action_override(base_action, chosen, min_raise, max_raise, observation)

        # --- River (street 3) ---
        if street == 3:
            base_action = river_action(
                my_cards, community_cards, remaining_card_pool, valid_actions, min_raise, max_raise,
                dead_cards=dead_cards, pot_size=pot_size, my_bet=my_bet, opp_bet=opp_bet, opp_action_probs=opp_action_probs,
            )
            chosen = self._dbbr.select_action(observation, legal_action_names)
            return self._apply_action_override(base_action, chosen, min_raise, max_raise, observation)

        # Fallback
        return self.action_types.FOLD.value, 0, 0, 0

    def observe(self, observation, reward, terminated, truncated, info) -> None:
        _ = reward, truncated, info
        # Before opponent acts, cache the public state; action label will arrive as opp_last_action.
        if int(observation.get("acting_agent", -1)) != self._my_player_id:
            self._pending_opp_state = dict(observation)
            self._pending_opp_parent_key = self._last_opp_public_key

        # Match-level DBBR schedule is hand-based; update on terminal observation.
        if terminated:
            self._hand_counter += 1
            self._dbbr.maybe_update_model(self._hand_counter)

