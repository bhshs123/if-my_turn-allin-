from agents.agent import Agent
from gym_env import PokerEnv
from submission.strategies.basic import remaining_card_pool
from submission.action import preflop_action, flop_action, discard_action, turn_action, river_action


class PlayerAgent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.action_types = PokerEnv.ActionType

    def __name__(self):
        return "PlayerAgent"

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
        street = observation["street"]
        valid_actions = observation["valid_actions"]
        my_cards = observation.get("my_cards", [])
        community_cards = observation.get("community_cards", [])
        min_raise = observation.get("min_raise", 1)
        max_raise = observation.get("max_raise", 100)

        # --- Pre-flop (street 0) ---
        if street == 0:
            return preflop_action(my_cards, remaining_card_pool, valid_actions, min_raise, max_raise)

        # --- Flop discard round (street 1) ---
        if street == 1 and valid_actions[self.action_types.DISCARD.value]:
            return discard_action(my_cards, community_cards, remaining_card_pool)

        # --- Flop betting (street 1 after discard) ---
        if street == 1:
            return flop_action(my_cards, community_cards, remaining_card_pool, valid_actions, min_raise, max_raise)

        # --- Turn (street 2) ---
        if street == 2:
            return turn_action(my_cards, community_cards, remaining_card_pool, valid_actions, min_raise, max_raise)

        # --- River (street 3) ---
        if street == 3:
            return river_action(my_cards, community_cards, remaining_card_pool, valid_actions, min_raise, max_raise)

        # Fallback
        return self.action_types.FOLD.value, 0, 0, 0

