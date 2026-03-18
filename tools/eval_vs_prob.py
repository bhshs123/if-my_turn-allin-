import sys
from collections import Counter, defaultdict

sys.path.insert(0, '.')

from gym_env import PokerEnv
from submission.player import PlayerAgent
from agents.prob_agent import ProbabilityAgent


def run_batch(hero_seat: int, hands: int = 1200):
    env = PokerEnv()
    hero = PlayerAgent(stream=False)
    opp = ProbabilityAgent(stream=False)

    total = 0
    wins = 0
    losses = 0
    ties = 0
    abs_loss = 0
    abs_win = 0

    node_action = defaultdict(Counter)
    node_seen = Counter()
    invalids = 0

    for _ in range(hands):
        obs_pair, info = env.reset()
        done = False
        reward = (0, 0)
        while not done:
            acting = int(obs_pair[0]["acting_agent"])
            if acting == hero_seat:
                obs = obs_pair[hero_seat]
                action = hero.act(obs, reward[hero_seat], done, False, info)
                d = getattr(hero, "_last_decision", {}) or {}
                anti = d.get("anti_predict") or {}
                node = anti.get("node")
                final = d.get("final")
                if node and final:
                    node_seen[node] += 1
                    node_action[node][final] += 1
            else:
                obs = obs_pair[acting]
                action = opp.act(obs, reward[acting], done, False, info)

            obs_pair, reward, done, _, info = env.step(action)
            if bool(info.get("invalid_action", False)):
                invalids += 1

        r = reward[hero_seat]
        total += r
        if r > 0:
            wins += 1
            abs_win += r
        elif r < 0:
            losses += 1
            abs_loss += -r
        else:
            ties += 1

    return {
        "hero_seat": hero_seat,
        "hands": hands,
        "total": total,
        "bb_per_hand": total / hands,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "avg_win": (abs_win / wins) if wins else 0.0,
        "avg_loss": (abs_loss / losses) if losses else 0.0,
        "node_seen": node_seen,
        "node_action": node_action,
        "invalids": invalids,
    }


def print_summary(res):
    print(f"seat={res['hero_seat']} hands={res['hands']} total={res['total']} bb/hand={res['bb_per_hand']:.4f} W/L/T={res['wins']}/{res['losses']}/{res['ties']} invalid={res['invalids']}")
    key_nodes = [
        "preflop_unopened",
        "flop_check_to_us",
        "turn_check_to_us",
        "river_check_to_us",
        "facing_turn_raise",
        "facing_river_raise",
    ]
    for n in key_nodes:
        total = int(res["node_seen"].get(n, 0))
        if total == 0:
            continue
        print(f"  {n}: {total}")
        for a, c in sorted(res["node_action"][n].items()):
            print(f"    {a}: {c} ({100.0*c/total:.1f}%)")


if __name__ == "__main__":
    r0 = run_batch(hero_seat=0, hands=1200)
    r1 = run_batch(hero_seat=1, hands=1200)
    print_summary(r0)
    print_summary(r1)
