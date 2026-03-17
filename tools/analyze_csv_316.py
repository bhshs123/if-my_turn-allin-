import csv
import glob
from collections import defaultdict, Counter


def to_int(s):
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    try:
        return int(s)
    except ValueError:
        return None


def parse_file(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        first_line = f.readline()
        if not first_line:
            return []
        if first_line.startswith("# Team"):
            header_line = f.readline()
            if not header_line:
                return []
            fieldnames = [x.strip() for x in header_line.strip().split(",")]
            reader = csv.DictReader(f, fieldnames=fieldnames)
        else:
            f.seek(0)
            reader = csv.DictReader(f)
        for r in reader:
            if to_int(r.get("hand_number")) is None:
                continue
            rows.append(r)
    by_hand = defaultdict(list)
    for r in rows:
        by_hand[to_int(r["hand_number"])].append(r)

    hands = sorted(by_hand)
    transitions = []
    prev_end = None

    for h in hands:
        hrows = by_hand[h]
        valid_br = [to_int(x.get("team_0_bankroll")) for x in hrows]
        valid_br = [x for x in valid_br if x is not None]
        if not valid_br:
            continue
        end_br = valid_br[-1]
        if prev_end is None:
            prev_end = end_br
            continue

        delta = end_br - prev_end
        prev_end = end_br

        our_rows = [x for x in hrows if str(x.get("active_team", "")).strip() == "0"]

        def has_action(street, action, min_amt=None):
            for rr in our_rows:
                if str(rr.get("street", "")).strip() != street:
                    continue
                if str(rr.get("action_type", "")).strip() != action:
                    continue
                if min_amt is not None:
                    a = to_int(rr.get("action_amount"))
                    if a is None or a < min_amt:
                        continue
                return True
            return False

        def max_action_amount(street, action):
            vals = []
            for rr in our_rows:
                if str(rr.get("street", "")).strip() != street:
                    continue
                if str(rr.get("action_type", "")).strip() != action:
                    continue
                a = to_int(rr.get("action_amount"))
                if a is not None:
                    vals.append(a)
            return max(vals) if vals else 0

        transitions.append(
            {
                "hand": h,
                "delta": delta,
                "river_call": has_action("River", "CALL"),
                "turn_call": has_action("Turn", "CALL"),
                "river_raise": has_action("River", "RAISE"),
                "turn_raise": has_action("Turn", "RAISE"),
                "big_river_call": has_action("River", "CALL", 12),
                "big_turn_call": has_action("Turn", "CALL", 10),
                "big_river_raise": has_action("River", "RAISE", 15),
                "big_turn_raise": has_action("Turn", "RAISE", 12),
                "preflop_raise": has_action("Pre-Flop", "RAISE"),
                "river_call_amt": max_action_amount("River", "CALL"),
                "turn_call_amt": max_action_amount("Turn", "CALL"),
            }
        )

    return transitions


def rate(rows, key):
    if not rows:
        return 0.0
    return round(sum(1 for r in rows if r[key]) / len(rows), 3)


def summarize(transitions):
    loss = [x for x in transitions if x["delta"] < 0]
    win = [x for x in transitions if x["delta"] > 0]

    result = {
        "n": len(transitions),
        "sum_delta": sum(x["delta"] for x in transitions),
        "loss_n": len(loss),
        "win_n": len(win),
        "loss_avg": round(sum(x["delta"] for x in loss) / len(loss), 2) if loss else 0.0,
        "win_avg": round(sum(x["delta"] for x in win) / len(win), 2) if win else 0.0,
        "loss_rates": {
            "river_call": rate(loss, "river_call"),
            "turn_call": rate(loss, "turn_call"),
            "river_raise": rate(loss, "river_raise"),
            "turn_raise": rate(loss, "turn_raise"),
            "big_river_call": rate(loss, "big_river_call"),
            "big_turn_call": rate(loss, "big_turn_call"),
            "big_river_raise": rate(loss, "big_river_raise"),
            "big_turn_raise": rate(loss, "big_turn_raise"),
            "preflop_raise": rate(loss, "preflop_raise"),
        },
        "win_rates": {
            "river_call": rate(win, "river_call"),
            "turn_call": rate(win, "turn_call"),
            "river_raise": rate(win, "river_raise"),
            "turn_raise": rate(win, "turn_raise"),
            "big_river_call": rate(win, "big_river_call"),
            "big_turn_call": rate(win, "big_turn_call"),
            "big_river_raise": rate(win, "big_river_raise"),
            "big_turn_raise": rate(win, "big_turn_raise"),
            "preflop_raise": rate(win, "preflop_raise"),
        },
        "worst": sorted(transitions, key=lambda x: x["delta"])[:12],
    }
    return result


def main():
    files = sorted(glob.glob("CSV/3.16/*.csv"))
    if not files:
        print("No CSV files found in CSV/3.16")
        return

    global_transitions = []
    per_file = {}

    for f in files:
        t = parse_file(f)
        per_file[f] = summarize(t)
        global_transitions.extend(t)

    global_summary = summarize(global_transitions)

    print("=== GLOBAL SUMMARY ===")
    print(global_summary)

    print("\n=== PER FILE SUMMARY ===")
    ranked = sorted(per_file.items(), key=lambda kv: kv[1]["sum_delta"])
    for f, s in ranked:
        print(f"{f}: sum_delta={s['sum_delta']} n={s['n']} loss_n={s['loss_n']} win_n={s['win_n']}")

    print("\n=== WORST FILE DETAILS ===")
    for f, s in ranked[:5]:
        print(f"\n[{f}]\nloss_rates={s['loss_rates']}\nwin_rates={s['win_rates']}")
        print("worst_hands:")
        for row in s["worst"][:8]:
            print(row)


if __name__ == "__main__":
    main()
