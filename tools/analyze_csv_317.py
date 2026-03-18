import csv
import glob
from collections import defaultdict


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
                "file": path,
                "hand": h,
                "delta": delta,
                "preflop_fold": has_action("Pre-Flop", "FOLD"),
                "turn_fold": has_action("Turn", "FOLD"),
                "river_fold": has_action("River", "FOLD"),
                "turn_call": has_action("Turn", "CALL"),
                "river_call": has_action("River", "CALL"),
                "turn_raise": has_action("Turn", "RAISE"),
                "river_raise": has_action("River", "RAISE"),
                "big_turn_call": has_action("Turn", "CALL", 12),
                "big_river_call": has_action("River", "CALL", 12),
                "allin_call": has_action("Turn", "CALL", 40) or has_action("River", "CALL", 40),
                "turn_call_amt": max_action_amount("Turn", "CALL"),
                "river_call_amt": max_action_amount("River", "CALL"),
            }
        )

    return transitions


def rate(rows, key):
    if not rows:
        return 0.0
    return round(sum(1 for r in rows if r[key]) / len(rows), 3)


def summarize(rows):
    loss = [x for x in rows if x["delta"] < 0]
    win = [x for x in rows if x["delta"] > 0]
    tie = [x for x in rows if x["delta"] == 0]

    return {
        "n": len(rows),
        "sum_delta": sum(x["delta"] for x in rows),
        "loss_n": len(loss),
        "win_n": len(win),
        "tie_n": len(tie),
        "loss_avg": round(sum(x["delta"] for x in loss) / len(loss), 2) if loss else 0.0,
        "win_avg": round(sum(x["delta"] for x in win) / len(win), 2) if win else 0.0,
        "loss_rates": {
            "preflop_fold": rate(loss, "preflop_fold"),
            "turn_fold": rate(loss, "turn_fold"),
            "river_fold": rate(loss, "river_fold"),
            "turn_call": rate(loss, "turn_call"),
            "river_call": rate(loss, "river_call"),
            "turn_raise": rate(loss, "turn_raise"),
            "river_raise": rate(loss, "river_raise"),
            "big_turn_call": rate(loss, "big_turn_call"),
            "big_river_call": rate(loss, "big_river_call"),
            "allin_call": rate(loss, "allin_call"),
        },
        "win_rates": {
            "preflop_fold": rate(win, "preflop_fold"),
            "turn_fold": rate(win, "turn_fold"),
            "river_fold": rate(win, "river_fold"),
            "turn_call": rate(win, "turn_call"),
            "river_call": rate(win, "river_call"),
            "turn_raise": rate(win, "turn_raise"),
            "river_raise": rate(win, "river_raise"),
            "big_turn_call": rate(win, "big_turn_call"),
            "big_river_call": rate(win, "big_river_call"),
            "allin_call": rate(win, "allin_call"),
        },
        "worst": sorted(rows, key=lambda x: x["delta"])[:20],
        "best": sorted(rows, key=lambda x: x["delta"], reverse=True)[:20],
    }


def main():
    files = sorted(glob.glob("CSV/3.17/*.csv"))
    if not files:
        print("No CSV files found in CSV/3.17")
        return

    all_rows = []
    per_file = {}
    for f in files:
        rows = parse_file(f)
        per_file[f] = summarize(rows)
        all_rows.extend(rows)

    s = summarize(all_rows)
    print("=== GLOBAL 3.17 SUMMARY ===")
    print(s)

    print("\n=== FILE RANKING (sum_delta) ===")
    ranked = sorted(per_file.items(), key=lambda kv: kv[1]["sum_delta"])
    for f, rs in ranked:
        print(f"{f}: sum_delta={rs['sum_delta']} n={rs['n']} loss_n={rs['loss_n']} win_n={rs['win_n']}")

    print("\n=== WORST 10 HAND TRANSITIONS ===")
    for row in s["worst"][:10]:
        print(row)


if __name__ == "__main__":
    main()
