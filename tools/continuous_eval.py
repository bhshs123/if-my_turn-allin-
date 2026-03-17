import argparse
import csv
import datetime as dt
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_match_bankroll(match_csv: Path):
    if not match_csv.exists():
        return None

    with match_csv.open("r", encoding="utf-8", newline="") as f:
        first = f.readline()
        if not first:
            return None

        if first.startswith("# Team"):
            header = f.readline().strip()
            if not header:
                return None
            fieldnames = [x.strip() for x in header.split(",")]
            reader = csv.DictReader(f, fieldnames=fieldnames)
        else:
            f.seek(0)
            reader = csv.DictReader(f)

        last_valid = None
        for row in reader:
            v = (row.get("team_0_bankroll") or "").strip()
            if v.lstrip("-").isdigit():
                last_valid = int(v)

        return last_valid


def run_one_match(python_exe: str, hands_per_match: int, stdout_path: Path, stderr_path: Path, timeout_sec: int):
    code = (
        "import run,match;"
        f"run.run_api_match=lambda *a,**k: match.run_api_match(*a,**k,num_hands={hands_per_match});"
        "run.main()"
    )
    with stdout_path.open("w", encoding="utf-8") as out_f, stderr_path.open("w", encoding="utf-8") as err_f:
        try:
            proc = subprocess.run(
                [python_exe, "-c", code],
                stdout=out_f,
                stderr=err_f,
                text=True,
                timeout=max(30, int(timeout_sec)),
            )
            return proc.returncode, False
        except subprocess.TimeoutExpired:
            err_f.write("\n[continuous_eval] match timeout reached\n")
            return 124, True


def main():
    parser = argparse.ArgumentParser(description="Continuous evaluation loop for poker bot.")
    parser.add_argument("--hours", type=float, default=4.0, help="Total wall-clock duration in hours.")
    parser.add_argument("--hands-per-match", type=int, default=24, help="Hands per match run.")
    parser.add_argument("--max-matches", type=int, default=0, help="Stop after N matches (0 means unlimited until time).")
    parser.add_argument("--out-dir", type=str, default="CSV/long_eval", help="Directory for archived match CSV files.")
    parser.add_argument("--match-timeout-sec", type=int, default=1100, help="Per-match timeout in seconds.")
    args = parser.parse_args()

    root = Path.cwd()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = out_dir / f"session_{ts}"
    session_dir.mkdir(parents=True, exist_ok=True)

    summary_path = session_dir / "summary.jsonl"
    live_path = root / "match.csv"

    deadline = time.time() + args.hours * 3600
    python_exe = sys.executable

    match_idx = 0
    wins = 0
    losses = 0
    ties = 0
    bankroll_sum = 0

    print(f"[continuous_eval] start session={session_dir}")
    print(f"[continuous_eval] config: hours={args.hours}, hands_per_match={args.hands_per_match}, max_matches={args.max_matches}")

    while time.time() < deadline:
        if args.max_matches > 0 and match_idx >= args.max_matches:
            break

        start = time.time()
        stdout_log = session_dir / f"match_{match_idx + 1:04d}.stdout.log"
        stderr_log = session_dir / f"match_{match_idx + 1:04d}.stderr.log"
        return_code, timed_out = run_one_match(
            python_exe,
            args.hands_per_match,
            stdout_path=stdout_log,
            stderr_path=stderr_log,
            timeout_sec=args.match_timeout_sec,
        )
        elapsed = round(time.time() - start, 2)

        match_idx += 1
        end_bankroll = parse_match_bankroll(live_path)

        if end_bankroll is None:
            outcome = "unknown"
        elif end_bankroll > 0:
            outcome = "win"
            wins += 1
        elif end_bankroll < 0:
            outcome = "loss"
            losses += 1
        else:
            outcome = "tie"
            ties += 1

        if end_bankroll is not None:
            bankroll_sum += end_bankroll

        archived_csv = session_dir / f"match_{match_idx:04d}.csv"
        if live_path.exists():
            shutil.copy2(live_path, archived_csv)

        record = {
            "match_idx": match_idx,
            "elapsed_sec": elapsed,
            "return_code": return_code,
            "timed_out": timed_out,
            "outcome": outcome,
            "team0_bankroll": end_bankroll,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "bankroll_sum": bankroll_sum,
            "bankroll_avg": round(bankroll_sum / max(1, (wins + losses + ties)), 2),
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        }
        with summary_path.open("a", encoding="utf-8") as sf:
            sf.write(json.dumps(record, ensure_ascii=True) + "\n")

        print(
            f"[continuous_eval] match={match_idx} rc={return_code} timeout={timed_out} outcome={outcome} "
            f"bankroll={end_bankroll} elapsed={elapsed}s "
            f"W/L/T={wins}/{losses}/{ties} avg={record['bankroll_avg']}"
        )

    print("[continuous_eval] done")
    print(f"[continuous_eval] summary: matches={match_idx}, W/L/T={wins}/{losses}/{ties}, bankroll_sum={bankroll_sum}")
    print(f"[continuous_eval] output={session_dir}")


if __name__ == "__main__":
    main()
