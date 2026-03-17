import argparse
import json
import logging
import multiprocessing
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from match import run_api_match, bankrolls
from run import _wait_for_port, _wait_for_port_free, load_agent_class
import match as match_module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=3)
    parser.add_argument("--hands", type=int, default=250)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    with open("agent_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    bot0_class = load_agent_class(config["bot0"]["file_path"])
    bot1_class = load_agent_class(config["bot1"]["file_path"])

    for match_idx in range(1, args.matches + 1):
        logger.info("Starting batch match %s/%s", match_idx, args.matches)

        bankrolls[0] = 0
        bankrolls[1] = 0
        match_module.time_used_0 = 0.0
        match_module.time_used_1 = 0.0

        for port_cfg, name in [(config["bot0"]["port"], "bot0"), (config["bot1"]["port"], "bot1")]:
            port = int(port_cfg)
            if not _wait_for_port_free("0.0.0.0", port, timeout_sec=15.0):
                raise RuntimeError(f"Port {port} ({name}) is still in use")

        process0 = multiprocessing.Process(
            target=bot0_class.run,
            args=(True, config["bot0"]["port"]),
            kwargs={"player_id": config["bot0"]["player_id"]},
        )
        process1 = multiprocessing.Process(
            target=bot1_class.run,
            args=(True, config["bot1"]["port"]),
            kwargs={"player_id": config["bot1"]["player_id"]},
        )

        process0.start()
        process1.start()

        bot0_ready = _wait_for_port("127.0.0.1", int(config["bot0"]["port"]))
        bot1_ready = _wait_for_port("127.0.0.1", int(config["bot1"]["port"]))
        if not (bot0_ready and bot1_ready):
            p0_alive, p1_alive = process0.is_alive(), process1.is_alive()
            p0_code, p1_code = process0.exitcode, process1.exitcode
            process0.terminate()
            process1.terminate()
            process0.join()
            process1.join()
            raise RuntimeError(
                "Agent server startup failed. "
                f"bot0_ready={bot0_ready}, bot1_ready={bot1_ready}, "
                f"bot0_alive={p0_alive}, bot1_alive={p1_alive}, "
                f"bot0_exit={p0_code}, bot1_exit={p1_code}."
            )

        csv_path = os.path.join(args.output_dir, f"match_{match_idx:03d}.csv")
        try:
            result = run_api_match(
                f"http://localhost:{config['bot0']['port']}",
                f"http://localhost:{config['bot1']['port']}",
                logger,
                num_hands=args.hands,
                csv_path=csv_path,
                team_0_name=bot0_class.__name__,
                team_1_name=bot1_class.__name__,
            )
            logger.info("Match %s result: %s", match_idx, result)
        finally:
            process0.terminate()
            process1.terminate()
            process0.join()
            process1.join()
            time.sleep(1.0)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()