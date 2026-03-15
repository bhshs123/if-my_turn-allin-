import logging
import multiprocessing
import json
import importlib
import socket
import sys
import time

from match import run_api_match


def _wait_for_port_free(host: str, port: int, timeout_sec: float = 15.0) -> bool:
    """Wait until a TCP port is free to bind (not already in use)."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return True  # bind succeeded — port is free
            except OSError:
                time.sleep(0.5)
    return False


def _wait_for_port(host: str, port: int, timeout_sec: float = 12.0) -> bool:
    """Wait until TCP port is accepting connections."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                s.connect((host, port))
                return True
            except OSError:
                time.sleep(0.2)
    return False

def load_agent_class(file_path):
    """
    Dynamically imports and returns an agent class from a string path.
    Example: 'agents.test_agents.AllInAgent' -> AllInAgent class
    """
    module_path, class_name = file_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def main():
    # Load configuration
    with open('agent_config.json', 'r') as f:
        config = json.load(f)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Load agent classes dynamically
    bot0_class = load_agent_class(config['bot0']['file_path'])
    bot1_class = load_agent_class(config['bot1']['file_path'])

    # Create processes using the configuration (stream=True so agent logs appear in console)
    # To disable agent logs, set stream=False
    process0 = multiprocessing.Process(
        target=bot0_class.run,
        args=(True, config['bot0']['port']),
        kwargs={"player_id": config['bot0']['player_id']}
    )
    process1 = multiprocessing.Process(
        target=bot1_class.run,
        args=(True, config['bot1']['port']),
        kwargs={"player_id": config['bot1']['player_id']}
    )

    # Ensure ports are free before spawning bot processes (avoids WinError 10048
    # when re-running immediately after a previous match).
    for port_cfg, name in [(config['bot0']['port'], 'bot0'), (config['bot1']['port'], 'bot1')]:
        port = int(port_cfg)
        if not _wait_for_port_free("0.0.0.0", port, timeout_sec=15.0):
            raise RuntimeError(
                f"Port {port} ({name}) is still in use after 15 s. "
                "Kill any lingering Python processes or wait a moment and retry."
            )

    process0.start()
    process1.start()

    # Fail fast with a clear message if agent servers did not boot.
    bot0_ready = _wait_for_port("127.0.0.1", int(config['bot0']['port']))
    bot1_ready = _wait_for_port("127.0.0.1", int(config['bot1']['port']))
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
            f"bot0_exit={p0_code}, bot1_exit={p1_code}. "
            "Tip: run with your project venv interpreter, e.g. "
            "`pokerenv/Scripts/python.exe run.py` (or activate pokerenv first). "
            f"Current interpreter: {sys.executable}"
        )

    logger.info("Starting API-based match")
    result = run_api_match(
        f"http://localhost:{config['bot0']['port']}",
        f"http://localhost:{config['bot1']['port']}",
        logger,
        csv_path=config['match_settings']['csv_output_path'],
        team_0_name=bot0_class.__name__,
        team_1_name=bot1_class.__name__
    )
    logger.info(f"Match result: {result}")

    # Clean up processes
    process0.terminate()
    process1.terminate()
    process0.join()
    process1.join()


if __name__ == "__main__":
    main()
