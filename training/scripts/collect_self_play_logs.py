#!/usr/bin/env python
# encoding: utf-8
"""
Collect self-play game logs for GRPO training.

Runs the Werewolf environment with `camwolf` agents on all players. Every
discussion turn is recorded in `camwolf_structured.jsonl` under each
player's log directory, and the ground-truth role assignments are saved to
`roles.json` per game.

Prerequisite: serve the policy model behind an OpenAI-compatible API, e.g.:

    vllm serve /path/to/Qwen2.5-14B-Instruct \
        --served-model-name Qwen/Qwen2.5-14B-Instruct \
        --port 8002 --max-model-len 16384

Example:

    python training/scripts/collect_self_play_logs.py \
        --base_config werewolf/config.local.json \
        --num_games 500 --output_dir logs/werewolf/self_play
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
WEREWOLF_DIR = os.path.join(REPO_ROOT, "werewolf")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default=os.path.join(
        WEREWOLF_DIR, "config.local.json"),
        help="Base config providing model/API settings.")
    parser.add_argument("--num_games", type=int, default=500)
    parser.add_argument("--start_game_idx", type=int, default=0)
    parser.add_argument("--output_dir", default="logs/werewolf/self_play")
    parser.add_argument("--exp_name", default="selfplay")
    args = parser.parse_args()

    with open(args.base_config, encoding="utf-8") as f:
        config = json.load(f)

    config["game"]["game_count"] = args.num_games
    config["game"]["start_game_idx"] = args.start_game_idx
    config["game"]["output_dir"] = args.output_dir
    config["game"]["exp_name"] = args.exp_name
    for player in config["players"]:
        player["agent_type"] = "camwolf"

    with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump(config, tf, indent=4)
        tmp_config = tf.name

    print(f"[collect] Running {args.num_games} self-play games "
          f"(config: {tmp_config})")
    print(f"[collect] Logs will be written to "
          f"{os.path.join(WEREWOLF_DIR, args.output_dir)}")
    try:
        subprocess.run(
            [sys.executable, "run_werewolf_battle.py", "-c", tmp_config],
            cwd=WEREWOLF_DIR, check=True)
    finally:
        os.unlink(tmp_config)

    print("[collect] Done. Next: convert logs with "
          "training/scripts/convert_logs_to_verl_data.py")

if __name__ == "__main__":
    main()
