#!/usr/bin/env python
# encoding: utf-8
"""
Convert self-play game logs into a verl GRPO training dataset.

Joins each `camwolf_structured.jsonl` record with the ground-truth role
assignments (`roles.json`) of its game, and emits one training sample per
speaking turn. A sample contains the prompt (system + user messages) and
the metadata required by the causal reward function (game log, alive
players, candidate roles, ground-truth roles).

Example:

    python training/scripts/convert_logs_to_verl_data.py \
        --log_dir werewolf/logs/werewolf/self_play \
        --output work/data/grpo_train_data.jsonl \
        --num_samples 3000
"""
import argparse
import glob
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "werewolf"))

from prompt.camwolf_prompts import CAMWOLF_GAME_RULES  # noqa: E402

def iter_records(log_dir: str):
    """Yield (record, ground_truth_roles) for every logged speaking turn."""
    game_dirs = sorted(
        d for d in glob.glob(os.path.join(log_dir, "*")) if os.path.isdir(d))
    for game_dir in game_dirs:
        roles_path = os.path.join(game_dir, "roles.json")
        if not os.path.exists(roles_path):
            continue
        with open(roles_path, encoding="utf-8") as f:
            ground_truth = json.load(f)

        for jsonl_path in sorted(
                glob.glob(os.path.join(game_dir, "*", "camwolf_structured.jsonl"))):
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line), ground_truth

def build_sample(record: dict, ground_truth: dict) -> dict:
    game_state = (
        f"{record['phase']}\n"
        f"Living players: {', '.join(record['alive_players'])}")
    return {
        "data_source": "camwolf",
        "prompt": [
            {"role": "system", "content": record["system_prompt"]},
            {"role": "user", "content": record["user_prompt"]},
        ],
        "ability": "social_deduction",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": {
            "game_log": record["game_log"],
            "alive_players": record["alive_players"],
            "candidate_roles": record["candidate_roles"],
            "player_name": record["player"],
            "game_rules": CAMWOLF_GAME_RULES,
            "game_state": game_state,
        },
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True,
                        help="Directory containing self-play game logs.")
    parser.add_argument("--output", required=True,
                        help="Output file (.jsonl or .parquet).")
    parser.add_argument("--num_samples", type=int, default=3000,
                        help="Number of speaking turns to sample.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Skip records missing fields required by the reward.
    records = [(rec, gt) for rec, gt in iter_records(args.log_dir)
               if rec.get("game_log") and rec.get("alive_players")]
    print(f"[convert] Found {len(records)} speaking turns "
          f"with complete metadata")

    random.seed(args.seed)
    random.shuffle(records)
    records = records[:args.num_samples]
    samples = [build_sample(rec, gt) for rec, gt in records]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)),
                exist_ok=True)
    if args.output.endswith(".parquet"):
        import pandas as pd
        pd.DataFrame(samples).to_parquet(args.output, index=False)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[convert] Wrote {len(samples)} samples to {args.output}")

if __name__ == "__main__":
    main()
