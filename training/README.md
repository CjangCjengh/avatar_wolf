# Training

GRPO training of the causal-aware Reasoner, based on [verl](https://github.com/verl-project/verl).

## Pipeline

1. **Self-play data collection** — run the Werewolf environment with `camwolf` agents on all players:

   ```bash
   # serve the policy model first, e.g. on port 8002
   vllm serve /path/to/Qwen2.5-72B-Instruct \
       --served-model-name Qwen/Qwen2.5-72B-Instruct \
       --port 8002 --max-model-len 16384

   python training/scripts/collect_self_play_logs.py \
       --base_config werewolf/config.local.json \
       --num_games 500 --output_dir logs/werewolf/self_play
   ```

   Every discussion turn is logged to `camwolf_structured.jsonl` (structured output + game log + alive players), and each game writes `roles.json` with the ground-truth role assignments.

2. **Dataset conversion** — join speaking turns with ground-truth roles and sample 3,000 turns:

   ```bash
   python training/scripts/convert_logs_to_verl_data.py \
       --log_dir werewolf/logs/werewolf/self_play \
       --output data/grpo_train_data.jsonl --num_samples 3000
   ```

3. **GRPO optimization** — verl recipe at [`configs/grpo.yaml`](configs/grpo.yaml). The custom reward function is bound in [`rewards/verl_reward.py`](rewards/verl_reward.py) and configured via environment variables (`INTERVENTION_API_BASE`, `INTERVENTION_MODEL`, ...).

   ```bash
   # serve the intervention LLM (can share the policy server in small setups)
   vllm serve /path/to/Qwen2.5-14B-Instruct \
       --served-model-name Qwen/Qwen2.5-14B-Instruct --port 8001

   INTERVENTION_API_BASE=http://127.0.0.1:8001/v1 \
   python -m verl.trainer.main_ppo \
       --config-path training/configs --config-name grpo
   ```
