#!/bin/bash
# GRPO smoke run: a few optimization steps on a small self-play dataset.
#
# Prerequisites:
#   - intervention LLM served (default: port 8002, override INTERVENTION_API_BASE)
#   - a converted dataset (training/scripts/convert_logs_to_verl_data.py)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=5 bash training/scripts/run_grpo_smoke.sh \
#       /path/to/grpo_smoke.parquet
set -e

DATA=${1:-/home/zhangzheng/Projects/mm_opensource/work/data/grpo_smoke.parquet}
MODEL=${MODEL_PATH:-/home/zhangzheng/Models/Qwen2.5-7B-Instruct}
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

export INTERVENTION_API_BASE=${INTERVENTION_API_BASE:-http://127.0.0.1:8002/v1}
export INTERVENTION_API_KEY=${INTERVENTION_API_KEY:-EMPTY}
export INTERVENTION_MODEL=${INTERVENTION_MODEL:-Qwen/Qwen2.5-14B-Instruct}
export INTERVENTION_MAX_PARALLEL=${INTERVENTION_MAX_PARALLEL:-16}
export TOKENIZERS_PARALLELISM=false

cd "$REPO_ROOT"
exec python -m verl.trainer.main_ppo \
    data.train_files="$DATA" \
    data.val_files="$DATA" \
    data.train_batch_size=8 \
    data.max_prompt_length=3072 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path="$MODEL" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.lora.merge=true \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    +actor_rollout_ref.rollout.enable_sleep_mode=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    algorithm.adv_estimator=grpo \
    reward.custom_reward_function.path="$REPO_ROOT/training/rewards/verl_reward.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.total_epochs=1 \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.logger='["console"]' \
    trainer.save_freq=-1 \
    trainer.val_before_train=False \
    trainer.project_name=camwolf \
    trainer.experiment_name=grpo_smoke
