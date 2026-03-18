#!/usr/bin/env python
# encoding: utf-8
"""
verl entrypoint for the CaM-Wolf causal reward.

verl loads the function named by `custom_reward_function.name` from the file
at `custom_reward_function.path` with importlib, so this file must be
importable standalone (no package-relative imports).

Environment variables:
    INTERVENTION_API_BASE  OpenAI-compatible API of the intervention LLM
                           (default: http://127.0.0.1:8001/v1)
    INTERVENTION_API_KEY   API key (default: EMPTY)
    INTERVENTION_MODEL     Served model name
                           (default: Qwen/Qwen2.5-14B-Instruct)
    INTERVENTION_MAX_PARALLEL  Max concurrent interventions (default: 32)
"""
import os
import sys
from functools import partial

# Make the repository's `training` package importable when this file is
# loaded standalone by verl's importlib machinery.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from training.rewards.causal_reward import (  # noqa: E402
    InterventionClient, compute_score as _compute_score)

_client = InterventionClient(
    api_base=os.environ.get("INTERVENTION_API_BASE",
                            "http://127.0.0.1:8001/v1"),
    api_key=os.environ.get("INTERVENTION_API_KEY", "EMPTY"),
    model=os.environ.get("INTERVENTION_MODEL",
                         "Qwen/Qwen2.5-14B-Instruct"),
    max_parallel=int(os.environ.get("INTERVENTION_MAX_PARALLEL", "32")),
)

# Signature: compute_score(data_source, solution_str, ground_truth, extra_info)
compute_score = partial(_compute_score, client=_client)
