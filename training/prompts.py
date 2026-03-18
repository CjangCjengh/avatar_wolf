#!/usr/bin/env python
# encoding: utf-8
"""
Prompts used by the CaM-Wolf training pipeline:
- INTERVENTION_*: counterfactual premise intervention (diff-format output).
- REINFER_*: role re-inference on an intervened game log.
- EVAL_*: role identification accuracy evaluation.
"""

# ---------- Counterfactual premise intervention ----------

INTERVENTION_SYSTEM_PROMPT = """You are a premise intervention specialist for social deduction games.
Your task is to remove a specific behavioral observation (premise) from a game log while preserving all other information.
This process simulates a counterfactual intervention: what would the game log look like if a particular behavioral observation had not occurred?
Rules:
1. Remove only the specified premise and its direct traces.
2. Preserve all other entries in the game log.
3. Maintain natural flow and coherence.
4. Do not add new information or interpretations."""

# Placeholders: {target_premise}, {game_log}
INTERVENTION_USER_PROMPT = """Target premise to remove: {target_premise}
Original game log:
{game_log}
Output the modifications in diff format. Only show the changed lines:
[BEGIN_DIFF]
- <original line containing the premise>
+ <modified line with the premise removed>
[END_DIFF]
If removing the premise makes an entire speech entry empty, output:
[BEGIN_DIFF]
- <the entire empty entry>
[END_DIFF]
Do not output any text outside the [BEGIN_DIFF] ... [END_DIFF] block."""

# ---------- Role re-inference on the intervened log ----------

REINFER_SYSTEM_PROMPT = """You are a social deduction game analyst.
{game_rules}
Current game state:
{game_state}
Here is the game log:
{intervened_game_log}"""

# Placeholders: {target_player}, {candidate_roles}
REINFER_USER_PROMPT = """Based on the game log above, predict the role of {target_player}.
Candidate roles: {candidate_roles}
You must respond with exactly one line in this format:
[ROLE] <one role from the candidate roles>
Do not output any other text."""

# ---------- Role identification accuracy evaluation ----------

EVAL_SYSTEM_PROMPT = """You are a social deduction game analyst.
{game_rules}
Your name is {player_name}. Your role is {player_role}.
Current game state:
{game_state}
Now it is the Day phase. Here is the game log:
{game_log}"""

# Placeholders: {candidate_roles}
EVAL_USER_PROMPT = """Based on the game log, predict the role of each other living player.
Candidate roles: {candidate_roles}
You must respond with exactly one line per player in this format:
[PREDICT] player_name -> role
where role must be exactly one of the candidate roles listed above.
Output one [PREDICT] line for each other living player. Do not include yourself. Do not output any other text."""
