#!/usr/bin/env python
# encoding: utf-8
"""
Causal rewards for GRPO training of the CaM-Wolf Reasoner.

For each sampled response we compute:

- Correctness reward: +1 per conclusion matching the ground-truth role, -1
  otherwise.
- Faithfulness reward (only for correct conclusions):
  * Relevant premise intervention: removing a cited premise from the game
    log should change the conclusion. If it does not, the premise is
    spurious: penalty -1/|P_i| per such premise.
  * Irrelevant premise intervention: removing a premise cited for a
    *different* identification should not change this conclusion. If it
    does, the model failed to cite evidence it relies on: penalty -0.5.
- Format reward: -1 if the output violates the structured template.
- Repetition penalty: -1 per redundant identification of the same player.

Premise interventions and role re-inferences are executed by a lightweight
LLM served behind an OpenAI-compatible API (e.g. vLLM), and are fully
parallelized across premises.
"""
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from openai import OpenAI

from .parsing import parse_identifications, parse_response_fields, parse_think
from ..prompts import (
    INTERVENTION_SYSTEM_PROMPT, INTERVENTION_USER_PROMPT,
    REINFER_SYSTEM_PROMPT, REINFER_USER_PROMPT,
)

logger = logging.getLogger(__name__)

DEFAULT_REWARD_CFG = {
    "correct": 1.0,
    "incorrect": -1.0,
    "faith_irrelevant_penalty": -0.5,
    "format_penalty": -1.0,
    "repeat_penalty": -1.0,
}

def apply_diff(game_log: str, diff_text: str) -> str:
    """
    Apply a [BEGIN_DIFF] ... [END_DIFF] block to the game log.

    Each change is either a (- old / + new) line pair (replace old with new)
    or a lone `- line` (delete the line entirely).
    """
    m = re.search(r"\[BEGIN_DIFF\](.*?)(?:\[END_DIFF\]|$)", diff_text, re.S)
    if not m:
        logger.warning("No diff block found in intervention output")
        return game_log

    lines = m.group(1).strip().splitlines()
    result = game_log
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("- "):
            old = line[2:]
            new = None
            if i + 1 < len(lines) and lines[i + 1].startswith("+ "):
                new = lines[i + 1][2:]
                i += 1
            if old in result:
                result = result.replace(old, new if new is not None else "", 1)
            else:
                logger.warning("Diff line not found in game log: %s", old[:80])
        i += 1

    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()

class InterventionClient:
    """OpenAI-API client for premise intervention and role re-inference."""

    def __init__(self, api_base: str, api_key: str, model: str,
                 max_parallel: int = 32, temperature: float = 0.0,
                 max_retries: int = 2):
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model = model
        self.max_parallel = max_parallel
        self.temperature = temperature
        self.max_retries = max_retries

    def _chat(self, system: str, user: str, max_tokens: int = 2048) -> str:
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                logger.warning("Intervention LLM call failed (%s), attempt %d",
                               e, attempt + 1)
        return ""

    def intervene(self, target_premise: str, game_log: str) -> str:
        """Remove a premise from the game log; return the intervened log."""
        out = self._chat(
            INTERVENTION_SYSTEM_PROMPT,
            INTERVENTION_USER_PROMPT.format(
                target_premise=target_premise, game_log=game_log),
        )
        if not out:
            return game_log
        return apply_diff(game_log, out)

    def reinfer(self, intervened_log: str, game_rules: str, game_state: str,
                target_player: str, candidate_roles: List[str]) -> Optional[str]:
        """Re-infer the role of target_player from the intervened log."""
        out = self._chat(
            REINFER_SYSTEM_PROMPT.format(
                game_rules=game_rules, game_state=game_state,
                intervened_game_log=intervened_log),
            REINFER_USER_PROMPT.format(
                target_player=target_player,
                candidate_roles=", ".join(candidate_roles)),
            max_tokens=64,
        )
        return normalize_role(out, candidate_roles)

def normalize_role(text: str, candidate_roles: List[str]) -> Optional[str]:
    """Extract a candidate role from a [ROLE] line (or raw text)."""
    m = re.search(r"\[ROLE\]\s*(.+)", text)
    cand = m.group(1).strip() if m else text.strip()
    cand = cand.strip().strip(".").lower()
    for role in candidate_roles:
        if role.lower() == cand:
            return role
    for role in candidate_roles:
        if role.lower() in cand:
            return role
    return None

def check_format(raw_output: str, other_living: List[str],
                 identifications: List[dict]) -> bool:
    """True if the output complies with the structured template."""
    if not parse_think(raw_output):
        return False
    if "[BEGIN_IDENTIFICATION]" not in raw_output:
        return False
    if "[BEGIN_RESPONSE]" not in raw_output:
        return False
    if not parse_response_fields(raw_output)["speech"]:
        return False
    covered = {i["player"] for i in identifications}
    return all(p in covered for p in other_living)

def compute_causal_reward(raw_output: str, ground_truth_roles: dict,
                          game_log: str, alive_players: List[str],
                          candidate_roles: List[str], player_name: str,
                          game_rules: str, game_state: str,
                          client: InterventionClient,
                          reward_cfg: dict = None) -> dict:
    """
    Compute the composite causal reward for one sampled response.

    Returns {"reward": float, "details": {...}}.
    """
    cfg = {**DEFAULT_REWARD_CFG, **(reward_cfg or {})}
    other_living = [p for p in alive_players if p != player_name]

    identifications = parse_identifications(raw_output)

    details = {"identifications": [], "format_ok": True,
               "repeat_violations": 0, "interventions": []}

    r_format = 0.0
    if not check_format(raw_output, other_living, identifications):
        r_format = cfg["format_penalty"]
        details["format_ok"] = False

    seen = {}
    r_repeat = 0.0
    for ident in identifications:
        seen[ident["player"]] = seen.get(ident["player"], 0) + 1
    duplicates = sum(c - 1 for c in seen.values() if c > 1)
    r_repeat = cfg["repeat_penalty"] * duplicates
    details["repeat_violations"] = duplicates

    r_total_idents = 0.0

    def _premises_of_others(exclude_player: str) -> List[str]:
        ps = []
        for ident in identifications:
            if ident["player"] != exclude_player:
                ps.extend(ident["premises"])
        return ps

    # Only the first identification per player is scored.
    scored_players = set()
    intervention_jobs = []  # (kind, ident, premise)

    for ident in identifications:
        pname = ident["player"]
        if pname in scored_players:
            continue
        scored_players.add(pname)

        truth = ground_truth_roles.get(pname)
        correct = (
            truth is not None
            and ident["conclusion"].strip().lower() == truth.lower()
        )
        r_correct = cfg["correct"] if correct else cfg["incorrect"]
        r_total_idents += r_correct
        ident_detail = {"player": pname, "conclusion": ident["conclusion"],
                        "ground_truth": truth, "correct": correct,
                        "r_correct": r_correct, "faith_penalty": 0.0}
        details["identifications"].append(ident_detail)

        if not correct:
            continue

        n_premises = max(len(ident["premises"]), 1)
        for premise in ident["premises"]:
            intervention_jobs.append(("relevant", ident, premise, n_premises,
                                      ident_detail))
        others = _premises_of_others(pname)
        if others:
            q = random.choice(others)
            intervention_jobs.append(("irrelevant", ident, q, n_premises,
                                      ident_detail))

    def _run(job):
        kind, ident, premise, n_premises, _ = job
        intervened = client.intervene(premise, game_log)
        new_role = client.reinfer(
            intervened, game_rules, game_state,
            ident["player"], candidate_roles)
        changed = (
            new_role is not None
            and new_role.lower() != ident["conclusion"].strip().lower()
        )
        return kind, ident["player"], premise, n_premises, changed, new_role

    with ThreadPoolExecutor(max_workers=client.max_parallel) as pool:
        results = list(pool.map(_run, intervention_jobs))

    for kind, pname, premise, n_premises, changed, new_role in results:
        ident_detail = next(
            d for d in details["identifications"] if d["player"] == pname)
        if kind == "relevant":
            if not changed:
                # Removing a cited premise failed to change the conclusion:
                # the premise is spurious.
                penalty = cfg["incorrect"] / n_premises  # == -1/|P_i|
                ident_detail["faith_penalty"] += penalty
                r_total_idents += penalty
        else:
            if changed:
                # Removing an irrelevant premise changed the conclusion:
                # the model relies on evidence it did not cite.
                penalty = cfg["faith_irrelevant_penalty"]
                ident_detail["faith_penalty"] += penalty
                r_total_idents += penalty
        details["interventions"].append({
            "kind": kind, "player": pname, "premise": premise,
            "conclusion_changed": changed, "new_role": new_role,
        })

    total = r_total_idents + r_format + r_repeat
    details.update({
        "r_idents": r_total_idents,
        "r_format": r_format,
        "r_repeat": r_repeat,
    })
    return {"reward": total, "details": details}

def compute_score(data_source: str, solution_str: str, ground_truth,
                  extra_info: dict, client: InterventionClient = None,
                  reward_cfg: dict = None, **kwargs) -> float:
    """
    verl-compatible reward function.

    Args:
        data_source: Dataset source tag ("camwolf").
        solution_str: The policy's sampled response.
        ground_truth: Ground-truth role assignments (dict or JSON string).
        extra_info: Per-sample metadata written by the data converter:
            game_log, alive_players, candidate_roles, player_name,
            game_rules, game_state.
        client: InterventionClient instance (must be provided by the
            caller, e.g. via functools.partial in the training script).
    """
    import json as _json

    if isinstance(ground_truth, str):
        ground_truth = _json.loads(ground_truth)
    if client is None:
        raise ValueError(
            "compute_score requires an InterventionClient; bind one with "
            "functools.partial before registering with verl.")

    return compute_causal_reward(
        raw_output=solution_str,
        ground_truth_roles=ground_truth,
        game_log=extra_info["game_log"],
        alive_players=extra_info["alive_players"],
        candidate_roles=extra_info["candidate_roles"],
        player_name=extra_info["player_name"],
        game_rules=extra_info["game_rules"],
        game_state=extra_info["game_state"],
        client=client,
        reward_cfg=reward_cfg,
    )["reward"]
