#!/usr/bin/env python
# encoding: utf-8
"""
CaM-Wolf Reasoner agent.

On day-phase discussion turns, the agent performs causal-aware structured
reasoning: free-form reasoning in a <think> block, one
(premises, deduction, conclusion) role identification per other living
player, and a final response containing speech content, a vocal style
description, and a vision description.

On all other turns (night actions, voting, confirmations), it falls back to
direct response generation so that the game engine can parse player numbers
from the output.

The full structured output of every discussion turn is logged to
``camwolf_structured.jsonl`` under the agent's output directory; these records
are the raw material for GRPO training data construction.
"""
import json
import os
import re
from typing import List, Optional

from .chatgpt_agent import BaseWerewolfAgent, extract_response

DISCUSSION_MARKER = "it is your turn to speak"


def parse_identifications(text: str) -> List[dict]:
    """
    Parse [BEGIN_IDENTIFICATION] ... [END_IDENTIFICATION] blocks.

    Returns a list of dicts with keys: player, premises (list of str),
    deduction, conclusion.
    """
    block_match = re.search(
        r"\[BEGIN_IDENTIFICATION\](.*?)(?:\[END_IDENTIFICATION\]|$)",
        text, re.S)
    if not block_match:
        return []
    block = block_match.group(1)

    identifications = []
    chunks = re.split(r"\[PLAYER\]", block)
    for chunk in chunks[1:]:
        player_match = re.match(r"\s*(.+)", chunk)
        player = player_match.group(1).strip() if player_match else None

        premises = []
        prem_match = re.search(
            r"\[PREMISES\](.*?)(?=\[DEDUCTION\]|$)", chunk, re.S)
        if prem_match:
            for line in prem_match.group(1).strip().splitlines():
                line = line.strip()
                if line.startswith("-"):
                    premises.append(line.lstrip("- ").strip())

        ded_match = re.search(
            r"\[DEDUCTION\](.*?)(?=\[CONCLUSION\]|$)", chunk, re.S)
        deduction = ded_match.group(1).strip() if ded_match else ""

        con_match = re.search(
            r"\[CONCLUSION\](.*?)(?=\[PLAYER\]|\[END_IDENTIFICATION\]|\[BEGIN_RESPONSE\]|$)",
            chunk, re.S)
        conclusion = con_match.group(1).strip() if con_match else ""

        if player:
            identifications.append({
                "player": player,
                "premises": premises,
                "deduction": deduction,
                "conclusion": conclusion,
            })
    return identifications


def parse_response_fields(text: str) -> dict:
    """Parse [SPEECH] / [VOCAL] / [VISION] from the response block."""
    def _field(tag: str) -> str:
        m = re.search(
            r"\[%s\](.*?)(?=\[(?:SPEECH|VOCAL|VISION|END_RESPONSE)\]|$)" % tag,
            text, re.S)
        return m.group(1).strip() if m else ""

    return {
        "speech": _field("SPEECH"),
        "vocal": _field("VOCAL"),
        "vision": _field("VISION"),
    }


class CamWolfAgent(BaseWerewolfAgent):
    """
    CaM-Wolf Reasoner agent (1 API call per discussion turn).

    Extra constructor args:
        camwolf_system_prompt: System prompt template ({game_rules},
            {player_name}, {player_role}, {game_state}, {game_log}).
        camwolf_user_prompt: User prompt template ({player_name},
            {candidate_roles}, {other_living_players}).
        game_rules: Game rules text for the system prompt.
        candidate_roles: List of roles allowed as conclusions.
        response_prompt: Fallback prompt for non-discussion turns
            (same as DirectAgent).
        include_vision_in_speech: If True, append the vision description to
            the utterance broadcast to other players (agent-versus-agent
            games feed other agents both the speech content and the vision
            description).
    """

    def __init__(self, camwolf_system_prompt: str, camwolf_user_prompt: str,
                 game_rules: str, candidate_roles: List[str],
                 response_prompt: str,
                 include_vision_in_speech: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.camwolf_system_prompt = camwolf_system_prompt
        self.camwolf_user_prompt = camwolf_user_prompt
        self.game_rules = game_rules
        self.candidate_roles = candidate_roles
        self.response_prompt = response_prompt
        self.include_vision_in_speech = include_vision_in_speech

    def step(self, message: str) -> str:
        temp_phase = message.split("|")[0]
        self.phase = temp_phase
        message = message.split("|")[1]

        if DISCUSSION_MARKER in message:
            return self._discussion_step(message)
        return self._action_step(message)

    def _discussion_step(self, instruction: str) -> str:
        game_log = "\n".join(
            f"{item['name']}: {item['message']}"
            for item in self.conversation_history) or "None"

        other_living = self._parse_other_living_players(instruction)
        game_state = f"{self.phase}\nLiving players: {', '.join(other_living + [self.name])}"

        system = self.camwolf_system_prompt.format(
            game_rules=self.game_rules,
            player_name=self.name,
            player_role=self.role,
            game_state=game_state,
            game_log=game_log,
        )
        if self.night_info:
            system = f"{system}\n\n{self.night_info}"

        user = self.camwolf_user_prompt.format(
            player_name=self.name,
            candidate_roles=", ".join(self.candidate_roles),
            other_living_players=", ".join(other_living),
        )

        output = self.send_messages([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])

        think = self._parse_think(output)
        identifications = parse_identifications(output)
        fields = parse_response_fields(output)
        speech = fields["speech"] or extract_response(output)

        # Exposed for wrappers (e.g. the GUI performer hook).
        self.last_raw_output = output
        self.last_fields = fields
        self.last_identifications = identifications

        self.emit_thinking("CaM-Wolf Reasoning", think or output[:500])
        self.emit_thinking(
            "Role Identifications",
            json.dumps(identifications, ensure_ascii=False, indent=2))

        self._log_structured(instruction, system, user, output,
                             think, identifications, fields,
                             game_log=game_log,
                             alive_players=other_living + [self.name])

        self.log(f"{self.output_dir}/response.txt",
                 f"phase:{self.phase}\ninput:{user}\noutput:\n{output}\n"
                 f"--------------------")

        utterance = speech
        if self.include_vision_in_speech and fields["vision"]:
            utterance = f"{speech}\n(vision: {fields['vision']})"

        self.conversation_history.append({"name": "Host", "message": instruction})
        self.conversation_history.append({"name": self.name, "message": utterance})
        return utterance

    def _parse_other_living_players(self, instruction: str) -> List[str]:
        """Extract other living players from the speak order in the instruction."""
        m = re.search(r"speak in order:\s*([^.]+)", instruction)
        if not m:
            return []
        players = [p.strip() for p in m.group(1).split(",")]
        return [p for p in players if p and p != self.name]

    @staticmethod
    def _parse_think(output: str) -> str:
        m = re.search(r"<think>(.*?)(?:</think>|$)", output, re.S)
        return m.group(1).strip() if m else ""

    def _log_structured(self, instruction, system, user, output,
                        think, identifications, fields,
                        game_log=None, alive_players=None):
        record = {
            "phase": self.phase,
            "player": self.name,
            "role": self.role,
            "system_prompt": system,
            "user_prompt": user,
            "raw_output": output,
            "think": think,
            "identifications": identifications,
            "speech": fields["speech"],
            "vocal": fields["vocal"],
            "vision": fields["vision"],
            "game_log": game_log,
            "alive_players": alive_players,
            "candidate_roles": self.candidate_roles,
        }
        path = os.path.join(self.output_dir, "camwolf_structured.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _action_step(self, instruction: str) -> str:
        """Direct-style response so the engine can parse player numbers."""
        context = self.get_conversation_context()

        prompt = self.response_prompt.format(
            name=self.name, phase=self.phase, role=self.role,
            introduction=self.introduction, strategy=self.strategy,
            summary=context, plan="None", question=instruction, actions="None")

        output = self.send_messages([
            {"role": "system", "content": self.get_system_prompt_with_night_info()},
            {"role": "user", "content": prompt},
        ])

        response = extract_response(output)

        self.log(f"{self.output_dir}/response.txt",
                 f"phase:{self.phase}\ninput:{prompt}\noutput:\n{output}\n"
                 f"--------------------")

        self.conversation_history.append({"name": "Host", "message": instruction})
        self.conversation_history.append({"name": self.name, "message": response})
        return response
