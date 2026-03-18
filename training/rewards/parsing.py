#!/usr/bin/env python
# encoding: utf-8
"""
Parsers for CaM-Wolf structured outputs.

Kept in sync with werewolf/src/agents/llm_agent/camwolf_agent.py. Duplicated
on purpose so the training module does not depend on the game package.
"""
import re
from typing import List

def parse_identifications(text: str) -> List[dict]:
    """Parse [BEGIN_IDENTIFICATION] ... [END_IDENTIFICATION] blocks."""
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

def parse_think(output: str) -> str:
    m = re.search(r"<think>(.*?)(?:</think>|$)", output, re.S)
    return m.group(1).strip() if m else ""
