#!/usr/bin/env python
# encoding: utf-8
"""
Perceiver: converts video inputs from human players into structured text.

Given a player's video segment, the Perceiver produces:
- speech transcription S_t: what the player said;
- vision description D_t: facial expressions, gestures, body language.

Backend: Qwen2.5-Omni-7B served behind an OpenAI-compatible API (vLLM),
producing S_t (speech) and D_t (vision description) from the video V_t.
"""
import base64
import logging
import mimetypes
import os
import re

from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_SPEECH_INSTRUCTION = (
    "This is a video of a player speaking during a social deduction game "
    "(Werewolf). Transcribe the player's speech content exactly once. "
    "Output only the transcription, no repetition, no commentary.")

DEFAULT_VISION_INSTRUCTION = (
    "This is a video of a player speaking during a social deduction game "
    "(Werewolf). Describe the player's facial expressions, gestures, and "
    "body language in at most 60 words. Output only the description.")

def encode_video_data_url(video_path: str) -> str:
    """Encode a local video file as a data URL for the chat API."""
    mime, _ = mimetypes.guess_type(video_path)
    mime = mime or "video/mp4"
    with open(video_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def parse_perception(text: str) -> dict:
    """Parse [SPEECH] / [VISION] sections from the model output."""
    speech = ""
    vision = ""
    m = re.search(r"\[SPEECH\](.*?)(?=\[VISION\]|$)", text, re.S)
    if m:
        speech = m.group(1).strip()
    m = re.search(r"\[VISION\](.*?)$", text, re.S)
    if m:
        vision = m.group(1).strip()
    if not speech and not vision:
        speech = text.strip()
    return {"speech": speech, "vision": vision}

class Perceiver:
    """Video -> {speech, vision} via an OpenAI-compatible multimodal API."""

    def __init__(self, api_base: str, api_key: str, model: str,
                 speech_instruction: str = DEFAULT_SPEECH_INSTRUCTION,
                 vision_instruction: str = DEFAULT_VISION_INSTRUCTION,
                 max_tokens: int = 512, temperature: float = 0.0,
                 max_retries: int = 2):
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model = model
        self.speech_instruction = speech_instruction
        self.vision_instruction = vision_instruction
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries

    @classmethod
    def from_config(cls, config_path: str) -> "Perceiver":
        import json
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        serve = cfg.get("serve", {})
        prompts = cfg.get("prompt", {})
        return cls(
            api_base=serve.get("api_base", "http://127.0.0.1:8003/v1"),
            api_key=serve.get("api_key", "EMPTY"),
            model=serve.get("served_name", "Qwen/Qwen2.5-Omni-7B"),
            speech_instruction=prompts.get(
                "speech_transcription", DEFAULT_SPEECH_INSTRUCTION),
            vision_instruction=prompts.get(
                "vision_description", DEFAULT_VISION_INSTRUCTION),
            max_tokens=cfg.get("max_new_tokens", 512),
        )

    def _ask(self, data_url: str, instruction: str) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": data_url}},
                {"type": "text", "text": instruction},
            ],
        }]
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001
                logger.warning("Perceiver call failed (%s), attempt %d",
                               e, attempt + 1)
        return ""

    def perceive(self, video_path: str) -> dict:
        """
        Process one video segment.

        Two calls are made (speech transcription and vision description),
        so a degenerate loop in one task cannot starve the other.

        Returns {"speech": str, "vision": str}.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)

        data_url = encode_video_data_url(video_path)
        speech = self._ask(data_url, self.speech_instruction)
        vision = self._ask(data_url, self.vision_instruction)
        return {"speech": speech, "vision": vision}
