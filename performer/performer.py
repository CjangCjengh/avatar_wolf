#!/usr/bin/env python
# encoding: utf-8
"""
Avatar image generation (GPT-4o-Image) and the Performer pipeline.

An avatar image is generated once per agent player before the game starts
(F_avatar). During gameplay, each response (speech content S, vocal style
A, vision description D) is turned into audio via EmotiVoice and then into
a talking-avatar video via OmniAvatar.
"""
import base64
import json
import logging
import os

from openai import OpenAI

from .tts import EmotiVoiceTTS
from .avatar_video import OmniAvatarVideo

logger = logging.getLogger(__name__)

DEFAULT_AVATAR_PROMPT = (
    "A realistic half-body portrait of a person sitting at a table, facing "
    "the camera directly, plain indoor background, natural lighting, "
    "photorealistic, high detail. {description}")


class AvatarImageGenerator:
    """Generate per-player avatar images with an image generation API."""

    def __init__(self, api_key: str, api_base: str = None,
                 model: str = "gpt-image-1"):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model

    @classmethod
    def from_config(cls, cfg: dict) -> "AvatarImageGenerator":
        img = cfg["avatar_image"]
        return cls(api_key=img["api_key"], api_base=img.get("api_base"),
                   model=img.get("model", "gpt-image-1"))

    def generate(self, description: str, out_path: str) -> str:
        """Generate one avatar image and save it to out_path."""
        resp = self.client.images.generate(
            model=self.model,
            prompt=DEFAULT_AVATAR_PROMPT.format(description=description),
            n=1, size="1024x1024")
        b64 = resp.data[0].b64_json
        os.makedirs(os.path.dirname(os.path.abspath(out_path)),
                    exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(b64))
        return out_path


class Performer:
    """
    Full Performer pipeline: (speech, vocal, vision) -> avatar video.

    Avatar images are cached per player; audio and video are synthesized
    per speaking turn.
    """

    def __init__(self, config_path: str, assets_dir: str = "avatar_assets"):
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        self.cfg = cfg
        self.tts = EmotiVoiceTTS.from_config(cfg)
        self.video = OmniAvatarVideo.from_config(cfg)
        self.image_gen = AvatarImageGenerator.from_config(cfg)
        self.assets_dir = assets_dir
        os.makedirs(assets_dir, exist_ok=True)

    def prepare_avatar(self, player_name: str, description: str) -> str:
        """Generate (or return cached) avatar image for a player."""
        path = os.path.join(self.assets_dir, f"{player_name}.png")
        if not os.path.exists(path):
            self.image_gen.generate(description, path)
        return path

    def perform(self, player_name: str, speech: str, vocal: str,
                vision: str, turn_id: str) -> str:
        """
        Produce the avatar video for one speaking turn.

        Returns the path to the generated mp4.
        """
        avatar = os.path.join(self.assets_dir, f"{player_name}.png")
        if not os.path.exists(avatar):
            raise FileNotFoundError(
                f"Avatar image not prepared for {player_name}; "
                f"call prepare_avatar() first")

        audio_path = os.path.join(self.assets_dir, f"{turn_id}.wav")
        self.tts.synthesize(speech, vocal, audio_path)

        video_path = os.path.join(self.assets_dir, f"{turn_id}.mp4")
        self.video.generate(vision, avatar, audio_path, video_path)
        return video_path
