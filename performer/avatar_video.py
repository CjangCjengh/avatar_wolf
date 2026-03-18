#!/usr/bin/env python
# encoding: utf-8
"""
OmniAvatar video generation wrapper.

Generates a talking-avatar video from an avatar image, a speech audio, and
a vision description, by driving the OmniAvatar inference
script through torchrun.

Expected OmniAvatar repo layout (see performer/README.md):

    <repo>/pretrained_models/Wan2.1-T2V-14B/...
    <repo>/pretrained_models/OmniAvatar-14B/pytorch_model.pt
    <repo>/pretrained_models/wav2vec2-base-960h/...
"""
import glob
import logging
import os
import subprocess
import tempfile

logger = logging.getLogger(__name__)


class OmniAvatarVideo:
    """(avatar image, audio, vision description) -> mp4 via OmniAvatar."""

    def __init__(self, repo_path: str, python_bin: str = "python",
                 config: str = "configs/inference.yaml",
                 hp_overrides: dict = None, timeout: int = 3600):
        self.repo_path = os.path.abspath(repo_path)
        self.python_bin = python_bin
        self.config = config
        self.hp_overrides = hp_overrides or {}
        self.timeout = timeout

    @classmethod
    def from_config(cls, cfg: dict) -> "OmniAvatarVideo":
        video = cfg["video"]
        hp = {}
        for key in ("num_steps", "guidance_scale", "audio_scale",
                    "overlap_frame", "max_tokens",
                    "tea_cache_l1_thresh", "num_persistent_param_in_dit"):
            if video.get(key) is not None:
                hp[key] = video[key]
        return cls(
            repo_path=video["repo_path"],
            python_bin=video.get("python_bin", "python"),
            config=video.get("config", "configs/inference.yaml"),
            hp_overrides=hp,
        )

    def generate(self, prompt: str, image_path: str, audio_path: str,
                 out_video: str) -> str:
        """
        Generate a talking-avatar video.

        Args:
            prompt: Vision description controlling the avatar's behavior.
            image_path: Reference avatar image.
            audio_path: Speech audio driving lip-sync and expressions.
            out_video: Destination mp4 path.

        Returns the path to the generated video.
        """
        with tempfile.TemporaryDirectory(prefix="omniavatar_") as work_dir:
            samples = os.path.join(work_dir, "samples.txt")
            with open(samples, "w", encoding="utf-8") as f:
                f.write(f"{prompt}@@{os.path.abspath(image_path)}@@"
                        f"{os.path.abspath(audio_path)}\n")

            torchrun = os.path.join(os.path.dirname(self.python_bin),
                                    "torchrun")
            if not os.path.exists(torchrun):
                torchrun = "torchrun"
            cmd = [torchrun, "--standalone", "--nproc_per_node=1",
                   "scripts/inference.py",
                   "--config", self.config,
                   "--input_file", samples]
            if self.hp_overrides:
                hp = ",".join(f"{k}={v}" for k, v in
                              self.hp_overrides.items())
                cmd += ["--hp", hp]

            out = subprocess.run(cmd, cwd=self.repo_path,
                                 capture_output=True, text=True,
                                 timeout=self.timeout)
            if out.returncode != 0:
                raise RuntimeError(
                    f"OmniAvatar inference failed: {out.stderr[-1000:]}")

            # Output: demo_out/<exp>/res_<stem>_..._<date>/result_000.mp4
            stem = os.path.splitext(os.path.basename(samples))[0]
            candidates = sorted(glob.glob(os.path.join(
                self.repo_path, "demo_out", "**", f"res_{stem}_*")),
                key=os.path.getmtime)
            videos = []
            for d in candidates[::-1]:
                videos = sorted(glob.glob(os.path.join(d, "result_*.mp4")))
                if videos:
                    break
            if not videos:
                raise RuntimeError(
                    "OmniAvatar produced no video under demo_out")
            os.makedirs(os.path.dirname(os.path.abspath(out_video)),
                        exist_ok=True)
            with open(videos[0], "rb") as src, open(out_video, "wb") as dst:
                dst.write(src.read())
        return out_video
