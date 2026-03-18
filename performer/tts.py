#!/usr/bin/env python
# encoding: utf-8
"""
EmotiVoice TTS wrapper.

Synthesizes speech audio from the Reasoner's speech content S and vocal
style description, by driving the EmotiVoice inference
scripts through subprocess calls.

Expected EmotiVoice repo layout (see performer/README.md):

    <repo>/outputs/prompt_tts_open_source_joint/ckpt/g_00140000
    <repo>/outputs/style_encoder/ckpt/checkpoint_163431
    <repo>/WangZeJun/simbert-base-chinese
"""
import glob
import logging
import os
import subprocess
import tempfile

logger = logging.getLogger(__name__)

DEFAULT_SPEAKER = "8051"


class EmotiVoiceTTS:
    """Text + vocal style -> wav audio via EmotiVoice."""

    def __init__(self, repo_path: str, python_bin: str = "python",
                 speaker: str = DEFAULT_SPEAKER,
                 checkpoint: str = "g_00140000",
                 logdir: str = "prompt_tts_open_source_joint",
                 config_folder: str = "config/joint",
                 timeout: int = 300):
        self.repo_path = os.path.abspath(repo_path)
        self.python_bin = python_bin
        self.speaker = speaker
        self.checkpoint = checkpoint
        self.logdir = logdir
        self.config_folder = config_folder
        self.timeout = timeout

    @classmethod
    def from_config(cls, cfg: dict) -> "EmotiVoiceTTS":
        tts = cfg["tts"]
        return cls(
            repo_path=tts["repo_path"],
            python_bin=tts.get("python_bin", "python"),
            speaker=tts.get("speaker", DEFAULT_SPEAKER),
        )

    def _phonemize(self, text: str, work_dir: str) -> str:
        """Run EmotiVoice frontend to get phonemes for the text."""
        text_file = os.path.join(work_dir, "raw.txt")
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(text.strip() + "\n")
        out = subprocess.run(
            [self.python_bin, "frontend.py", text_file],
            cwd=self.repo_path, capture_output=True, text=True,
            timeout=self.timeout)
        if out.returncode != 0:
            raise RuntimeError(f"frontend.py failed: {out.stderr[-500:]}")
        phoneme = out.stdout.strip().splitlines()[0]
        return phoneme

    def synthesize(self, text: str, vocal_style: str, out_wav: str,
                   speaker: str = None) -> str:
        """
        Synthesize `text` with the given vocal style description.

        Returns the path to the synthesized wav file.
        """
        speaker = speaker or self.speaker
        with tempfile.TemporaryDirectory(prefix="emotivoice_") as work_dir:
            phoneme = self._phonemize(text, work_dir)

            test_file = os.path.join(work_dir, "test.txt")
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(f"{speaker}|{vocal_style or 'Neutral'}|"
                        f"{phoneme}|{text.strip()}\n")

            out = subprocess.run(
                [self.python_bin, "inference_am_vocoder_joint.py",
                 "--logdir", self.logdir,
                 "--config_folder", self.config_folder,
                 "--checkpoint", self.checkpoint,
                 "--test_file", test_file],
                cwd=self.repo_path, capture_output=True, text=True,
                timeout=self.timeout)
            if out.returncode != 0:
                raise RuntimeError(
                    f"EmotiVoice inference failed: {out.stderr[-500:]}")

            # Output layout: outputs/<logdir>/test_audio/audio/<checkpoint>/<i>.wav
            wav_dir = os.path.join(
                self.repo_path, "outputs", self.logdir,
                "test_audio", "audio", self.checkpoint)
            wavs = sorted(glob.glob(os.path.join(wav_dir, "*.wav")),
                          key=os.path.getmtime)
            if not wavs:
                raise RuntimeError(
                    "EmotiVoice produced no wav under test_audio")
            os.makedirs(os.path.dirname(os.path.abspath(out_wav)),
                        exist_ok=True)
            latest = wavs[-1]
            with open(latest, "rb") as src, open(out_wav, "wb") as dst:
                dst.write(src.read())
        return out_wav
