#!/usr/bin/env python
# encoding: utf-8
"""
FastAPI backend for the CaM-Wolf human-vs-agent GUI.

Runs the Werewolf game loop in a background thread with one HumanAgent
(driven by HTTP) and six CamWolf agents (driven by the Reasoner LLM).
The frontend is a static page that can be served anywhere; it talks to
this backend over HTTP, so the frontend can run locally while the backend
(and all GPU services) live on a remote server.

Game flow per speaking turn:
- AI agent: the Reasoner produces (speech, vocal, vision); EmotiVoice
  synthesizes audio immediately; OmniAvatar renders the talking-avatar
  video asynchronously and the frontend swaps it in when ready.
- Human player: in text mode the human types; in video mode the human
  records a clip, the Perceiver (Qwen2.5-Omni) transcribes it and
  describes the visuals.
"""
import json
import logging
import os
import queue
import sys
import threading
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(CODE_ROOT, "werewolf"))
sys.path.insert(0, CODE_ROOT)

from src.games.werewolf.werewolf import Werewolf  # noqa: E402
from src.agents.llm_agent.camwolf_agent import (  # noqa: E402
    CamWolfAgent, parse_response_fields, parse_identifications)
from prompt.camwolf_prompts import (  # noqa: E402
    CAMWOLF_GAME_RULES, CAMWOLF_SYSTEM_PROMPT, CAMWOLF_USER_PROMPT)
from prompt.werewolf_prompts import (  # noqa: E402
    response_prompt, system_prompt, init_strategies,
    role_introduction, role_target)
from human_agent import HumanAgent  # noqa: E402

from perceiver.perceiver import Perceiver  # noqa: E402
from performer.tts import EmotiVoiceTTS  # noqa: E402
from performer.avatar_video import OmniAvatarVideo  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gui.backend")

DEFAULT_ROLES = ["Werewolf", "Werewolf", "Seer", "Guardian",
                 "Villager", "Villager", "Villager"]

class GuiCamWolfAgent(CamWolfAgent):
    """CamWolfAgent that reports each discussion turn to a callback."""

    def __init__(self, on_speech=None, **kwargs):
        super().__init__(**kwargs)
        self.on_speech = on_speech

    def _discussion_step(self, instruction: str) -> str:
        utterance = super()._discussion_step(instruction)
        if self.on_speech:
            self.on_speech(self.name, self.phase, utterance,
                           getattr(self, "last_raw_output", None))
        return utterance

class GameSession:
    def __init__(self, session_id: str, mode: str, config: dict,
                 perceiver: Perceiver, performer_cfg: dict,
                 assets_root: str):
        self.id = session_id
        self.mode = mode  # "text" or "video"
        self.config = config
        self.perceiver = perceiver
        self.assets_dir = os.path.join(assets_root, session_id)
        os.makedirs(self.assets_dir, exist_ok=True)

        self.lock = threading.Lock()
        self.messages = []      # public feed: {name, message, kind}
        self.awaiting = None    # pending human request
        self.assets = {}        # turn_id -> {audio, video, player, phase}
        self.game_over = False
        self.winners = []
        self.turn_counter = 0

        self.performer_jobs = queue.Queue()
        self.video_jobs = queue.Queue()
        self.tts = EmotiVoiceTTS.from_config(performer_cfg)
        self.video_gen = OmniAvatarVideo.from_config(performer_cfg)
        self.default_avatars = performer_cfg.get("default_avatars_dir")

        self._build_game()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.performer_thread = threading.Thread(
            target=self._performer_loop, daemon=True)
        self.video_thread = threading.Thread(
            target=self._video_loop, daemon=True)

    def _setup_avatars(self):
        """Copy pre-generated per-player avatar images into the session."""
        avatar_dir = os.path.join(self.assets_dir, "avatars")
        os.makedirs(avatar_dir, exist_ok=True)
        if not self.default_avatars or not os.path.isdir(self.default_avatars):
            return
        import shutil
        for f in os.listdir(self.default_avatars):
            if f.endswith(".png"):
                shutil.copy(os.path.join(self.default_avatars, f),
                            os.path.join(avatar_dir, f))

    def _build_game(self):
        game_cfg = self.config["game"]
        model_cfg = self.config["default_model"]
        self.game_output_dir = os.path.join(
            self.assets_dir, "game_log")
        os.makedirs(self.game_output_dir, exist_ok=True)

        self.game = Werewolf(
            game_cfg.get("player_nums", 7), game_cfg.get("language", "english"),
            "play", model_cfg.get("model_name", ""), self.game_output_dir)

        import random
        roles = DEFAULT_ROLES.copy()
        random.shuffle(roles)

        player_args = []
        self.human_idx = random.randrange(7)
        for i in range(7):
            name = f"player {i + 1}"
            role = roles[i]
            if i == self.human_idx:
                self.human_name = name
                self.human_role = role
                player_args.append((HumanAgent, {"name": name, "role": role}))
            else:
                log_dir = os.path.join(self.game_output_dir, name)
                os.makedirs(log_dir, exist_ok=True)
                player_args.append((GuiCamWolfAgent, {
                    "name": name,
                    "role": role,
                    "role_intro": role_introduction.get(role.lower(), ""),
                    "game_goal": role_target.get(role, "Win the game."),
                    "strategy": init_strategies.get(role, "Play strategically."),
                    "system_prompt": system_prompt.format(
                        name=name, role=role,
                        strategy=init_strategies.get(role, ""),
                        suggestion="None", other_strategy="None"),
                    "model": model_cfg["model_name"],
                    "temperature": model_cfg.get("temperature", 0.3),
                    "api_key": model_cfg["api_key"],
                    "api_base": model_cfg.get("api_base"),
                    "output_dir": log_dir,
                    "camwolf_system_prompt": CAMWOLF_SYSTEM_PROMPT,
                    "camwolf_user_prompt": CAMWOLF_USER_PROMPT,
                    "game_rules": CAMWOLF_GAME_RULES,
                    "candidate_roles": list(dict.fromkeys(DEFAULT_ROLES)),
                    "response_prompt": response_prompt,
                    "on_speech": self._on_agent_speech,
                }))
        self.game.add_players(player_args)
        self.human = self.game.players[self.human_name]
        self._setup_avatars()

    def _run(self):
        human_thread = threading.Thread(
            target=self._human_request_loop, daemon=True)
        human_thread.start()
        try:
            self.game.start()
        except Exception:  # noqa: BLE001
            logger.exception("game loop crashed")
        self.game_over = True
        self.winners = self.game.winners

    def _human_request_loop(self):
        """Forward HumanAgent requests to the awaiting slot."""
        while not self.game_over:
            try:
                req = self.human.inbox.get(timeout=1)
            except queue.Empty:
                continue
            with self.lock:
                self.awaiting = req

    def _on_agent_speech(self, player, phase, utterance, raw):
        """Called when an AI agent speaks during discussion."""
        with self.lock:
            self.turn_counter += 1
            turn_id = f"turn_{self.turn_counter:03d}"
            self.assets[turn_id] = {
                "player": player, "phase": phase,
                "audio": None, "video": None,
            }
            self.messages.append({
                "name": player, "message": utterance,
                "kind": "speech", "turn_id": turn_id,
            })
        self.performer_jobs.put((turn_id, player, utterance, raw))

    def _performer_loop(self):
        """Fast path: TTS audio for each agent speech turn."""
        while True:
            job = self.performer_jobs.get()
            if job is None:
                continue
            turn_id, player, utterance, raw = job
            try:
                speech = utterance.split("\n(vision:")[0]
                vocal = ""
                if raw:
                    vocal = parse_response_fields(raw).get("vocal", "")
                audio_path = os.path.join(self.assets_dir, f"{turn_id}.wav")
                self.tts.synthesize(speech, vocal or "Neutral", audio_path)
                with self.lock:
                    self.assets[turn_id]["audio"] = f"/assets/{self.id}/{turn_id}.wav"
                self.video_jobs.put(job)
            except Exception:  # noqa: BLE001
                logger.exception("tts job failed for %s", turn_id)

    def _video_loop(self):
        """Slow path: OmniAvatar talking-head video, delivered when ready."""
        while True:
            job = self.video_jobs.get()
            if job is None:
                continue
            turn_id, player, utterance, raw = job
            try:
                avatar = os.path.join(self.assets_dir, "avatars",
                                      f"{player}.png")
                if not os.path.exists(avatar):
                    continue
                audio_path = os.path.join(self.assets_dir, f"{turn_id}.wav")
                video_path = os.path.join(self.assets_dir, f"{turn_id}.mp4")
                vision = ""
                if raw:
                    vision = parse_response_fields(raw).get("vision", "")
                self.video_gen.generate(
                    vision or "A person speaking to the camera",
                    avatar, audio_path, video_path)
                with self.lock:
                    self.assets[turn_id]["video"] = f"/assets/{self.id}/{turn_id}.mp4"
            except Exception:  # noqa: BLE001
                logger.exception("video job failed for %s", turn_id)

    def get_state(self, cursor: int = 0) -> dict:
        with self.lock:
            return {
                "mode": self.mode,
                "human_player": self.human_name,
                "human_role": self.human_role,
                "messages": self.messages[cursor:],
                "cursor": len(self.messages),
                "awaiting": self.awaiting,
                "assets": self.assets,
                "game_over": self.game_over,
                "winners": self.winners,
                "human_history": self.human.history[-50:],
            }

    def submit(self, text: str):
        if self.awaiting is None:
            raise HTTPException(409, "Not awaiting human input")
        self.awaiting = None
        self.human.outbox.put(text)
        with self.lock:
            self.messages.append({
                "name": self.human_name, "message": text, "kind": "human"})

    def submit_video(self, path: str):
        if self.awaiting is None:
            raise HTTPException(409, "Not awaiting human input")
        result = self.perceiver.perceive(path)
        text = result["speech"] or "(inaudible)"
        vision = result["vision"]
        message = f"{text}\n(vision: {vision})" if vision else text
        self.submit(message)

app = FastAPI(title="CaM-Wolf GUI backend")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"])

SESSIONS = {}
CONFIG = None
PERCEIVER = None
PERFORMER_CFG = None
ASSETS_ROOT = os.path.join(HERE, "assets")

@app.on_event("startup")
def _startup():
    global CONFIG, PERCEIVER, PERFORMER_CFG
    with open(os.path.join(CODE_ROOT, "werewolf", "config.local.json")) as f:
        CONFIG = json.load(f)
    PERCEIVER = Perceiver.from_config(
        os.path.join(CODE_ROOT, "perceiver", "config.local.json"))
    with open(os.path.join(CODE_ROOT, "performer", "config.local.json")) as f:
        PERFORMER_CFG = json.load(f)
    os.makedirs(ASSETS_ROOT, exist_ok=True)

@app.post("/session/new")
def new_session(mode: str = Form("text")):
    sid = uuid.uuid4().hex[:8]
    session = GameSession(sid, mode, CONFIG, PERCEIVER, PERFORMER_CFG,
                          ASSETS_ROOT)
    SESSIONS[sid] = session
    session.thread.start()
    session.performer_thread.start()
    session.video_thread.start()
    return {"session_id": sid, "human_player": session.human_name,
            "human_role": session.human_role}

@app.get("/session/{sid}/state")
def get_state(sid: str, cursor: int = 0):
    session = SESSIONS.get(sid)
    if not session:
        raise HTTPException(404, "session not found")
    return session.get_state(cursor)

@app.post("/session/{sid}/speak_text")
def speak_text(sid: str, text: str = Form(...)):
    session = SESSIONS.get(sid)
    if not session:
        raise HTTPException(404, "session not found")
    session.submit(text)
    return {"ok": True}

@app.post("/session/{sid}/speak_video")
def speak_video(sid: str, file: UploadFile = File(...)):
    session = SESSIONS.get(sid)
    if not session:
        raise HTTPException(404, "session not found")
    path = os.path.join(session.assets_dir, f"upload_{uuid.uuid4().hex[:6]}.mp4")
    with open(path, "wb") as f:
        f.write(file.file.read())
    session.submit_video(path)
    return {"ok": True}

@app.get("/assets/{sid}/{filename}")
def get_asset(sid: str, filename: str):
    path = os.path.join(ASSETS_ROOT, sid, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "asset not found")
    return FileResponse(path)

@app.get("/assets/{sid}/avatars/{filename}")
def get_avatar(sid: str, filename: str):
    path = os.path.join(ASSETS_ROOT, sid, "avatars", filename)
    if not os.path.exists(path):
        raise HTTPException(404, "avatar not found")
    return FileResponse(path)
