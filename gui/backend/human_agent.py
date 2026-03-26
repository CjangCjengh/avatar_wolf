#!/usr/bin/env python
# encoding: utf-8
"""
Human agent bridge for the GUI backend.

Plugs a human player into the Werewolf engine: the game loop runs in a
background thread; when it is the human's turn, `step()` blocks until the
web frontend submits a response through the FastAPI layer.
"""
import queue
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "werewolf"))

from src.agents.abs_agent import Agent  # noqa: E402

class HumanAgent(Agent):
    """Agent driven by HTTP requests from the GUI frontend."""

    def __init__(self, name: str, role: str, **kwargs):
        super().__init__(name=name, role=role)
        self.name = name
        self.role = role
        # Requests from the game loop to the human (speak / vote / action).
        self.inbox = queue.Queue()
        # Responses from the human back to the game loop.
        self.outbox = queue.Queue()
        # Message log shown to the human.
        self.history = []
        self.night_info = ""

    def step(self, message: str) -> str:
        phase, _, instruction = message.partition("|")
        request = {
            "phase": phase,
            "instruction": instruction,
            "role": self.role,
        }
        self.inbox.put(request)
        # Block until the frontend submits a response.
        response = self.outbox.get()
        self.history.append({"name": "Host", "message": instruction})
        self.history.append({"name": self.name, "message": response})
        return response

    def receive(self, name: str, message: str) -> None:
        _, _, content = message.partition("|")
        self.history.append({"name": name, "message": content})

    def set_night_info(self, info: str) -> None:
        self.night_info = info
