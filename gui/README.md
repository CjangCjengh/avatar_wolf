# CaM-Wolf GUI (human vs AI agents)

Frontend/backend separated: the backend runs on the GPU server, the frontend is a static page that can be opened anywhere (e.g. locally in a browser) and talks to the backend over HTTP.

## Prerequisites (server side)

1. Reasoner LLM (vLLM, port from `werewolf/config.local.json`):

   ```bash
   vllm serve /path/to/Qwen2.5-14B-Instruct \
       --served-model-name Qwen/Qwen2.5-14B-Instruct --port 8002
   ```

2. Perceiver (vLLM, port 8003):

   ```bash
   vllm serve /path/to/Qwen2.5-Omni-7B \
       --served-model-name Qwen/Qwen2.5-Omni-7B \
       --port 8003 --max-model-len 32768
   ```

3. EmotiVoice + OmniAvatar checkpoints laid out as in `performer/README.md`.

4. Avatar images at `gui/backend/assets/avatars/player N.png` (see `work/gen_avatars.py` for Qwen-Image generation).

## Run

Backend (on the GPU server):

```bash
uvicorn app:app --host 0.0.0.0 --port 8600  # from gui/backend/
```

Frontend (anywhere, e.g. locally):

```bash
# either open gui/frontend/index.html directly in a browser, or
python -m http.server 8080 --directory gui/frontend
```

Enter the backend URL (e.g. `http://<server>:8600`), pick a mode (Text-Text or Video-Video), and start a game.
