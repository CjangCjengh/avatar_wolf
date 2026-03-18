# Performer

Presents the agent's response through an animated avatar.

Pipeline:

1. **Avatar image** — generated once per player before the game (GPT-4o-Image), cached under the assets directory.
2. **Audio synthesis** — speech content + vocal style description → audio, via [EmotiVoice](https://github.com/netease-youdao/EmotiVoice).
3. **Video generation** — avatar image + audio + vision description → talking-avatar video, via [OmniAvatar](https://github.com/Omni-Avatar/OmniAvatar) (LoRA on Wan2.1-T2V-14B).

## Setup

```bash
cp config.example.json config.local.json  # edit paths
```

### EmotiVoice layout

The wrapper drives EmotiVoice's own inference scripts. Expected layout inside the cloned EmotiVoice repo (symlinks are fine):

```
<repo>/outputs/prompt_tts_open_source_joint/ckpt/g_00140000
<repo>/outputs/style_encoder/ckpt/checkpoint_163431
<repo>/WangZeJun/simbert-base-chinese
```

Notes:

- `simbert-base-chinese/config.json` needs `"model_type": "bert"` and `"architectures": ["BertModel"]` added for recent transformers versions.
- Output wavs land in `outputs/<logdir>/test_audio/audio/<checkpoint>/`.

### OmniAvatar layout

The wrapper runs `torchrun scripts/inference.py` in the cloned OmniAvatar repo. Expected layout (symlinks are fine):

```
<repo>/pretrained_models/Wan2.1-T2V-14B/...
<repo>/pretrained_models/OmniAvatar-14B/pytorch_model.pt
<repo>/pretrained_models/wav2vec2-base-960h/...
```

OmniAvatar pins `transformers==4.52.3`, `numpy==1.26.4`, and `torch==2.4.0`, which conflict with the main environment — use a separate conda env (e.g. `omniavatar`) and set `video.python_bin` accordingly.
