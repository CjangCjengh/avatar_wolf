# Perceiver

Converts video inputs from human players into structured text:

- **Speech transcription** $S_t$: what the player said.
- **Vision description** $D_t$: facial expressions, gestures, body language.

Backend: [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B), served with vLLM.

## Setup

```bash
cp config.example.json config.local.json  # edit paths
```

## Serving

```bash
vllm serve /path/to/Qwen2.5-Omni-7B \
    --served-model-name Qwen/Qwen2.5-Omni-7B \
    --port 8003 --max-model-len 32768
```

Note: video inputs are token-hungry (~700 tokens/s at default sampling); give the server a generous `--max-model-len`.
