# CaM-Wolf

[![Accepted](https://img.shields.io/badge/Accepted%20by-ACM%20MM%202026-brightgreen)](https://openreview.net/forum?id=djnKfsiN3p) [![Paper](https://img.shields.io/badge/Paper-ACM%20MM%202026-red)](https://openreview.net/forum?id=djnKfsiN3p) [![Project Page](https://img.shields.io/badge/Project%20Page-GitHub%20Pages-blue)](https://cjangcjengh.github.io/avatar_wolf/)

> 🎉 **News:** CaM-Wolf has been accepted to **ACM Multimedia 2026 (ACM MM '26)**!

## Repository Structure

| Directory | Description |
|-----------|-------------|
| [werewolf/](werewolf/README.md) | Werewolf game environment (7-player) |
| [onuw/](onuw/README.md) | One Night Ultimate Werewolf environment (5-player) |
| [perceiver/](perceiver/) | Multimodal perception module (video → text) |
| [performer/](performer/) | Avatar generation module (text → audio → video) |
| [training/](training/) | GRPO training pipeline with causal intervention rewards (verl) |
| [docs/](docs/) | Project page (GitHub Pages) |

## Configuration

Every runnable module ships two config files:

- `config.example.json` — committed template with placeholder paths. Copy it and fill in your own values.
- `config.local.json` — your local config.

```bash
cp config.example.json config.local.json
# edit config.local.json, then e.g.:
python run_werewolf_battle.py -c config.local.json
```

## Citation

```bibtex
@inproceedings{10.1145/3767308.3836043,
author = {Zhang, Zheng and Yao, Nanjie and He, Jiarui and Ye, Deheng and Zhao, Peilin and Wang, Hao},
title = {CaM-Wolf: Causal-Aware Multimodal Agents for Social Deduction Games},
year = {2026},
isbn = {9798400722134},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3767308.3836043},
doi = {10.1145/3767308.3836043},
abstract = {Social deduction games (SDGs) such as Werewolf have become challenging testbeds for AI agents. These games require complex social skills such as reasoning, deception, and collaboration. While recent advances in large language models (LLMs) have driven significant progress in SDG agents, current approaches are predominantly text-based, overlooking the multimodal nature that is fundamental to human social interaction. To bridge this gap, we introduce CaM-Wolf, the first SDG agent that integrates multimodal perception and generation. CaM-Wolf processes video inputs from other players, employs a causal-aware Reasoner trained via reinforcement learning to establish logical chains between observable behaviors and hidden roles, and presents itself through an animated avatar. Our experiments and user study show that CaM-Wolf achieves superior agent gameplay performance and enhances the quality of human-AI interaction. This work represents a significant advancement towards creating more human-like AI agents capable of participating in nuanced social dynamics. Our code is available at https://3dagentworld.github.io/avatar_wolf.},
booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
pages = {15176–15185},
numpages = {10},
keywords = {social deduction games, large language models, multimodal reasoning, video recognition, video generation},
location = {Rio de Janeiro, Brazil},
series = {MM '26}
}
```
