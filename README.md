# CaM-Wolf

[![Paper](https://img.shields.io/badge/Paper-ACM%20MM%202026-red)]() [![Project Page](https://img.shields.io/badge/Project%20Page-GitHub%20Pages-blue)](https://cjangcjengh.github.io/avatar_wolf/)

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
@inproceedings{zhang2026camwolf,
    title = {CaM-Wolf: Causal-Aware Multimodal Agents for Social Deduction Games},
    author = {Zhang, Zheng and Yao, Nanjie and He, Jiarui and Ye, Deheng and Zhao, Peilin and Wang, Hao},
    booktitle = {Proceedings of the 35th ACM International Conference on Multimedia (MM '26)},
    year = {2026}
}
```
