# Werewolf

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

Copy and edit the configuration file:

```bash
cp config.json my_config.json
```

Edit `my_config.json` to set your API key, model name, and other parameters.

### 3. Run

```bash
python run_werewolf_battle.py -c my_config.json
```

In watch mode, non-`direct` agents display intermediate thinking processes in cyan.

## Game Rules

**Werewolf** is a classic social deduction game with 7 players:
- **2 Werewolves**: Know each other. Eliminate one player each night.
- **1 Seer**: Investigates one player each night to learn if they are a Werewolf.
- **1 Guardian**: Protects one player each night from elimination.
- **3 Villagers**: No special abilities.

**Night Phase**: Werewolves choose a target → Seer investigates → Guardian protects. **Day Phase**: Announce night results → Discussion → Vote to eliminate.

**Victory**: Village team wins when both Werewolves are eliminated. Werewolf team wins when Werewolves equal or outnumber Village team members.

## Configuration

The repo ships two config files:

- `config.example.json` — template with placeholder paths.
- `config.local.json` — your local config. Copy from the example and fill in real values.

```bash
cp config.example.json config.local.json
```
