# ONUW (One Night Ultimate Werewolf)

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
python run_onuw_battle.py -c my_config.json
```

In watch mode, non-`direct` agents display intermediate thinking processes in cyan.

## Game Rules

**One Night Ultimate Werewolf (ONUW)** is a fast-paced social deduction game with 5 players and 7 roles. Each player receives one role, and the remaining 2 roles are placed in the center pool.

**Roles** (7 total):
- **Werewolf** (1): Team Werewolf. Wakes up at night to check for other Werewolves.
- **Villager** (2): Team Village. No special abilities.
- **Seer** (1): Team Village. May examine one player's role or two center pool roles at night.
- **Robber** (1): Team Village. May swap their role with another player and view their new role.
- **Troublemaker** (1): Team Village. May swap two other players' roles without viewing them.
- **Insomniac** (1): Team Village. Views their own final role at night's end.

**Game Flow**: The game has three sequential phases:
1. **Night Phase**: Players with night abilities act in order: Werewolf → Seer → Robber → Troublemaker → Insomniac. Role swaps during the night create uncertainty.
2. **Day Phase**: All players discuss openly for multiple rounds to identify the Werewolf. Concealing and deceiving are encouraged.
3. **Voting Phase**: All players simultaneously vote to eliminate one player. The player(s) with the most votes are eliminated.

**Victory**: Team Village wins if the Werewolf is eliminated. Team Werewolf wins if the Werewolf avoids elimination. If no Werewolf exists among players (both in center) and no one dies, it's a draw.
