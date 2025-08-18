# Chromatic Warfare Game Server

**Want to run a bot?** Check the `bots/` folder - it has its own README with bot setup instructions.

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install websockets
   ```

2. **Run the Server**
   ```bash
   python server.py
   ```

The server will start on `localhost:8765` by default.

## Server Commands

Once running, you can type commands in the server console:

- `games` - List all active games
- `create <game_id>` - Create a new game
- `start <game_id>` - Force start a game
- `reset <game_id>` - Reset a game
- `set turns <game_id> <number>` - Set max turns for a game
- `set duration <game_id> <seconds>` - Set turn duration
- `map <game_id> <type> [options]` - Change map (grid/hex)
- `status` - Show server status
- `quit` - Stop the server

## Configuration

Edit `config.py` to change default settings:

- **Server**: Host, port, game log path
- **Game**: Grid size, turn duration, max turns, starting units
- **Bots**: Difficulty levels, auto-start behavior
- **Maps**: Size limits, weight ranges, removal probability

## WebSocket API

Bots connect via WebSocket and send JSON messages:

```json
{
  "type": "join_as_bot",
  "game_id": "my_game",
  "player_id": "my_bot"
}
```

See the bot examples in `bots/` for complete implementation details.

## Logs

Game results are automatically logged to `game_results.log` with timestamps and final rankings.
