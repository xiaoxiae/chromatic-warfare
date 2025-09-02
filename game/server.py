"""
WebSocket Game Server for Chromatic Warfare.
"""

import asyncio
import json
import logging
import websockets
import time
import random
from websockets.asyncio.server import ServerConnection
from websockets.typing import Data
from typing import Dict, Set, Optional, Any, List, Tuple
from core import GameState, GameEngine, Command, PlayerStatus
from datetime import datetime

from config import ServerConfig, GameDefaults, BotConfig, MapConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConnectionUtils:
    """Utility class for WebSocket communication."""

    @staticmethod
    async def send_message(websocket: ServerConnection, message: Dict[str, Any]) -> bool:
        """Send a message to a specific websocket connection."""
        try:
            await websocket.send(json.dumps(message))
            return True
        except websockets.exceptions.ConnectionClosed:
            return False
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False

    @staticmethod
    async def send_error(websocket: ServerConnection, error_message: str) -> bool:
        """Send an error message to a client."""
        return await ConnectionUtils.send_message(websocket, {
            "type": "error",
            "message": error_message
        })

    @staticmethod
    async def broadcast_to_connections(connections: Set[ServerConnection],
                                     message: Dict[str, Any]) -> List[ServerConnection]:
        """
        Broadcast a message to multiple connections.
        Returns list of disconnected websockets.
        """
        if not connections:
            return []

        message_json = json.dumps(message)
        disconnected = []

        for websocket in connections.copy():
            try:
                await websocket.send(message_json)
            except websockets.exceptions.ConnectionClosed:
                disconnected.append(websocket)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.append(websocket)

        return disconnected


class GameInstance:
    """
    Single game instance managing one game with its own state and logic.
    No direct WebSocket handling - all communication goes through GameServer.
    """

    def __init__(self, game_id: str, grid_width: int | None = None, grid_height: int | None = None,
             max_turns: int | None = None, starting_units: int | None = None, map_type: str | None = None,
             vertex_weight_range: Tuple[int, int] | None = None,
             vertex_remove_probability: float | None = None,
             maze_width: int | None = None, maze_height: int | None = None,
             server_callback=None):
        """Initialize the game instance."""
        self.game_id = game_id
        self.grid_width = grid_width or GameDefaults.GRID_WIDTH
        self.grid_height = grid_height or GameDefaults.GRID_HEIGHT
        self.max_turns = max_turns or GameDefaults.MAX_TURNS
        self.starting_units = starting_units or GameDefaults.STARTING_UNITS
        self.map_type = map_type or GameDefaults.MAP_TYPE
        self.vertex_weight_range = vertex_weight_range or GameDefaults.VERTEX_WEIGHT_RANGE
        self.vertex_remove_probability = vertex_remove_probability or GameDefaults.VERTEX_REMOVE_PROBABILITY
        self.maze_width = maze_width or self.grid_width
        self.maze_height = maze_height or self.grid_height

        # Grace period management
        self.minimum_players = GameDefaults.MINIMUM_PLAYERS

        # Turn management
        self.turn_duration_seconds = GameDefaults.TURN_DURATION

        # Server callback for broadcasting messages
        self.server_callback = server_callback

        # Player management
        self.bot_players: Set[str] = set()  # Just track player IDs

        # Game state - initialize immediately with grid
        self.game_state = GameState(max_turns=self.max_turns)
        self._generate_map()

        self.game_engine: Optional[GameEngine] = None
        self.game_started = False

        # Turn management
        self.turn_commands: Dict[str, List[Command]] = {}  # player_id -> list of commands
        self.turn_timeout_seconds = 30
        self.turn_timer_task: Optional[asyncio.Task] = None

        # Game control
        self.game_ended_time: Optional[float] = None  # Timestamp when game ended
        self.spawned_bots: List[asyncio.Task] = []  # Track spawned bot tasks

        logger.info(f"Game instance {self.game_id} created with grid {grid_width}x{grid_height}")

    # Add this method to the GameInstance class
    async def log_game_completion(self) -> None:
        """Log the completed game to the results file."""
        try:
            if not self.game_state or not self.game_state.final_rankings:
                return

            # Format: YYYY-MM-DD HH:MM:SS [player1, player2, player3]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"{timestamp} {self.game_state.final_rankings}\n"

            # Append to log file
            with open(ServerConfig.GAME_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(log_entry)

            logger.info(f"Game {self.game_id} logged: {self.game_state.final_rankings}")

        except Exception as e:
            logger.error(f"Failed to log game {self.game_id}: {e}")

    async def broadcast_message(self, message: Dict[str, Any], to_bots: bool = True, to_viewers: bool = True) -> None:
        """Broadcast a message via the server callback."""
        if self.server_callback:
            await self.server_callback(self.game_id, message, to_bots, to_viewers)

    async def send_to_bot(self, player_id: str, message: Dict[str, Any]) -> None:
        """Send a message to a specific bot via the server callback."""
        if self.server_callback:
            await self.server_callback(self.game_id, message, to_bots=True, to_viewers=False, specific_player=player_id)

    def _generate_map(self) -> None:
        """Generate the game map based on current settings."""
        if self.map_type == "hex":
            self.game_state.graph.generate_hex_graph(
                self.maze_width, self.maze_height,
                self.vertex_weight_range, self.vertex_remove_probability
            )
        else:  # default to grid
            self.game_state.graph.generate_grid_graph(
                self.maze_width, self.maze_height,
                self.vertex_weight_range, self.vertex_remove_probability
            )
        """Generate the game map based on current settings."""
        if self.map_type == "hex":
            self.game_state.graph.generate_hex_graph(
                self.maze_width, self.maze_height,
                self.vertex_weight_range, self.vertex_remove_probability
            )
        else:  # default to grid
            self.game_state.graph.generate_grid_graph(
                self.maze_width, self.maze_height,
                self.vertex_weight_range, self.vertex_remove_probability
            )

    async def spawn_bots(self, num_bots: int, difficulty: str, requesting_player: str) -> bool:
        """Spawn AI bots to join this game."""
        try:
            if self.game_started:
                logger.warning(f"Game {self.game_id}: Cannot spawn bots - game already started")
                return False

            if difficulty not in BotConfig.DIFFICULTIES:
                logger.warning(f"Game {self.game_id}: Invalid difficulty requested: {difficulty}")
                return False

            logger.info(f"Game {self.game_id}: Spawning {num_bots} {difficulty} bots (requested by {requesting_player})")

            # Import the appropriate bot class
            try:
                bot_prefix = BotConfig.BOT_NAME_PREFIX[difficulty]

                if difficulty == "easy":
                    from bot.easy import EasyBot
                    bot_class = EasyBot
                elif difficulty == "medium":
                    from bot.medium import MediumBot
                    bot_class = MediumBot
                elif difficulty == "hard":
                    from bot.hard import HardBot
                    bot_class = HardBot
                else:
                    raise ValueError(f"Invalid difficulty: {difficulty}")

            except ImportError as e:
                logger.error(f"Game {self.game_id}: Failed to import {difficulty} bot: {e}")
                return False

            # Spawn the bots
            for i in range(num_bots):
                bot_id = f"{bot_prefix}_{len(self.spawned_bots) + i + 1}_{self.game_id}"
                bot_task = asyncio.create_task(self._run_bot(bot_class, bot_id))
                self.spawned_bots.append(bot_task)

            # Notify requesting player
            await self.send_to_bot(requesting_player, {
                "type": "bots_spawned",
                "num_bots": num_bots,
                "difficulty": difficulty,
                "message": f"Spawned {num_bots} {difficulty} AI bots"
            })

            return True

        except Exception as e:
            logger.error(f"Game {self.game_id}: Error spawning bots: {e}")
            return False

    async def _run_bot(self, bot_class, bot_id: str) -> None:
        """Run a bot instance in this game."""
        try:
            # Create bot with connection to localhost (assuming server is local)
            bot = bot_class(game_id=self.game_id, player_id=bot_id, server_url="ws://localhost:8765")

            # Set up bot to automatically start when ready
            original_on_connection_confirmed = bot.on_connection_confirmed

            def on_bot_connected(message):
                if original_on_connection_confirmed:
                    original_on_connection_confirmed(message)

                # If this is the first bot and we now have minimum players, start immediately
                asyncio.create_task(self._check_auto_start())

            bot.on_connection_confirmed = on_bot_connected

            # Run the bot
            await bot._run_async()

        except Exception as e:
            logger.error(f"Game {self.game_id}: Bot {bot_id} crashed: {e}")

    async def _check_auto_start(self) -> None:
        """Check if we should auto-start the game with bots."""
        try:
            # Small delay to allow all bots to connect
            await asyncio.sleep(0.5)

            if (not self.game_started and
                len(self.bot_players) >= self.minimum_players and
                len(self.spawned_bots) > 0 and
                BotConfig.AUTO_START_WITH_BOTS):

                logger.info(f"Game {self.game_id}: Auto-starting with {len(self.bot_players)} players (including bots)")

                await self.start_game()

        except Exception as e:
            logger.error(f"Game {self.game_id}: Error in auto-start: {e}")

    def get_player_count(self) -> int:
        """Get number of players in this game."""
        return len(self.bot_players)

    def should_be_cleaned_up(self, cleanup_delay_seconds: float = 60.0) -> bool:
        """Check if this game instance should be cleaned up."""
        # Clean up if no players
        if len(self.bot_players) == 0:
            return True

        # Clean up if game ended more than cleanup_delay_seconds ago
        if self.game_ended_time is not None:
            time_since_end = time.time() - self.game_ended_time
            if time_since_end > cleanup_delay_seconds:
                return True

        return False

    def reset_game(self) -> None:
        """Reset the game state to allow for a new game."""
        try:
            logger.info(f"Game {self.game_id}: Resetting game state...")

            # Cancel any running tasks
            if self.turn_timer_task and not self.turn_timer_task.done():
                self.turn_timer_task.cancel()

            # Clear game state
            self.game_started = False
            self.game_engine = None
            self.turn_commands.clear()
            self.turn_timer_task = None
            self.game_ended_time = None

            # Reset game state with fresh grid
            self.game_state = GameState(max_turns=self.max_turns)
            self._generate_map()

            # Clear players (they will need to rejoin)
            self.bot_players.clear()

            logger.info(f"Game {self.game_id} reset complete. Grid regenerated with {len(self.game_state.graph.vertices)} vertices")

        except Exception as e:
            logger.error(f"Error resetting game {self.game_id}: {e}")

    def force_start_game(self) -> bool:
        """Force start the game if there's at least one connected player."""
        try:
            if self.game_started:
                logger.warning(f"Game {self.game_id} is already running")
                return False

            if len(self.bot_players) == 0:
                logger.warning(f"Cannot start game {self.game_id} - no players connected")
                return False

            logger.info(f"Game {self.game_id}: Force starting with {len(self.bot_players)} players")

            # Override minimum players requirement temporarily
            original_minimum = self.minimum_players
            self.minimum_players = 1

            # Start the game
            asyncio.create_task(self.start_game())

            # Restore original minimum
            self.minimum_players = original_minimum

            return True

        except Exception as e:
            logger.error(f"Error force starting game {self.game_id}: {e}")
            return False

    async def add_bot_player(self, player_id: str) -> bool:
        """Add a new bot player to the game."""
        try:
            # If game has ended, reset it to allow new connections
            if self.game_state and self.game_state.status.value == "ended":
                logger.info(f"Game {self.game_id}: Auto-resetting ended game for new bot {player_id}")
                self.reset_game()
                await asyncio.sleep(0.1)

            if self.game_started:
                logger.warning(f"Game {self.game_id}: Cannot add player {player_id} - game already started")
                return False

            if player_id in self.bot_players:
                logger.warning(f"Game {self.game_id}: Player {player_id} already in game")
                return False

            # Check if we have enough uncontrolled vertices for this player
            uncontrolled_vertices = self.game_state.graph.get_uncontrolled_vertices()
            if len(uncontrolled_vertices) == 0:
                logger.warning(f"Game {self.game_id}: No available starting positions for {player_id}")
                return False

            # Add player to tracking
            self.bot_players.add(player_id)

            # Add player to game state and assign starting vertex immediately
            self.game_state.add_player(player_id)

            # Assign a random starting vertex
            starting_vertex = random.choice(uncontrolled_vertices)
            starting_vertex.controller = player_id
            starting_vertex.units = self.starting_units

            # Update player's total units
            self.game_state.players[player_id].update_total_units(self.game_state.graph)

            logger.info(f"Game {self.game_id}: Bot {player_id} added and assigned vertex {starting_vertex.id}. Total bots: {len(self.bot_players)}")

            # Broadcast updated game state
            await self.broadcast_message(self.get_game_state_dict())

            return True

        except Exception as e:
            logger.error(f"Error adding bot player {player_id} to game {self.game_id}: {e}")
            return False

    async def remove_player(self, player_id: str) -> bool:
        """Remove a player from the game during the grace period."""
        try:
            if self.game_started:
                logger.warning(f"Game {self.game_id}: Cannot remove player {player_id} - game already started")
                return False

            if player_id not in self.bot_players:
                logger.warning(f"Game {self.game_id}: Player {player_id} not found")
                return False

            # Get the player's controlled vertices and make them neutral
            if player_id in self.game_state.players:
                controlled_vertices = self.game_state.graph.get_vertices_controlled_by_player(player_id)
                for vertex in controlled_vertices:
                    vertex.controller = None
                    vertex.units = 0

                # Remove player from game state
                del self.game_state.players[player_id]

            # Remove from tracking
            self.bot_players.discard(player_id)

            logger.info(f"Game {self.game_id}: Player {player_id} removed. Remaining players: {len(self.bot_players)}")

            # Broadcast updated game state
            await self.broadcast_message(self.get_game_state_dict())

            return True

        except Exception as e:
            logger.error(f"Error removing player {player_id} from game {self.game_id}: {e}")
            return False

    def has_only_bots_remaining(self) -> bool:
        """Check if the game only has AI bots remaining (no human players)."""
        if not self.bot_players:
            return False

        # Check if all connected players are spawned bots
        for player_id in self.bot_players:
            # Spawned bots have names like "EasyBot_1_game_id", "MediumBot_2_game_id", etc.
            if not any(player_id.startswith(f"{bot_type}Bot_") for bot_type in ["Easy", "Medium", "Hard"]):
                return False

        return True

    async def check_and_handle_bot_only_game(self) -> None:
        """Check if game only has bots and handle appropriately."""
        try:
            if self.has_only_bots_remaining():
                logger.info(f"Game {self.game_id}: Only bots remaining, terminating game")

                # Cancel any running tasks
                if self.turn_timer_task and not self.turn_timer_task.done():
                    self.turn_timer_task.cancel()

                # Cancel all spawned bot tasks
                for bot_task in self.spawned_bots:
                    if not bot_task.done():
                        bot_task.cancel()

                # Mark game as ended
                if self.game_state:
                    self.game_state.status = self.game_state.status.__class__.ENDED
                self.game_ended_time = time.time()

                # Clear players
                self.bot_players.clear()

        except Exception as e:
            logger.error(f"Error handling bot-only game {self.game_id}: {e}")

    async def handle_player_disconnect(self, player_id: str) -> None:
        """Handle a player disconnecting during the game."""
        try:
            if not self.game_started:
                # During grace period, remove the player completely
                await self.remove_player(player_id)
                await self.check_and_handle_bot_only_game()
            else:
                # During game, just mark as disconnected
                if player_id in self.bot_players:
                    self.bot_players.discard(player_id)
                    logger.info(f"Game {self.game_id}: Bot {player_id} disconnected during game. Total bots: {len(self.bot_players)}")

                    # Update player status in game
                    if player_id in self.game_state.players:
                        self.game_state.players[player_id].status = PlayerStatus.DISCONNECTED

                    # Check if only bots remain and handle appropriately
                    await self.check_and_handle_bot_only_game()
        except Exception as e:
            logger.error(f"Error handling player disconnect for {player_id} in game {self.game_id}: {e}")

    def handle_move_command(self, player_id: str, commands_data: List[Dict[str, Any]]) -> bool:
        """Handle a movement command from a player."""
        try:
            if not self.game_started or not self.game_state:
                return False

            if player_id not in self.bot_players:
                return False

            # Parse commands
            commands = []
            for cmd_data in commands_data:
                try:
                    command = Command(
                        player_id=player_id,
                        from_vertex=cmd_data["from"],
                        to_vertex=cmd_data["to"],
                        units=cmd_data["units"]
                    )
                    commands.append(command)
                except (KeyError, ValueError):
                    return False

            # Store commands for this turn
            self.turn_commands[player_id] = commands

            logger.info(f"Game {self.game_id}: Received {len(commands)} commands from {player_id}")
            return True

        except Exception as e:
            logger.error(f"Error handling move command in game {self.game_id}: {e}")
            return False

    async def start_game(self) -> None:
        """Initialize and start the game."""
        try:
            if self.game_started:
                return

            if len(self.bot_players) < self.minimum_players and self.minimum_players > 1:
                logger.warning(f"Game {self.game_id}: Cannot start - insufficient players ({len(self.bot_players)}/{self.minimum_players})")
                return

            self.game_started = True
            logger.info(f"Game {self.game_id}: Starting with {len(self.bot_players)} players")

            # Update game status and turn
            self.game_state.status = self.game_state.status.__class__.ACTIVE
            self.game_state.current_turn = 1

            # Initialize game engine
            self.game_engine = GameEngine(self.game_state)

            # Start the turn timer
            self.turn_timer_task = asyncio.create_task(self.turn_timer_loop())

            # Broadcast initial game state
            await self.broadcast_message(self.get_game_state_dict())

            logger.info(f"Game {self.game_id} started on turn {self.game_state.current_turn} with {self.turn_duration_seconds}s turn duration")

        except Exception as e:
            logger.error(f"Error starting game {self.game_id}: {e}")
            self.game_started = False

    async def turn_timer_loop(self) -> None:
        """Main turn timer loop that processes turns at regular intervals."""
        try:
            while not self.game_state.status.value == "ended":
                # Wait for the turn duration
                await asyncio.sleep(self.turn_duration_seconds)

                # Process the current turn
                await self.process_turn()

                # Check if game is over
                if self.game_state.status.value == "ended":
                    break

        except asyncio.CancelledError:
            logger.info(f"Game {self.game_id}: Turn timer loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in turn timer loop for game {self.game_id}: {e}")

    async def process_turn(self) -> Dict[str, Any]:
        """Process all commands for the current turn and generate detailed turn data."""
        try:
            if not self.game_engine or not self.game_state:
                return {}

            logger.info(f"Game {self.game_id}: Processing turn {self.game_state.current_turn}")

            # Capture state before turn processing for animation data
            vertices_before = {}
            for vertex in self.game_state.graph.vertices.values():
                vertices_before[vertex.id] = {
                    "controller": vertex.controller,
                    "units": vertex.units
                }

            # Process movements and capture move data
            move_animations = []

            # Get all valid movements for animation
            valid_movements = []
            for player_id, commands in self.turn_commands.items():
                for command in commands:
                    if self.game_engine.process_movement_command(
                        command.player_id, command.from_vertex,
                        command.to_vertex, command.units
                    ):
                        valid_movements.append(command)

                        # Record movement for animation
                        move_animations.append({
                            "player_id": command.player_id,
                            "from_vertex": command.from_vertex,
                            "to_vertex": command.to_vertex,
                            "units": command.units,
                            "cost": 0  # No movement costs in new system
                        })

            # Process the turn using the existing engine
            turn_result = self.game_engine.process_turn(self.turn_commands)

            # Capture unit generation data by comparing before/after states
            unit_generation_data = []
            for vertex in self.game_state.graph.vertices.values():
                before = vertices_before[vertex.id]
                current_controller = vertex.controller

                if current_controller is not None:
                    # Calculate units that moved out from this vertex
                    units_moved_out = 0
                    for move in move_animations:
                        if move["from_vertex"] == vertex.id and move["player_id"] == before["controller"]:
                            units_moved_out += move["units"]

                    # Expected units after movements but before generation and combat
                    expected_after_moves = before["units"] - units_moved_out

                    # If this vertex changed hands or gained significant units, record it
                    if current_controller != before["controller"]:
                        # Vertex changed hands - all current units are from combat
                        unit_generation_data.append({
                            "vertex_id": vertex.id,
                            "controller": current_controller,
                            "units_added": vertex.weight if vertex.weight > 0 else 0,
                            "units_from_combat": max(0, vertex.units - vertex.weight)
                        })
                    elif vertex.units > expected_after_moves:
                        # Same controller, but gained units (generation + possibly combat)
                        total_gained = vertex.units - expected_after_moves
                        generation = vertex.weight if vertex.weight > 0 else 0
                        combat_gain = max(0, total_gained - generation)

                        unit_generation_data.append({
                            "vertex_id": vertex.id,
                            "controller": current_controller,
                            "units_added": generation,
                            "units_from_combat": combat_gain
                        })
                    elif current_controller == before["controller"] and vertex.weight > 0:
                        # Normal unit generation for unchanged controller
                        unit_generation_data.append({
                            "vertex_id": vertex.id,
                            "controller": current_controller,
                            "units_added": vertex.weight,
                            "units_from_combat": 0
                        })

            # Clear commands for next turn
            self.turn_commands.clear()

            # Create detailed turn result for animations
            detailed_turn_result = {
                "type": "turn_processed",
                "turn": self.game_state.current_turn,
                "move_animations": move_animations,
                "unit_generation": unit_generation_data,
                "game_over": turn_result["game_over"],
                "turn_duration_seconds": self.turn_duration_seconds,  # Add timing info
                "max_turns": self.max_turns
            }

            # Broadcast detailed turn data first (for animations)
            logger.info(f"Game {self.game_id}: Broadcasting turn data: {len(move_animations)} moves, {len(unit_generation_data)} generations")
            await self.broadcast_message(detailed_turn_result)

            # Small delay to ensure turn_processed is handled before game_state
            await asyncio.sleep(0.05)

            # Then broadcast updated game state
            await self.broadcast_message(self.get_game_state_dict())

            if turn_result["game_over"]:
                logger.info(f"Game {self.game_id}: Game over! Rankings: {self.game_state.final_rankings}")

                await self.log_game_completion()

                # Mark game as ended
                self.game_ended_time = time.time()

                # Stop the turn timer
                if self.turn_timer_task and not self.turn_timer_task.done():
                    self.turn_timer_task.cancel()

                await self.broadcast_message({
                    "type": "game_over",
                    "turn": self.game_state.current_turn,
                    "final_rankings": self.game_state.final_rankings
                })

            return detailed_turn_result

        except Exception as e:
            logger.error(f"Error processing turn for game {self.game_id}: {e}")
            return {}

    def get_game_state_dict(self) -> Dict[str, Any]:
        """Get the current game state as a dictionary."""
        if self.game_state:
            game_state_dict = self.game_state.to_dict()

            # Add server timing information
            game_state_dict["turn_duration_seconds"] = self.turn_duration_seconds
            game_state_dict["max_turns"] = self.max_turns

            return game_state_dict
        return {}


class GameServer:
    """
    Multi-game WebSocket server that manages multiple GameInstance objects.
    Handles all WebSocket communication and routes messages to appropriate games.
    """
    def __init__(self, default_grid_width: int | None = None, default_grid_height: int | None = None,
                 default_max_turns: int | None = None, default_starting_units: int | None = None,
                 default_turn_duration: float | None = None):
        """Initialize the multi-game server."""
        self.games: Dict[str, GameInstance] = {}

        # Default settings for new games (use config values as fallbacks)
        self.default_grid_width = default_grid_width or GameDefaults.GRID_WIDTH
        self.default_grid_height = default_grid_height or GameDefaults.GRID_HEIGHT
        self.default_max_turns = default_max_turns or GameDefaults.MAX_TURNS
        self.default_starting_units = default_starting_units or GameDefaults.STARTING_UNITS
        self.default_turn_duration = default_turn_duration or GameDefaults.TURN_DURATION

        # Connection tracking
        self.bot_connections: Dict[str, Dict[str, ServerConnection]] = {}  # game_id -> {player_id -> websocket}
        self.viewer_connections: Dict[str, Set[ServerConnection]] = {}  # game_id -> set of websockets
        self.connection_to_game: Dict[ServerConnection, str] = {}  # websocket -> game_id
        self.connection_to_player: Dict[ServerConnection, str] = {}  # websocket -> player_id (for bots only)

        # Server control
        self.shutdown_requested = False
        self.cleanup_task: Optional[asyncio.Task] = None

        logger.info("Multi-game server initialized")

    def create_game(self, game_id: str, grid_width: int | None = None, grid_height: int | None = None,
               max_turns: int | None = None, starting_units: int | None = None,
               turn_duration: float | None = None, map_type: str = "grid",
               vertex_weight_range: Tuple[int, int] | None = None,
               vertex_remove_probability: float | None = None,
               maze_width: int | None = None, maze_height: int | None = None) -> GameInstance:
        """Create a new game instance with the given ID."""
        if game_id in self.games:
            raise ValueError(f"Game {game_id} already exists")

        # Use provided values or defaults
        width = grid_width if grid_width is not None else self.default_grid_width
        height = grid_height if grid_height is not None else self.default_grid_height
        turns = max_turns if max_turns is not None else self.default_max_turns
        units = starting_units if starting_units is not None else self.default_starting_units
        duration = turn_duration if turn_duration is not None else self.default_turn_duration

        game = GameInstance(game_id, width, height, turns, units, map_type,
                          vertex_weight_range, vertex_remove_probability,
                          maze_width, maze_height, self.game_message_callback)
        game.turn_duration_seconds = duration
        self.games[game_id] = game

        # Initialize connection tracking for this game
        self.bot_connections[game_id] = {}
        self.viewer_connections[game_id] = set()

        logger.info(f"Created game {game_id} with {len(game.game_state.graph.vertices)} vertices")
        return game

    async def game_message_callback(self, game_id: str, message: Dict[str, Any],
                                   to_bots: bool = True, to_viewers: bool = True,
                                   specific_player: str | None = None) -> None:
        """Callback for GameInstance to send messages through the server."""
        if specific_player:
            # Send to specific player
            if (game_id in self.bot_connections and
                specific_player in self.bot_connections[game_id]):
                websocket = self.bot_connections[game_id][specific_player]
                await ConnectionUtils.send_message(websocket, message)
        else:
            # Broadcast to bots and/or viewers
            if to_bots:
                await self.broadcast_to_bots(game_id, message)
            if to_viewers:
                await self.broadcast_to_viewers(game_id, message)

    def get_game(self, game_id: str) -> Optional[GameInstance]:
        """Get a game instance by ID."""
        return self.games.get(game_id)

    def list_games(self) -> List[str]:
        """List all active game IDs."""
        return list(self.games.keys())

    def remove_game(self, game_id: str) -> bool:
        """Remove a game instance."""
        if game_id not in self.games:
            return False

        # Clean up any connections associated with this game
        connections_to_remove = [ws for ws, gid in self.connection_to_game.items() if gid == game_id]
        for ws in connections_to_remove:
            del self.connection_to_game[ws]
            if ws in self.connection_to_player:
                del self.connection_to_player[ws]

        # Clean up connection tracking
        if game_id in self.bot_connections:
            del self.bot_connections[game_id]
        if game_id in self.viewer_connections:
            del self.viewer_connections[game_id]

        del self.games[game_id]
        logger.info(f"Removed game {game_id}")
        return True

    async def handle_client(self, websocket: ServerConnection, path: str = "/") -> None:
        """Handle a new WebSocket client connection."""
        logger.info(f"New connection from {websocket.remote_address}")

        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed: {websocket.remote_address}")
        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            await self.cleanup_connection(websocket)

    async def handle_message(self, websocket: ServerConnection, message: Data) -> None:
        """Parse and route incoming messages to the appropriate game."""
        try:
            data = json.loads(message)
            message_type = data.get("type")

            if message_type == "join_as_bot":
                await self.handle_bot_join(websocket, data)

            elif message_type == "join_as_viewer":
                await self.handle_viewer_join(websocket, data)

            elif message_type == "move_command":
                await self.handle_move_command(websocket, data)

            elif message_type == "request_bots":
                await self.handle_request_bots(websocket, data)

            elif message_type == "create_game":
                await self.handle_create_game(websocket, data)

            elif message_type == "list_games":
                await self.handle_list_games(websocket)

            elif message_type == "server_command":
                await self.handle_server_command(websocket, data)

            else:
                await ConnectionUtils.send_error(websocket, f"Unknown message type: {message_type}")

        except json.JSONDecodeError:
            await ConnectionUtils.send_error(websocket, "Invalid JSON format")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await ConnectionUtils.send_error(websocket, "Internal server error")

    async def handle_bot_join(self, websocket: ServerConnection, data: Dict[str, Any]) -> None:
        """Handle bot joining a specific game."""
        player_id = data.get("player_id")
        game_id = data.get("game_id")

        if not player_id:
            await ConnectionUtils.send_error(websocket, "player_id is required")
            return

        if not game_id:
            await ConnectionUtils.send_error(websocket, "game_id is required")
            return

        # Get or create the game
        game = self.games.get(game_id)
        if not game:
            # Auto-create game if it doesn't exist
            try:
                game = self.create_game(game_id)
                logger.info(f"Auto-created game {game_id} for bot {player_id}")
            except Exception as e:
                await ConnectionUtils.send_error(websocket, f"Failed to create game {game_id}: {e}")
                return

        # Check if player already connected
        if player_id in self.bot_connections[game_id]:
            await ConnectionUtils.send_error(websocket, f"Player {player_id} already connected")
            return

        # Add bot to the game logic
        success = await game.add_bot_player(player_id)
        if not success:
            await ConnectionUtils.send_error(websocket, "Failed to join game")
            return

        # Track the connection
        self.bot_connections[game_id][player_id] = websocket
        self.connection_to_game[websocket] = game_id
        self.connection_to_player[websocket] = player_id

        # Get starting vertex info
        starting_vertex = None
        for vertex in game.game_state.graph.vertices.values():
            if vertex.controller == player_id:
                starting_vertex = vertex.id
                break

        # Send connection confirmation
        await ConnectionUtils.send_message(websocket, {
            "type": "connection_confirmed",
            "game_id": game_id,
            "player_id": player_id,
            "starting_vertex": starting_vertex,
            "message": f"Successfully connected to game {game_id}"
        })

        # Broadcast updated game state
        await self.broadcast_game_state(game_id)

        logger.info(f"Bot {player_id} connected to game {game_id}")

    async def handle_viewer_join(self, websocket: ServerConnection, data: Dict[str, Any]) -> None:
        """Handle viewer joining a specific game."""
        game_id = data.get("game_id")

        if not game_id:
            await ConnectionUtils.send_error(websocket, "game_id is required")
            return

        # Get or create the game
        game = self.games.get(game_id)
        if not game:
            # Auto-create game if it doesn't exist
            try:
                game = self.create_game(game_id)
            except Exception as e:
                await ConnectionUtils.send_error(websocket, f"Failed to create game {game_id}: {e}")
                return

        # Add viewer
        self.viewer_connections[game_id].add(websocket)
        self.connection_to_game[websocket] = game_id

        logger.info(f"Viewer connected to game {game_id}. Total viewers: {len(self.viewer_connections[game_id])}")

        # Send connection confirmation
        await ConnectionUtils.send_message(websocket, {
            "type": "viewer_connected",
            "message": f"Connected as viewer to game {game_id}"
        })

        # Send current game state
        game_state = game.get_game_state_dict()
        if game_state:
            await ConnectionUtils.send_message(websocket, game_state)

    async def handle_move_command(self, websocket: ServerConnection, data: Dict[str, Any]) -> None:
        """Handle a movement command from a bot."""
        game_id = self.connection_to_game.get(websocket)
        player_id = self.connection_to_player.get(websocket)

        if not game_id or not player_id:
            await ConnectionUtils.send_error(websocket, "Not connected to any game")
            return

        game = self.games.get(game_id)
        if not game:
            await ConnectionUtils.send_error(websocket, f"Game {game_id} no longer exists")
            return

        commands_data = data.get("commands", [])
        success = game.handle_move_command(player_id, commands_data)

        if not success:
            await ConnectionUtils.send_error(websocket, "Invalid command or game not ready")

    async def handle_request_bots(self, websocket: ServerConnection, data: Dict[str, Any]) -> None:
        """Handle request to spawn AI bots."""
        game_id = self.connection_to_game.get(websocket)
        player_id = self.connection_to_player.get(websocket)

        if not game_id or not player_id:
            await ConnectionUtils.send_error(websocket, "Not connected to any game")
            return

        game = self.games.get(game_id)
        if not game:
            await ConnectionUtils.send_error(websocket, f"Game {game_id} no longer exists")
            return

        num_bots = data.get("num_bots", 1)
        difficulty = data.get("difficulty", BotConfig.DIFFICULTIES[0])

        if not isinstance(num_bots, int) or num_bots < 1:
            await ConnectionUtils.send_error(websocket, "num_bots must be a positive integer")
            return

        if difficulty not in BotConfig.DIFFICULTIES:
            await ConnectionUtils.send_error(websocket, f"difficulty must be one of: {', '.join(BotConfig.DIFFICULTIES)}")
            return

        success = await game.spawn_bots(num_bots, difficulty, player_id)
        if success:
            await ConnectionUtils.send_message(websocket, {
                "type": "bots_spawned",
                "num_bots": num_bots,
                "difficulty": difficulty,
                "message": f"Spawned {num_bots} {difficulty} AI bots"
            })
        else:
            await ConnectionUtils.send_error(websocket, f"Failed to spawn {difficulty} bots")

    async def handle_create_game(self, websocket: ServerConnection, data: Dict[str, Any]) -> None:
        """Handle explicit game creation request."""
        game_id = data.get("game_id")
        if not game_id:
            await ConnectionUtils.send_error(websocket, "game_id is required")
            return

        try:
            grid_width = data.get("grid_width")
            grid_height = data.get("grid_height")
            max_turns = data.get("max_turns")
            starting_units = data.get("starting_units")
            turn_duration = data.get("turn_duration")
            map_type = data.get("map_type", "grid")

            # Parse vertex weight range
            vertex_weight_range = None
            if "vertex_weight_range" in data:
                weight_range = data["vertex_weight_range"]
                if isinstance(weight_range, list) and len(weight_range) == 2:
                    vertex_weight_range = tuple(weight_range)

            vertex_remove_probability = data.get("vertex_remove_probability")
            maze_width = data.get("maze_width")
            maze_height = data.get("maze_height")

            game = self.create_game(game_id, grid_width, grid_height, max_turns,
                                  starting_units, turn_duration, map_type,
                                  vertex_weight_range, vertex_remove_probability,
                                  maze_width, maze_height)

            await ConnectionUtils.send_message(websocket, {
                "type": "game_created",
                "game_id": game_id,
                "map_type": map_type,
                "vertices": len(game.game_state.graph.vertices),
                "message": f"Game {game_id} created successfully with {map_type} map"
            })

        except ValueError as e:
            await ConnectionUtils.send_error(websocket, str(e))
        except Exception as e:
            await ConnectionUtils.send_error(websocket, f"Failed to create game: {e}")

    async def handle_list_games(self, websocket: ServerConnection) -> None:
        """Handle request to list all games."""
        games_info = []
        for game_id, game in self.games.items():
            games_info.append({
                "game_id": game_id,
                "status": game.game_state.status.value if game.game_state else "unknown",
                "players": len(self.bot_connections[game_id]),
                "viewers": len(self.viewer_connections[game_id]),
                "turn": game.game_state.current_turn if game.game_state else 0
            })

        await ConnectionUtils.send_message(websocket, {
            "type": "games_list",
            "games": games_info
        })

    async def handle_server_command(self, websocket: ServerConnection, data: Dict[str, Any]) -> None:
        """Handle server commands from viewers (like game settings changes)."""
        try:
            command = data.get("command")
            game_id = data.get("game_id")

            if not game_id:
                await ConnectionUtils.send_error(websocket, "game_id is required for server commands")
                return

            game = self.games.get(game_id)
            if not game:
                await ConnectionUtils.send_error(websocket, f"Game {game_id} does not exist")
                return

            if command == "set_turns":
                max_turns = data.get("max_turns")
                if not isinstance(max_turns, int) or max_turns < 1:
                    await ConnectionUtils.send_error(websocket, "max_turns must be a positive integer")
                    return

                # Add reasonable upper limit
                if max_turns > 1000:  # or use a config value
                    await ConnectionUtils.send_error(websocket, "max_turns cannot exceed 1000")
                    return

                if game.game_started:
                    await ConnectionUtils.send_error(websocket, f"Cannot change max turns - game {game_id} has already started")
                    return

                old_turns = game.max_turns
                game.max_turns = max_turns
                game.game_state.max_turns = max_turns

                logger.info(f"Game {game_id}: Max turns changed from {old_turns} to {max_turns} (via viewer)")
                await self.broadcast_game_state(game_id)

                await ConnectionUtils.send_message(websocket, {
                    "type": "command_success",
                    "command": "set_turns",
                    "message": f"Max turns set to {max_turns}"
                })

            elif command == "set_duration":
                duration = data.get("duration")
                if not isinstance(duration, (int, float)) or duration <= 0:
                    await ConnectionUtils.send_error(websocket, "duration must be a positive number")
                    return

                old_duration = game.turn_duration_seconds
                game.turn_duration_seconds = float(duration)

                logger.info(f"Game {game_id}: Turn duration changed from {old_duration}s to {duration}s (via viewer)")
                await self.broadcast_game_state(game_id)

                await ConnectionUtils.send_message(websocket, {
                    "type": "command_success",
                    "command": "set_duration",
                    "message": f"Turn duration set to {duration}s"
                })

            elif command == "map":
                if game.game_started:
                    await ConnectionUtils.send_error(websocket, f"Cannot change map - game {game_id} has already started")
                    return

                map_type = data.get("map_type", "grid")
                options = data.get("options", [])

                if map_type not in ["grid", "hex"]:
                    await ConnectionUtils.send_error(websocket, "map_type must be 'grid' or 'hex'")
                    return

                # Parse options
                vertex_weight_range = game.vertex_weight_range
                vertex_remove_probability = game.vertex_remove_probability
                maze_width = game.maze_width
                maze_height = game.maze_height

                for option in options:
                    if "=" in option:
                        key, value = option.split("=", 1)
                        try:
                            if key == "weight_min":
                                min_val = int(value)

                                if not (MapConfig.MIN_VERTEX_WEIGHT <= min_val <= MapConfig.MAX_VERTEX_WEIGHT):
                                    await ConnectionUtils.send_error(websocket, f"Weight minimum must be between {MapConfig.MIN_VERTEX_WEIGHT} and {MapConfig.MAX_VERTEX_WEIGHT}")
                                    return

                                if vertex_weight_range:
                                    vertex_weight_range = (min_val, vertex_weight_range[1])
                                else:
                                    vertex_weight_range = (min_val, min_val + 5)
                            elif key == "weight_max":
                                max_val = int(value)

                                if not (MapConfig.MIN_VERTEX_WEIGHT <= max_val <= MapConfig.MAX_VERTEX_WEIGHT):
                                    await ConnectionUtils.send_error(websocket, f"Weight maximum must be between {MapConfig.MIN_VERTEX_WEIGHT} and {MapConfig.MAX_VERTEX_WEIGHT}")
                                    return

                                if vertex_weight_range:
                                    vertex_weight_range = (vertex_weight_range[0], max_val)
                                else:
                                    vertex_weight_range = (1, max_val)
                            elif key == "remove_prob":
                                vertex_remove_probability = float(value)

                                if not (MapConfig.MIN_REMOVE_PROBABILITY <= vertex_remove_probability <= MapConfig.MAX_REMOVE_PROBABILITY):
                                    await ConnectionUtils.send_error(websocket, f"Remove probability must be between {MapConfig.MIN_REMOVE_PROBABILITY} and {MapConfig.MAX_REMOVE_PROBABILITY}")
                                    return
                            elif key == "maze_width":
                                maze_width = int(value)
                                if not (MapConfig.MIN_WIDTH <= maze_width <= MapConfig.MAX_WIDTH):
                                    await ConnectionUtils.send_error(websocket, f"Maze width must be between {MapConfig.MIN_WIDTH} and {MapConfig.MAX_WIDTH}")
                                    return
                            elif key == "maze_height":
                                maze_height = int(value)
                                if not (MapConfig.MIN_HEIGHT <= maze_height <= MapConfig.MAX_HEIGHT):
                                    await ConnectionUtils.send_error(websocket, f"Maze height must be between {MapConfig.MIN_HEIGHT} and {MapConfig.MAX_HEIGHT}")
                                    return
                        except ValueError:
                            await ConnectionUtils.send_error(websocket, f"Invalid option value: {key}={value}")
                            return

                # Update game parameters
                old_type = game.map_type
                game.map_type = map_type
                game.vertex_weight_range = vertex_weight_range
                game.vertex_remove_probability = vertex_remove_probability
                game.maze_width = maze_width
                game.maze_height = maze_height

                # Regenerate the map
                game._generate_map()

                # Clear any existing players since map changed
                await self.clear_game_players(game_id)
                game.bot_players.clear()
                game.game_state.players.clear()

                logger.info(f"Game {game_id}: Map changed from {old_type} to {map_type} (via viewer)")
                logger.info(f"  Maze size: {maze_width}x{maze_height}")
                logger.info(f"  Vertices: {len(game.game_state.graph.vertices)}")

                await self.broadcast_game_state(game_id)

                await ConnectionUtils.send_message(websocket, {
                    "type": "command_success",
                    "command": "map",
                    "message": f"Map changed to {map_type} {maze_width}x{maze_height}"
                })

            elif command == "reset":
                await self.reset_game(game_id)
                logger.info(f"Game {game_id} reset via viewer command")

                await ConnectionUtils.send_message(websocket, {
                    "type": "command_success",
                    "command": "reset",
                    "message": f"Game {game_id} has been reset"
                })

            elif command == "start":
                if game.game_started:
                    await ConnectionUtils.send_error(websocket, f"Game {game_id} is already running")
                    return

                if len(self.bot_connections[game_id]) == 0:
                    await ConnectionUtils.send_error(websocket, f"Cannot start game {game_id} - no players connected")
                    return

                logger.info(f"Game {game_id}: Force starting via viewer command with {len(self.bot_connections[game_id])} players")
                success = game.force_start_game()

                if success:
                    await ConnectionUtils.send_message(websocket, {
                        "type": "command_success",
                        "command": "start",
                        "message": f"Game {game_id} force-started successfully"
                    })
                else:
                    await ConnectionUtils.send_error(websocket, f"Failed to force start game {game_id}")

            else:
                await ConnectionUtils.send_error(websocket, f"Unknown server command: {command}")

        except Exception as e:
            logger.error(f"Error handling server command: {e}")
            await ConnectionUtils.send_error(websocket, "Error processing server command")

    async def cleanup_connection(self, websocket: ServerConnection) -> None:
        """Clean up a disconnected client."""
        try:
            game_id = self.connection_to_game.get(websocket)
            if not game_id:
                return

            # Remove from viewers
            if game_id in self.viewer_connections and websocket in self.viewer_connections[game_id]:
                self.viewer_connections[game_id].discard(websocket)
                logger.info(f"Viewer disconnected from game {game_id}. Total viewers: {len(self.viewer_connections[game_id])}")

            # Remove from bots
            player_id = self.connection_to_player.get(websocket)
            if player_id and game_id in self.bot_connections and player_id in self.bot_connections[game_id]:
                del self.bot_connections[game_id][player_id]

                # Notify the game logic
                game = self.games.get(game_id)
                if game:
                    await game.handle_player_disconnect(player_id)
                    # Broadcast updated game state
                    await self.broadcast_game_state(game_id)

                logger.info(f"Bot {player_id} disconnected from game {game_id}")

            # Clean up connection tracking
            if websocket in self.connection_to_game:
                del self.connection_to_game[websocket]
            if websocket in self.connection_to_player:
                del self.connection_to_player[websocket]

        except Exception as e:
            logger.error(f"Error cleaning up connection: {e}")

    async def clear_game_players(self, game_id: str) -> None:
        """Clear all players from a game."""
        if game_id in self.bot_connections:
            self.bot_connections[game_id].clear()

        # Remove connection tracking for this game's players
        connections_to_remove = []
        for websocket, conn_game_id in self.connection_to_game.items():
            if conn_game_id == game_id and websocket in self.connection_to_player:
                connections_to_remove.append(websocket)

        for websocket in connections_to_remove:
            if websocket in self.connection_to_player:
                del self.connection_to_player[websocket]

    async def reset_game(self, game_id: str) -> bool:
        """Reset a specific game."""
        game = self.games.get(game_id)
        if not game:
            return False

        # Clear all player connections
        await self.clear_game_players(game_id)

        # Reset the game logic
        game.reset_game()

        # Notify all clients about the reset
        await self.broadcast_to_bots(game_id, {
            "type": "game_reset",
            "message": "Game has been reset. Please rejoin to participate in the next game."
        })

        await self.broadcast_to_viewers(game_id, {
            "type": "game_reset",
            "message": "Game has been reset."
        })

        # Send the fresh game state
        await self.broadcast_game_state(game_id)

        return True

    async def broadcast_to_bots(self, game_id: str, message: Dict[str, Any]) -> None:
        """Send a message to all bot clients in a specific game."""
        if game_id not in self.bot_connections or not self.bot_connections[game_id]:
            return

        connections = set(self.bot_connections[game_id].values())
        disconnected = await ConnectionUtils.broadcast_to_connections(connections, message)

        # Clean up disconnected clients
        for websocket in disconnected:
            await self.cleanup_connection(websocket)

    async def broadcast_to_viewers(self, game_id: str, message: Dict[str, Any]) -> None:
        """Send a message to all viewer clients in a specific game."""
        if game_id not in self.viewer_connections or not self.viewer_connections[game_id]:
            return

        disconnected = await ConnectionUtils.broadcast_to_connections(self.viewer_connections[game_id], message)

        # Clean up disconnected clients
        for websocket in disconnected:
            await self.cleanup_connection(websocket)

    async def broadcast_game_state(self, game_id: str) -> None:
        """Broadcast the current game state to all clients in a specific game."""
        game = self.games.get(game_id)
        if not game:
            return

        game_state_dict = game.get_game_state_dict()
        if game_state_dict:
            await self.broadcast_to_bots(game_id, game_state_dict)
            await self.broadcast_to_viewers(game_id, game_state_dict)

    async def process_game_turns(self) -> None:
        """Process turns for all active games and broadcast results."""
        for game_id, game in self.games.items():
            if game.game_started and game.turn_timer_task:
                # Game handles its own turn processing
                continue

    def start_cleanup_task(self) -> None:
        """Start the automatic game cleanup task."""
        if self.cleanup_task is None or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(self.cleanup_loop())

    def stop_cleanup_task(self) -> None:
        """Stop the automatic game cleanup task."""
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()

    async def cleanup_loop(self, check_interval_seconds: float = 1.0,
                          cleanup_delay_seconds: float = 60.0) -> None:
        """Periodically check for and remove games that should be cleaned up."""
        try:
            while not self.shutdown_requested:
                await asyncio.sleep(check_interval_seconds)

                # Find games to clean up
                games_to_remove = []
                for game_id, game in self.games.items():
                    # Check if game has no connections
                    total_connections = len(self.bot_connections.get(game_id, {})) + len(self.viewer_connections.get(game_id, set()))
                    if total_connections == 0 and game.should_be_cleaned_up(cleanup_delay_seconds):
                        games_to_remove.append(game_id)

                # Remove the games
                for game_id in games_to_remove:
                    game = self.games[game_id]
                    total_connections = len(self.bot_connections.get(game_id, {})) + len(self.viewer_connections.get(game_id, set()))
                    reason = "no connections" if total_connections == 0 else "ended >1 minute ago"
                    logger.info(f"Cleaning up game {game_id} ({reason})")
                    self.remove_game(game_id)

        except asyncio.CancelledError:
            logger.info("Game cleanup loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in cleanup loop: {e}")


async def keyboard_input_handler(server: GameServer) -> None:
    """Handle keyboard input for server commands."""
    logger.info("Keyboard command handler started. Available commands:")
    logger.info("  'quit' - Stop the server")
    logger.info("  'games' - List all active games")
    logger.info("  'create <game_id>' - Create a new game")
    logger.info("  'start <game_id>' - Force start a specific game")
    logger.info("  'reset <game_id>' - Reset a specific game")
    logger.info("  'set turns <game_id> <turns>' - Set max turns for a game")
    logger.info("  'set duration <game_id> <seconds>' - Set turn duration for a game")
    logger.info("  'status' - Show current server status")
    logger.info("  'help' - Show this help message")

    loop = asyncio.get_event_loop()

    try:
        while not server.shutdown_requested:
            try:
                command = await loop.run_in_executor(None, input, "Server> ")
                command = command.strip().lower()

                if command == "quit" or command == "exit":
                    logger.info("Shutdown requested by user")
                    server.shutdown_requested = True
                    break

                elif command == "games":
                    games = server.list_games()
                    if not games:
                        logger.info("No active games")
                    else:
                        logger.info(f"=== Active Games ({len(games)}) ===")
                        for game_id in games:
                            game = server.get_game(game_id)
                            if game:
                                status = "Active" if game.game_started else "Waiting"
                                turn = game.game_state.current_turn if game.game_state else 0
                                bot_count = len(server.bot_connections.get(game_id, {}))
                                viewer_count = len(server.viewer_connections.get(game_id, set()))
                                logger.info(f"  {game_id}: {status}, Turn {turn}/{game.max_turns}, {bot_count} bots, {viewer_count} viewers")
                                logger.info(f"    Turn Duration: {game.turn_duration_seconds}s")

                elif command.startswith("create "):
                    try:
                        game_id = command.split(" ", 1)[1].strip()
                        if not game_id:
                            logger.warning("Game ID required: create <game_id>")
                        else:
                            game = server.create_game(game_id)
                            logger.info(f"Created game {game_id} successfully")
                    except ValueError as e:
                        logger.warning(str(e))
                    except IndexError:
                        logger.warning("Game ID required: create <game_id>")
                    except Exception as e:
                        logger.error(f"Failed to create game: {e}")

                elif command.startswith("reset "):
                    try:
                        game_id = command.split(" ", 1)[1].strip()
                        success = await server.reset_game(game_id)
                        if success:
                            logger.info(f"Game {game_id} reset complete")
                        else:
                            logger.warning(f"Game {game_id} does not exist")
                    except IndexError:
                        logger.warning("Game ID required: reset <game_id>")
                    except Exception as e:
                        logger.error(f"Failed to reset game: {e}")

                elif command.startswith("start "):
                    try:
                        game_id = command.split(" ", 1)[1].strip()
                        game = server.get_game(game_id)
                        if not game:
                            logger.warning(f"Game {game_id} does not exist")
                        elif game.game_started:
                            logger.warning(f"Game {game_id} is already running")
                        elif len(server.bot_connections.get(game_id, {})) == 0:
                            logger.warning(f"Cannot start game {game_id} - no players connected")
                        else:
                            logger.info(f"Force starting game {game_id} with {len(server.bot_connections[game_id])} players")
                            success = game.force_start_game()
                            if success:
                                logger.info(f"Game {game_id} force-started successfully")
                            else:
                                logger.warning(f"Failed to force start game {game_id}")
                    except IndexError:
                        logger.warning("Game ID required: start <game_id>")
                    except Exception as e:
                        logger.error(f"Failed to start game: {e}")

                elif command.startswith("set turns "):
                    try:
                        parts = command.split()
                        if len(parts) < 4:
                            logger.warning("Usage: set turns <game_id> <max_turns>")
                        else:
                            game_id = parts[2]
                            max_turns = int(parts[3])

                            if max_turns < 1:
                                logger.warning("Max turns must be at least 1")
                                continue

                            game = server.get_game(game_id)
                            if not game:
                                logger.warning(f"Game {game_id} does not exist")
                            elif game.game_started:
                                logger.warning(f"Cannot change max turns for game {game_id} - game has already started")
                            else:
                                old_turns = game.max_turns
                                game.max_turns = max_turns
                                game.game_state.max_turns = max_turns
                                logger.info(f"Game {game_id}: Max turns changed from {old_turns} to {max_turns}")

                                # Broadcast updated game state to notify clients
                                await server.broadcast_game_state(game_id)

                    except ValueError:
                        logger.warning("Max turns must be a valid number")
                    except IndexError:
                        logger.warning("Usage: set turns <game_id> <max_turns>")
                    except Exception as e:
                        logger.error(f"Failed to set max turns: {e}")

                elif command.startswith("set duration "):
                    try:
                        parts = command.split()
                        if len(parts) < 4:
                            logger.warning("Usage: set duration <game_id> <seconds>")
                        else:
                            game_id = parts[2]
                            duration = float(parts[3])

                            if duration <= 0:
                                logger.warning("Turn duration must be greater than 0")
                                continue

                            game = server.get_game(game_id)
                            if not game:
                                logger.warning(f"Game {game_id} does not exist")
                            else:
                                old_duration = game.turn_duration_seconds
                                game.turn_duration_seconds = duration
                                logger.info(f"Game {game_id}: Turn duration changed from {old_duration}s to {duration}s")

                                if game.game_started:
                                    logger.info(f"Game {game_id} is running - new duration will apply to subsequent turns")

                                # Broadcast updated game state to notify clients
                                await server.broadcast_game_state(game_id)

                    except ValueError:
                        logger.warning("Duration must be a valid number")
                    except IndexError:
                        logger.warning("Usage: set duration <game_id> <seconds>")
                    except Exception as e:
                        logger.error(f"Failed to set turn duration: {e}")

                elif command == "status":
                    logger.info("=== Multi-Game Server Status ===")
                    logger.info(f"Active Games: {len(server.games)}")
                    logger.info(f"Total Connections: {len(server.connection_to_game)}")
                    if server.games:
                        total_bots = sum(len(server.bot_connections.get(gid, {})) for gid in server.games.keys())
                        total_viewers = sum(len(server.viewer_connections.get(gid, set())) for gid in server.games.keys())
                        logger.info(f"Total Bots: {total_bots}")
                        logger.info(f"Total Viewers: {total_viewers}")
                        logger.info(f"Games: {list(server.games.keys())}")

                        # Show detailed game info
                        for game_id, game in server.games.items():
                            status = "Active" if game.game_started else "Waiting"
                            turn = game.game_state.current_turn if game.game_state else 0
                            logger.info(f"  {game_id}: {status}, Turn {turn}/{game.max_turns}, Duration: {game.turn_duration_seconds}s")

                elif command.startswith("map "):
                    try:
                        parts = command.split()
                        if len(parts) < 3:
                            logger.warning("Usage: map <game_id> <type> [options]")
                            logger.warning("  Types: grid, hex")
                            logger.warning("  Options: weight_min=N weight_max=N remove_prob=0.1 maze_width=N maze_height=N")
                        else:
                            game_id = parts[1]
                            map_type = parts[2]

                            if map_type not in ["grid", "hex"]:
                                logger.warning("Map type must be 'grid' or 'hex'")
                                continue

                            game = server.get_game(game_id)
                            if not game:
                                logger.warning(f"Game {game_id} does not exist")
                                continue

                            if game.game_started:
                                logger.warning(f"Cannot change map for game {game_id} - game has already started")
                                continue

                            # Parse options
                            vertex_weight_range = game.vertex_weight_range
                            vertex_remove_probability = game.vertex_remove_probability
                            maze_width = game.maze_width
                            maze_height = game.maze_height

                            for part in parts[3:]:
                                if "=" in part:
                                    key, value = part.split("=", 1)
                                    if key == "weight_min":
                                        min_val = int(value)
                                        if vertex_weight_range:
                                            vertex_weight_range = (min_val, vertex_weight_range[1])
                                        else:
                                            vertex_weight_range = (min_val, min_val + 5)
                                    elif key == "weight_max":
                                        max_val = int(value)
                                        if vertex_weight_range:
                                            vertex_weight_range = (vertex_weight_range[0], max_val)
                                        else:
                                            vertex_weight_range = (1, max_val)
                                    elif key == "remove_prob":
                                        vertex_remove_probability = float(value)
                                        if not (0.0 <= vertex_remove_probability <= 1.0):
                                            logger.warning("Remove probability must be between 0.0 and 1.0")
                                            continue
                                    elif key == "maze_width":
                                        maze_width = int(value)
                                        if maze_width < 1:
                                            logger.warning("Maze width must be at least 1")
                                            continue
                                    elif key == "maze_height":
                                        maze_height = int(value)
                                        if maze_height < 1:
                                            logger.warning("Maze height must be at least 1")
                                            continue

                            # Update game parameters
                            old_type = game.map_type
                            game.map_type = map_type
                            game.vertex_weight_range = vertex_weight_range
                            game.vertex_remove_probability = vertex_remove_probability
                            game.maze_width = maze_width
                            game.maze_height = maze_height

                            # Regenerate the map
                            game._generate_map()

                            logger.info(f"Game {game_id}: Map changed from {old_type} to {map_type}")
                            logger.info(f"  Maze size: {maze_width}x{maze_height}")
                            logger.info(f"  Vertices: {len(game.game_state.graph.vertices)}")
                            if vertex_weight_range:
                                logger.info(f"  Weight range: {vertex_weight_range}")
                            if vertex_remove_probability:
                                logger.info(f"  Remove probability: {vertex_remove_probability}")

                            # Clear any existing players since map changed
                            await server.clear_game_players(game_id)
                            game.bot_players.clear()
                            game.game_state.players.clear()

                            # Broadcast updated game state
                            await server.broadcast_game_state(game_id)

                    except ValueError as e:
                        logger.warning(f"Invalid value: {e}")
                    except Exception as e:
                        logger.error(f"Failed to change map: {e}")

                elif command == "help":
                    logger.info("=== Available Commands ===")
                    logger.info("  'quit' or 'exit' - Stop the server")
                    logger.info("  'games' - List all active games")
                    logger.info("  'create <game_id>' - Create a new game")
                    logger.info("  'start <game_id>' - Force start a specific game")
                    logger.info("  'reset <game_id>' - Reset a specific game")
                    logger.info("  'set turns <game_id> <max_turns>' - Set maximum turns for a game (before start)")
                    logger.info("  'set duration <game_id> <seconds>' - Set turn duration for a game")
                    logger.info("  'map <game_id> <type> [options]' - Change map type and parameters (before start)")
                    logger.info("    Types: grid, hex")
                    logger.info("    Options: weight_min=N weight_max=N remove_prob=0.1 maze_width=N maze_height=N")
                    logger.info("  'status' - Show current server status")
                    logger.info("  'help' - Show this help message")

                elif command == "":
                    continue

                else:
                    logger.warning(f"Unknown command: '{command}'. Type 'help' for available commands.")

            except EOFError:
                logger.info("EOF received, shutting down server")
                server.shutdown_requested = True
                break
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, shutting down server")
                server.shutdown_requested = True
                break
            except Exception as e:
                logger.error(f"Error processing keyboard input: {e}")

    except Exception as e:
        logger.error(f"Error in keyboard handler: {e}")
    finally:
        logger.info("Keyboard command handler shutting down")


async def run_server(host: str | None = None, port: int | None = None,
                    grid_width: int | None = None, grid_height: int | None = None,
                    turn_duration: float | None = None) -> None:
    """Run the multi-game server with keyboard command support."""
    # Use config defaults if not provided
    host = host or ServerConfig.HOST
    port = port or ServerConfig.PORT
    grid_width = grid_width or GameDefaults.GRID_WIDTH
    grid_height = grid_height or GameDefaults.GRID_HEIGHT
    turn_duration = turn_duration or GameDefaults.TURN_DURATION

    server = GameServer(default_grid_width=grid_width, default_grid_height=grid_height,
                       default_turn_duration=turn_duration)

    logger.info(f"Starting multi-game server on {host}:{port}")
    logger.info(f"Default grid size: {grid_width}x{grid_height}")
    logger.info(f"Default turn duration: {turn_duration}s")
    logger.info("")
    logger.info("Multi-game server is ready for connections and keyboard commands.")

    # Start the WebSocket server
    websocket_server = await websockets.serve(server.handle_client, host, port)

    # Start the cleanup task
    server.start_cleanup_task()

    # Start the keyboard input handler
    keyboard_task = asyncio.create_task(keyboard_input_handler(server))

    try:
        # Wait for either the keyboard handler to request shutdown or the server to stop
        await keyboard_task

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        server.shutdown_requested = True

    finally:
        logger.info("Shutting down server...")

        # Stop cleanup task
        server.stop_cleanup_task()

        # Cancel any running tasks in all games
        for game in server.games.values():
            if game.turn_timer_task and not game.turn_timer_task.done():
                game.turn_timer_task.cancel()

        # Close WebSocket server
        websocket_server.close()
        await websocket_server.wait_closed()

        # Cancel keyboard task if still running
        if not keyboard_task.done():
            keyboard_task.cancel()

        # Notify any remaining clients in all games
        for game_id in server.games.keys():
            await server.broadcast_to_bots(game_id, {
                "type": "server_shutdown",
                "message": "Server is shutting down"
            })
            await server.broadcast_to_viewers(game_id, {
                "type": "server_shutdown",
                "message": "Server is shutting down"
            })

        logger.info("Server shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
