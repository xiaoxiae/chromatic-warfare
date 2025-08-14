"""
WebSocket Game Server for Turn-Based Strategy Game

This module provides the WebSocket server infrastructure for hosting the turn-based
strategy game, managing bot connections, viewer connections, and message routing.
"""

import asyncio
import json
import logging
import websockets
import sys
import time
from typing import Dict, Set, Optional, Any
from websockets.server import WebSocketServerProtocol
from core import GameState, GameEngine, Command, PlayerStatus
from typing import *

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GameInstance:
    """
    Single game instance managing one game with its own state, connections and logic.
    """
    
    def __init__(self, game_id: str, grid_width: int = 5, grid_height: int = 5, max_turns: int = 10, starting_units: int = 5):
        """
        Initialize the game server.
        
        Args:
            grid_width: Width of the game grid
            grid_height: Height of the game grid
            max_turns: Maximum number of turns before game ends
            starting_units: Number of units each player starts with
        """
        self.game_id = game_id
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.max_turns = max_turns
        self.starting_units = starting_units
        
        # Connection management
        self.bot_connections: Dict[str, WebSocketServerProtocol] = {}  # player_id -> websocket
        self.viewer_connections: Set[WebSocketServerProtocol] = set()
        self.connection_to_player: Dict[WebSocketServerProtocol, str] = {}  # websocket -> player_id
        
        # Game state - initialize immediately with grid
        self.game_state = GameState(max_turns=self.max_turns)
        self.game_state.graph.generate_grid_graph(self.grid_width, self.grid_height)
        self.game_engine: Optional[GameEngine] = None
        self.game_started = False
        
        # Grace period management
        self.grace_period_seconds = 15
        self.minimum_players = 2
        self.grace_period_task: Optional[asyncio.Task] = None
        
        # Turn management
        self.turn_commands: Dict[str, list] = {}  # player_id -> list of commands
        self.turn_timeout_seconds = 30
        self.turn_duration_seconds = 1.0  # Duration of each turn for consistent viewing
        self.turn_timer_task: Optional[asyncio.Task] = None
        
        # Game control
        self.shutdown_requested = False
        self.game_ended_time: Optional[float] = None  # Timestamp when game ended
        
        logger.info(f"Game instance {self.game_id} created with grid {grid_width}x{grid_height}")
    
    def get_total_connections(self) -> int:
        """Get total number of connections (bots + viewers)."""
        return len(self.bot_connections) + len(self.viewer_connections)
    
    def should_be_cleaned_up(self, cleanup_delay_seconds: float = 60.0) -> bool:
        """
        Check if this game instance should be cleaned up.
        
        Args:
            cleanup_delay_seconds: How long to wait after game ends before cleanup
        
        Returns:
            True if game should be cleaned up
        """
        # Clean up if no connections
        if self.get_total_connections() == 0:
            return True
        
        # Clean up if game ended more than cleanup_delay_seconds ago
        if self.game_ended_time is not None:
            time_since_end = time.time() - self.game_ended_time
            if time_since_end > cleanup_delay_seconds:
                return True
        
        return False
    
    def reset_game(self) -> None:
        """
        Reset the game state to allow for a new game.
        """
        try:
            logger.info("Resetting game state...")
            
            # Cancel any running tasks
            if self.grace_period_task and not self.grace_period_task.done():
                self.grace_period_task.cancel()
            if self.turn_timer_task and not self.turn_timer_task.done():
                self.turn_timer_task.cancel()
            
            # Clear game state
            self.game_started = False
            self.game_engine = None
            self.turn_commands.clear()
            self.grace_period_task = None
            self.turn_timer_task = None
            self.game_ended_time = None  # Reset game end time
            
            # Reset game state with fresh grid
            self.game_state = GameState(max_turns=self.max_turns)
            self.game_state.graph.generate_grid_graph(self.grid_width, self.grid_height)
            
            # Keep connections but clear their game associations
            # (Players will need to rejoin)
            self.connection_to_player.clear()
            
            # Notify all clients about the reset
            asyncio.create_task(self.broadcast_to_bots({
                "type": "game_reset",
                "message": "Game has been reset. Please rejoin to participate in the next game."
            }))
            
            asyncio.create_task(self.broadcast_to_viewers({
                "type": "game_reset",
                "message": "Game has been reset."
            }))
            
            # Send the fresh game state to viewers so they see the reset state immediately
            asyncio.create_task(self.broadcast_game_state())
            
            logger.info(f"Game {self.game_id} reset complete. Grid regenerated with {len(self.game_state.graph.vertices)} vertices")
            logger.info(f"Game {self.game_id}: Waiting for players to reconnect...")
            
        except Exception as e:
            logger.error(f"Error resetting game: {e}")
    
    def force_start_game(self) -> bool:
        """
        Force start the game if there's at least one connected player.
        
        Returns:
            True if game was started, False otherwise
        """
        try:
            if self.game_started:
                logger.warning("Game is already running")
                return False
            
            if len(self.bot_connections) == 0:
                logger.warning("Cannot start game - no players connected")
                return False
            
            logger.info(f"Game {self.game_id}: Force starting with {len(self.bot_connections)} players")
            
            # Cancel grace period if running
            if self.grace_period_task and not self.grace_period_task.done():
                self.grace_period_task.cancel()
            
            # Override minimum players requirement temporarily
            original_minimum = self.minimum_players
            self.minimum_players = 1
            
            # Start the game
            asyncio.create_task(self.start_game())
            
            # Restore original minimum
            self.minimum_players = original_minimum
            
            return True
            
        except Exception as e:
            logger.error(f"Error force starting game: {e}")
            return False
    
    async def add_bot_connection(self, websocket: WebSocketServerProtocol, player_id: str) -> bool:
        """
        Register a new bot connection.
        
        Args:
            websocket: WebSocket connection
            player_id: Unique identifier for the bot player
            
        Returns:
            True if connection was added successfully, False otherwise
        """
        try:
            if self.game_started:
                await self.send_error(websocket, "Game has already started")
                return False
            
            if player_id in self.bot_connections:
                await self.send_error(websocket, f"Player {player_id} already connected")
                return False
            
            # Check if we have enough uncontrolled vertices for this player
            uncontrolled_vertices = self.game_state.graph.get_uncontrolled_vertices()
            if len(uncontrolled_vertices) == 0:
                await self.send_error(websocket, "No available starting positions")
                return False
            
            # Add connection
            self.bot_connections[player_id] = websocket
            self.connection_to_player[websocket] = player_id
            
            # Add player to game state and assign starting vertex immediately
            self.game_state.add_player(player_id)
            
            # Assign a random starting vertex
            import random
            starting_vertex = random.choice(uncontrolled_vertices)
            starting_vertex.controller = player_id
            starting_vertex.units = self.starting_units
            
            # Update player's total units
            self.game_state.players[player_id].update_total_units(self.game_state.graph)
            
            logger.info(f"Game {self.game_id}: Bot {player_id} connected and assigned vertex {starting_vertex.id}. Total bots: {len(self.bot_connections)}")
            
            # Send connection confirmation with game ID
            await self.send_to_websocket(websocket, {
                "type": "connection_confirmed",
                "game_id": self.game_id,
                "player_id": player_id,
                "starting_vertex": starting_vertex.id,
                "message": f"Successfully connected to game {self.game_id} and assigned starting vertex {starting_vertex.id}"
            })
            
            # Broadcast updated game state to all clients
            await self.broadcast_game_state()
            
            # Check if we can start the grace period
            if len(self.bot_connections) >= self.minimum_players and not self.game_started:
                if self.grace_period_task is None or self.grace_period_task.done():
                    self.grace_period_task = asyncio.create_task(self.start_grace_period())
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding bot connection for {player_id}: {e}")
            return False
    
    async def remove_player(self, player_id: str) -> bool:
        """
        Remove a player from the game during the grace period.
        
        Args:
            player_id: ID of the player to remove
            
        Returns:
            True if player was removed successfully, False otherwise
        """
        try:
            if self.game_started:
                logger.warning(f"Cannot remove player {player_id} - game has already started")
                return False
            
            if player_id not in self.bot_connections:
                logger.warning(f"Player {player_id} not found in connections")
                return False
            
            # Get the player's controlled vertices and make them neutral
            if player_id in self.game_state.players:
                controlled_vertices = self.game_state.graph.get_vertices_controlled_by_player(player_id)
                for vertex in controlled_vertices:
                    vertex.controller = None
                    vertex.units = 0
                
                # Remove player from game state
                del self.game_state.players[player_id]
            
            # Remove from connection tracking
            websocket = self.bot_connections[player_id]
            del self.bot_connections[player_id]
            if websocket in self.connection_to_player:
                del self.connection_to_player[websocket]
            
            logger.info(f"Player {player_id} removed during grace period. Remaining players: {len(self.bot_connections)}")
            
            # Check if we still have enough players
            if len(self.bot_connections) < self.minimum_players:
                await self.cancel_grace_period()
            
            # Broadcast updated game state
            await self.broadcast_game_state()
            
            return True
            
        except Exception as e:
            logger.error(f"Error removing player {player_id}: {e}")
            return False
    
    async def cancel_grace_period(self) -> None:
        """
        Cancel the grace period if there are not enough players.
        """
        if self.grace_period_task and not self.grace_period_task.done():
            self.grace_period_task.cancel()
            logger.info(f"Grace period canceled - not enough players ({len(self.bot_connections)}/{self.minimum_players})")
            
            # Notify remaining players
            await self.broadcast_to_bots({
                "type": "grace_period_canceled",
                "reason": "insufficient_players",
                "current_players": len(self.bot_connections),
                "minimum_required": self.minimum_players
            })
    
    async def add_viewer_connection(self, websocket: WebSocketServerProtocol) -> bool:
        try:
            self.viewer_connections.add(websocket)
            logger.info(f"Viewer connected. Total viewers: {len(self.viewer_connections)}")

            # Send connection confirmation first
            await self.send_to_websocket(websocket, {
                "type": "viewer_connected",
                "message": "Connected as viewer"
            })

            # Then send game state
            await self.send_to_websocket(websocket, self.game_state.to_dict())

            return True
            
        except Exception as e:
            logger.error(f"Error adding viewer connection: {e}")
            return False
    
    async def remove_connection(self, websocket: WebSocketServerProtocol) -> None:
        """
        Clean up a disconnected client.
        
        Args:
            websocket: WebSocket connection to remove
        """
        try:
            # Remove from viewers
            if websocket in self.viewer_connections:
                self.viewer_connections.discard(websocket)
                logger.info(f"Viewer disconnected. Total viewers: {len(self.viewer_connections)}")
                return
            
            # Remove from bots
            if websocket in self.connection_to_player:
                player_id = self.connection_to_player[websocket]
                
                if not self.game_started:
                    # During grace period, remove the player completely
                    await self.remove_player(player_id)
                else:
                    # During game, just mark as disconnected
                    del self.connection_to_player[websocket]
                    
                    if player_id in self.bot_connections:
                        del self.bot_connections[player_id]
                        logger.info(f"Bot {player_id} disconnected during game. Total bots: {len(self.bot_connections)}")
                        
                        # Update player status in game
                        if player_id in self.game_state.players:
                            self.game_state.players[player_id].status = PlayerStatus.DISCONNECTED
                            asyncio.create_task(self.broadcast_game_state())
                        
        except Exception as e:
            logger.error(f"Error removing connection: {e}")
    
    async def broadcast_to_bots(self, message: Dict[str, Any]) -> None:
        """
        Send a message to all bot clients.
        
        Args:
            message: Dictionary message to send
        """
        if not self.bot_connections:
            return
        
        message_json = json.dumps(message)
        disconnected = []
        
        # Create a copy of the items to avoid dictionary changed size during iteration
        for player_id, websocket in list(self.bot_connections.items()):
            try:
                await websocket.send(message_json)
            except websockets.exceptions.ConnectionClosed:
                disconnected.append(websocket)
            except Exception as e:
                logger.error(f"Error sending to bot {player_id}: {e}")
                disconnected.append(websocket)
        
        # Clean up disconnected clients
        for websocket in disconnected:
            await self.remove_connection(websocket)
    
    async def broadcast_to_viewers(self, message: Dict[str, Any]) -> None:
        """
        Send a message to all visualization clients.
        
        Args:
            message: Dictionary message to send
        """
        if not self.viewer_connections:
            return
        
        message_json = json.dumps(message)
        disconnected = []
        
        for websocket in self.viewer_connections.copy():
            try:
                await websocket.send(message_json)
            except websockets.exceptions.ConnectionClosed:
                disconnected.append(websocket)
            except Exception as e:
                logger.error(f"Error sending to viewer: {e}")
                disconnected.append(websocket)
        
        # Clean up disconnected clients
        for websocket in disconnected:
            await self.remove_connection(websocket)
    
    async def send_to_bot(self, player_id: str, message: Dict[str, Any]) -> bool:
        """
        Send a message to a specific bot.
        
        Args:
            player_id: ID of the bot to send to
            message: Dictionary message to send
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        if player_id not in self.bot_connections:
            return False
        
        try:
            websocket = self.bot_connections[player_id]
            await self.send_to_websocket(websocket, message)
            return True
        except Exception as e:
            logger.error(f"Error sending to bot {player_id}: {e}")
            await self.remove_connection(self.bot_connections[player_id])
            return False
    
    async def send_to_websocket(self, websocket: WebSocketServerProtocol, message: Dict[str, Any]) -> None:
        """
        Send a message to a specific websocket connection.
        
        Args:
            websocket: WebSocket connection
            message: Dictionary message to send
        """
        try:
            await websocket.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            await self.remove_connection(websocket)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            await self.remove_connection(websocket)
    
    async def send_error(self, websocket: WebSocketServerProtocol, error_message: str) -> None:
        """
        Send an error message to a client.
        
        Args:
            websocket: WebSocket connection
            error_message: Error message to send
        """
        await self.send_to_websocket(websocket, {
            "type": "error",
            "message": error_message
        })
    
    async def handle_message(self, websocket: WebSocketServerProtocol, message: str) -> None:
        """
        Parse and route incoming messages.
        
        Args:
            websocket: WebSocket connection that sent the message
            message: Raw message string
        """
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "join_as_bot":
                player_id = data.get("player_id")
                if not player_id:
                    await self.send_error(websocket, "player_id is required")
                    return
                await self.add_bot_connection(websocket, player_id)
                
            elif message_type == "join_as_viewer":
                await self.add_viewer_connection(websocket)
                
            elif message_type == "move_command":
                await self.handle_move_command(websocket, data)
                
            else:
                await self.send_error(websocket, f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            await self.send_error(websocket, "Invalid JSON format")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await self.send_error(websocket, "Internal server error")
    
    async def handle_move_command(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]) -> None:
        """
        Handle a movement command from a bot.
        
        Args:
            websocket: WebSocket connection that sent the command
            data: Parsed command data
        """
        try:
            if not self.game_started or not self.game_state:
                await self.send_error(websocket, "Game not started")
                return
            
            if websocket not in self.connection_to_player:
                await self.send_error(websocket, "Not registered as bot")
                return
            
            player_id = self.connection_to_player[websocket]
            commands_data = data.get("commands", [])
            
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
                except (KeyError, ValueError) as e:
                    await self.send_error(websocket, f"Invalid command format: {e}")
                    return
            
            # Store commands for this turn
            self.turn_commands[player_id] = commands
            
            logger.info(f"Received {len(commands)} commands from {player_id}")
            
            # Commands are now processed on a timer, not when all players submit
                
        except Exception as e:
            logger.error(f"Error handling move command: {e}")
            await self.send_error(websocket, "Error processing command")
    
    async def start_grace_period(self) -> None:
        """
        Start the grace period for additional players to join.
        """
        try:
            if self.game_started:
                return
            
            logger.info(f"Starting {self.grace_period_seconds} second grace period for additional players")
            
            await self.broadcast_to_bots({
                "type": "grace_period_started",
                "duration_seconds": self.grace_period_seconds,
                "current_players": len(self.bot_connections)
            })
            
            await asyncio.sleep(self.grace_period_seconds)
            
            # Check again if we still have enough players and game hasn't started
            if not self.game_started and len(self.bot_connections) >= self.minimum_players:
                await self.start_game()
            else:
                logger.info("Grace period ended but conditions not met for game start")
                
        except asyncio.CancelledError:
            logger.info("Grace period was cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in grace period: {e}")
    
    async def start_game(self) -> None:
        """
        Initialize and start the game.
        """
        try:
            if self.game_started:
                return
            
            if len(self.bot_connections) < self.minimum_players and self.minimum_players > 1:
                logger.warning(f"Cannot start game - insufficient players ({len(self.bot_connections)}/{self.minimum_players})")
                return
            
            self.game_started = True
            logger.info(f"Game {self.game_id}: Starting with {len(self.bot_connections)} players")
            
            # Update game status and turn
            self.game_state.status = self.game_state.status.__class__.ACTIVE  # Set to ACTIVE
            self.game_state.current_turn = 1
            
            # Initialize game engine
            self.game_engine = GameEngine(self.game_state)
            
            # Start the turn timer
            self.turn_timer_task = asyncio.create_task(self.turn_timer_loop())
            
            # Broadcast initial game state
            await self.broadcast_game_state()
            
            logger.info(f"Game {self.game_id} started on turn {self.game_state.current_turn} with {self.turn_duration_seconds}s turn duration")
            
        except Exception as e:
            logger.error(f"Error starting game: {e}")
            self.game_started = False
    
    async def turn_timer_loop(self) -> None:
        """
        Main turn timer loop that processes turns at regular intervals.
        """
        try:
            while not self.game_state.status.value == "ended" and not self.shutdown_requested:
                # Wait for the turn duration
                await asyncio.sleep(self.turn_duration_seconds)
                
                # Process the current turn
                await self.process_turn()
                
                # Check if game is over
                if self.game_state.status.value == "ended":
                    break
                    
        except asyncio.CancelledError:
            logger.info("Turn timer loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in turn timer loop: {e}")
    
    async def process_turn(self) -> None:
        """
        Process all commands for the current turn and generate detailed turn data for animations.
        """
        try:
            if not self.game_engine or not self.game_state:
                return
            
            logger.info(f"Processing turn {self.game_state.current_turn}")
            
            # Capture state before turn processing for animation data
            vertices_before = {}
            for vertex in self.game_state.graph.vertices.values():
                vertices_before[vertex.id] = {
                    "controller": vertex.controller,
                    "units": vertex.units
                }
            
            # Process movements and capture move data
            move_animations = []
            unit_generation_data = []
            
            # Get all valid movements for animation (simplified since no movement costs now)
            valid_movements = []
            for player_id, commands in self.turn_commands.items():
                for command in commands:
                    if self.game_engine.process_movement_command(
                        command.player_id, command.from_vertex, 
                        command.to_vertex, command.units
                    ):
                        valid_movements.append(command)
                        
                        # Record movement for animation (no cost calculation needed)
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
                            "units_added": vertex.weight if vertex.weight > 0 else 0,  # Generation
                            "units_from_combat": max(0, vertex.units - vertex.weight)  # Combat result
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
                "eliminations": turn_result["eliminations"],
                "game_over": turn_result["game_over"],
                "turn_duration_seconds": self.turn_duration_seconds,  # Add timing info
                "max_turns": self.max_turns
            }
            
            # Broadcast detailed turn data first (for animations)
            logger.info(f"Broadcasting turn data: {len(move_animations)} moves, {len(unit_generation_data)} generations")
            await self.broadcast_to_viewers(detailed_turn_result)
            await self.broadcast_to_bots(detailed_turn_result)
            
            # Small delay to ensure turn_processed is handled before game_state
            await asyncio.sleep(0.05)
            
            # Then broadcast updated game state
            await self.broadcast_game_state()
            
            # Log turn results
            if turn_result["eliminations"]:
                logger.info(f"Players eliminated: {turn_result['eliminations']}")
            
            if turn_result["game_over"]:
                logger.info(f"Game over! Rankings: {self.game_state.final_rankings}")
                
                # Mark game as ended
                self.game_ended_time = time.time()
                
                # Stop the turn timer
                if self.turn_timer_task and not self.turn_timer_task.done():
                    self.turn_timer_task.cancel()
                    
                await self.broadcast_to_bots({
                    "type": "game_over",
                    "turn": self.game_state.current_turn,
                    "final_rankings": self.game_state.final_rankings
                })

                await self.broadcast_to_viewers({
                    "type": "game_over",
                    "turn": self.game_state.current_turn,
                    "final_rankings": self.game_state.final_rankings
                })
            
        except Exception as e:
            logger.error(f"Error processing turn: {e}")
    
    async def broadcast_game_state(self) -> None:
        """
        Broadcast the current game state to all clients.
        """
        if self.game_state:
            game_state_dict = self.game_state.to_dict()
            
            # Add server timing information
            game_state_dict["turn_duration_seconds"] = self.turn_duration_seconds
            game_state_dict["max_turns"] = self.max_turns
            
            await self.broadcast_to_bots(game_state_dict)
            await self.broadcast_to_viewers(game_state_dict)
    
    async def handle_client(self, websocket: WebSocketServerProtocol, path: str = "/") -> None:
        """
        Handle a new WebSocket client connection.
        
        Args:
            websocket: WebSocket connection
            path: Connection path (optional)
        """
        logger.info(f"New connection from {websocket.remote_address}")
        
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed: {websocket.remote_address}")
        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            await self.remove_connection(websocket)


class GameServer:
    """
    Multi-game WebSocket server that manages multiple GameInstance objects.
    Routes connections to specific games based on game_id.
    """
    
    def __init__(self, default_grid_width: int = 5, default_grid_height: int = 5, 
                 default_max_turns: int = 10, default_starting_units: int = 5,
                 default_turn_duration: float = 1.0):
        """
        Initialize the multi-game server.
        
        Args:
            default_grid_width: Default width for new games
            default_grid_height: Default height for new games
            default_max_turns: Default maximum turns for new games
            default_starting_units: Default starting units for new games
            default_turn_duration: Default turn duration for new games
        """
        self.games: Dict[str, GameInstance] = {}
        self.connection_to_game: Dict[WebSocketServerProtocol, str] = {}  # websocket -> game_id
        
        # Default settings for new games
        self.default_grid_width = default_grid_width
        self.default_grid_height = default_grid_height
        self.default_max_turns = default_max_turns
        self.default_starting_units = default_starting_units
        self.default_turn_duration = default_turn_duration
        
        # Server control
        self.shutdown_requested = False
        self.cleanup_task: Optional[asyncio.Task] = None
        
        logger.info(f"Multi-game server initialized")
    
    def create_game(self, game_id: str, grid_width: int = None, grid_height: int = None,
                   max_turns: int = None, starting_units: int = None, 
                   turn_duration: float = None) -> GameInstance:
        """Create a new game instance with the given ID."""
        if game_id in self.games:
            raise ValueError(f"Game {game_id} already exists")
        
        # Use provided values or defaults
        width = grid_width if grid_width is not None else self.default_grid_width
        height = grid_height if grid_height is not None else self.default_grid_height
        turns = max_turns if max_turns is not None else self.default_max_turns
        units = starting_units if starting_units is not None else self.default_starting_units
        duration = turn_duration if turn_duration is not None else self.default_turn_duration
        
        game = GameInstance(game_id, width, height, turns, units)
        game.turn_duration_seconds = duration
        self.games[game_id] = game
        
        logger.info(f"Created game {game_id} with {len(game.game_state.graph.vertices)} vertices")
        return game
    
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
        
        del self.games[game_id]
        logger.info(f"Removed game {game_id}")
        return True
    
    async def handle_client(self, websocket: WebSocketServerProtocol, path: str = "/") -> None:
        """
        Handle a new WebSocket client connection.
        
        Args:
            websocket: WebSocket connection
            path: Connection path (optional)
        """
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
    
    async def handle_message(self, websocket: WebSocketServerProtocol, message: str) -> None:
        """
        Parse and route incoming messages to the appropriate game.
        
        Args:
            websocket: WebSocket connection that sent the message
            message: Raw message string
        """
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "join_as_bot":
                await self.handle_bot_join(websocket, data)
                
            elif message_type == "join_as_viewer":
                await self.handle_viewer_join(websocket, data)
                
            elif message_type == "move_command":
                await self.route_to_game(websocket, data)
                
            elif message_type == "create_game":
                await self.handle_create_game(websocket, data)
                
            elif message_type == "list_games":
                await self.handle_list_games(websocket)
                
            else:
                await self.send_error(websocket, f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            await self.send_error(websocket, "Invalid JSON format")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await self.send_error(websocket, "Internal server error")
    
    async def handle_bot_join(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]) -> None:
        """Handle bot joining a specific game."""
        player_id = data.get("player_id")
        game_id = data.get("game_id")
        
        if not player_id:
            await self.send_error(websocket, "player_id is required")
            return
            
        if not game_id:
            await self.send_error(websocket, "game_id is required")
            return
        
        # Get or create the game
        game = self.games.get(game_id)
        if not game:
            # Auto-create game if it doesn't exist
            try:
                game = self.create_game(game_id)
            except Exception as e:
                await self.send_error(websocket, f"Failed to create game {game_id}: {e}")
                return
        
        # Add bot to the specific game
        success = await game.add_bot_connection(websocket, player_id)
        if success:
            self.connection_to_game[websocket] = game_id
        
    async def handle_viewer_join(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]) -> None:
        """Handle viewer joining a specific game."""
        game_id = data.get("game_id")
        
        if not game_id:
            await self.send_error(websocket, "game_id is required")
            return
        
        # Get or create the game
        game = self.games.get(game_id)
        if not game:
            # Auto-create game if it doesn't exist
            try:
                game = self.create_game(game_id)
            except Exception as e:
                await self.send_error(websocket, f"Failed to create game {game_id}: {e}")
                return
        
        # Add viewer to the specific game
        success = await game.add_viewer_connection(websocket)
        if success:
            self.connection_to_game[websocket] = game_id
    
    async def handle_create_game(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]) -> None:
        """Handle explicit game creation request."""
        game_id = data.get("game_id")
        if not game_id:
            await self.send_error(websocket, "game_id is required")
            return
        
        try:
            grid_width = data.get("grid_width")
            grid_height = data.get("grid_height")
            max_turns = data.get("max_turns")
            starting_units = data.get("starting_units")
            turn_duration = data.get("turn_duration")
            
            game = self.create_game(game_id, grid_width, grid_height, max_turns, starting_units, turn_duration)
            
            await self.send_message(websocket, {
                "type": "game_created",
                "game_id": game_id,
                "message": f"Game {game_id} created successfully"
            })
            
        except ValueError as e:
            await self.send_error(websocket, str(e))
        except Exception as e:
            await self.send_error(websocket, f"Failed to create game: {e}")
    
    async def handle_list_games(self, websocket: WebSocketServerProtocol) -> None:
        """Handle request to list all games."""
        games_info = []
        for game_id, game in self.games.items():
            games_info.append({
                "game_id": game_id,
                "status": game.game_state.status.value if game.game_state else "unknown",
                "players": len(game.bot_connections),
                "viewers": len(game.viewer_connections),
                "turn": game.game_state.current_turn if game.game_state else 0
            })
        
        await self.send_message(websocket, {
            "type": "games_list",
            "games": games_info
        })
    
    async def route_to_game(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]) -> None:
        """Route a message to the appropriate game instance."""
        game_id = self.connection_to_game.get(websocket)
        if not game_id:
            await self.send_error(websocket, "Not connected to any game")
            return
        
        game = self.games.get(game_id)
        if not game:
            await self.send_error(websocket, f"Game {game_id} no longer exists")
            return
        
        # Forward the message to the game instance
        if data.get("type") == "move_command":
            await game.handle_move_command(websocket, data)
    
    async def cleanup_connection(self, websocket: WebSocketServerProtocol) -> None:
        """Clean up a disconnected client."""
        try:
            game_id = self.connection_to_game.get(websocket)
            if game_id:
                game = self.games.get(game_id)
                if game:
                    await game.remove_connection(websocket)
                del self.connection_to_game[websocket]
        except Exception as e:
            logger.error(f"Error cleaning up connection: {e}")
    
    async def send_error(self, websocket: WebSocketServerProtocol, error_message: str) -> None:
        """Send an error message to a client."""
        await self.send_message(websocket, {
            "type": "error",
            "message": error_message
        })
    
    async def send_message(self, websocket: WebSocketServerProtocol, message: Dict[str, Any]) -> None:
        """Send a message to a specific websocket connection."""
        try:
            await websocket.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            pass  # Connection already closed
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
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
        """
        Periodically check for and remove games that should be cleaned up.
        
        Args:
            check_interval_seconds: How often to check for cleanup candidates (default: 1 second)
            cleanup_delay_seconds: How long to wait after game ends before cleanup (default: 60 seconds)
        """
        try:
            while not self.shutdown_requested:
                await asyncio.sleep(check_interval_seconds)
                
                # Find games to clean up
                games_to_remove = []
                for game_id, game in self.games.items():
                    if game.should_be_cleaned_up(cleanup_delay_seconds):
                        games_to_remove.append(game_id)
                
                # Remove the games and log only when actually removing
                for game_id in games_to_remove:
                    game = self.games[game_id]
                    reason = "no connections" if game.get_total_connections() == 0 else "ended >1 minute ago"
                    logger.info(f"Cleaning up game {game_id} ({reason})")
                    self.remove_game(game_id)
                    
        except asyncio.CancelledError:
            logger.info("Game cleanup loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in cleanup loop: {e}")


async def keyboard_input_handler(server: GameServer) -> None:
    """
    Handle keyboard input for server commands.
    
    Args:
        server: The game server instance
    """
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
                # Read keyboard input in a non-blocking way
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
                                logger.info(f"  {game_id}: {status}, Turn {turn}/{game.max_turns}, {len(game.bot_connections)} bots, {len(game.viewer_connections)} viewers")
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
                        game = server.get_game(game_id)
                        if not game:
                            logger.warning(f"Game {game_id} does not exist")
                        else:
                            game.reset_game()
                            logger.info(f"Game {game_id} reset complete")
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
                        elif len(game.bot_connections) == 0:
                            logger.warning(f"Cannot start game {game_id} - no players connected")
                        else:
                            logger.info(f"Force starting game {game_id} with {len(game.bot_connections)} players")
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
                                await game.broadcast_game_state()
                                
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
                                
                                # If game is running, the change will take effect on the next turn
                                if game.game_started:
                                    logger.info(f"Game {game_id} is running - new duration will apply to subsequent turns")
                                
                                # Broadcast updated game state to notify clients
                                await game.broadcast_game_state()
                                
                    except ValueError:
                        logger.warning("Duration must be a valid number")
                    except IndexError:
                        logger.warning("Usage: set duration <game_id> <seconds>")
                    except Exception as e:
                        logger.error(f"Failed to set turn duration: {e}")
                    
                elif command == "status":
                    logger.info(f"=== Multi-Game Server Status ===")
                    logger.info(f"Active Games: {len(server.games)}")
                    logger.info(f"Total Connections: {len(server.connection_to_game)}")
                    if server.games:
                        total_bots = sum(len(game.bot_connections) for game in server.games.values())
                        total_viewers = sum(len(game.viewer_connections) for game in server.games.values())
                        logger.info(f"Total Bots: {total_bots}")
                        logger.info(f"Total Viewers: {total_viewers}")
                        logger.info(f"Games: {list(server.games.keys())}")
                        
                        # Show detailed game info
                        for game_id, game in server.games.items():
                            status = "Active" if game.game_started else "Waiting"
                            turn = game.game_state.current_turn if game.game_state else 0
                            logger.info(f"  {game_id}: {status}, Turn {turn}/{game.max_turns}, Duration: {game.turn_duration_seconds}s")
                    
                elif command == "help":
                    logger.info("=== Available Commands ===")
                    logger.info("  'quit' or 'exit' - Stop the server")
                    logger.info("  'games' - List all active games")
                    logger.info("  'create <game_id>' - Create a new game")
                    logger.info("  'start <game_id>' - Force start a specific game")
                    logger.info("  'reset <game_id>' - Reset a specific game")
                    logger.info("  'set turns <game_id> <max_turns>' - Set maximum turns for a game (before start)")
                    logger.info("  'set duration <game_id> <seconds>' - Set turn duration for a game")
                    logger.info("  'status' - Show current server status")
                    logger.info("  'help' - Show this help message")
                    
                elif command == "":
                    # Empty command, just continue
                    continue
                    
                else:
                    logger.warning(f"Unknown command: '{command}'. Type 'help' for available commands.")
                    
            except EOFError:
                # Handle Ctrl+D
                logger.info("EOF received, shutting down server")
                server.shutdown_requested = True
                break
            except KeyboardInterrupt:
                # Handle Ctrl+C
                logger.info("Keyboard interrupt received, shutting down server")
                server.shutdown_requested = True
                break
            except Exception as e:
                logger.error(f"Error processing keyboard input: {e}")
                
    except Exception as e:
        logger.error(f"Error in keyboard handler: {e}")
    finally:
        logger.info("Keyboard command handler shutting down")


async def run_server(host: str = "localhost", port: int = 8765, 
                    grid_width: int = 5, grid_height: int = 5, 
                    turn_duration: float = 1.0) -> None:
    """
    Run the multi-game server with keyboard command support.
    
    Args:
        host: Host address to bind to
        port: Port to listen on
        grid_width: Default width for new games
        grid_height: Default height for new games
        turn_duration: Default duration of each turn in seconds
    """
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
            if game.grace_period_task and not game.grace_period_task.done():
                game.grace_period_task.cancel()
            if game.turn_timer_task and not game.turn_timer_task.done():
                game.turn_timer_task.cancel()
        
        # Close WebSocket server
        websocket_server.close()
        await websocket_server.wait_closed()
        
        # Cancel keyboard task if still running
        if not keyboard_task.done():
            keyboard_task.cancel()
        
        # Notify any remaining clients in all games
        for game in server.games.values():
            await game.broadcast_to_bots({
                "type": "server_shutdown",
                "message": "Server is shutting down"
            })
            await game.broadcast_to_viewers({
                "type": "server_shutdown", 
                "message": "Server is shutting down"
            })
        
        logger.info("Server shutdown complete")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run test client
        asyncio.run(test_client())
    elif len(sys.argv) > 1 and sys.argv[1] == "test_game":
        # Run two-player game test
        asyncio.run(test_two_player_game())
    elif len(sys.argv) > 1 and sys.argv[1] == "test_removal":
        # Run early assignment and removal test
        asyncio.run(test_early_assignment_and_removal())
    elif len(sys.argv) > 1 and sys.argv[1] == "fast":
        # Run server with fast turns for testing
        try:
            asyncio.run(run_server(turn_duration=0.5))
        except KeyboardInterrupt:
            logger.info("Fast server shutting down...")
    else:
        # Run server
        try:
            asyncio.run(run_server())
        except KeyboardInterrupt:
            logger.info("Server shutting down...")
