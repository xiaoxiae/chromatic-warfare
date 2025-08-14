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
from typing import Dict, Set, Optional, Any
from websockets.server import WebSocketServerProtocol
from game_core import GameState, GameEngine, Command, PlayerStatus

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GameServer:
    """
    WebSocket server for managing the turn-based strategy game.
    Handles bot connections, viewer connections, and message routing.
    """
    
    def __init__(self, grid_width: int = 5, grid_height: int = 5, max_turns: int = 100, starting_units: int = 5):
        """
        Initialize the game server.
        
        Args:
            grid_width: Width of the game grid
            grid_height: Height of the game grid
            max_turns: Maximum number of turns before game ends
            starting_units: Number of units each player starts with
        """
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
        
        # Server control
        self.shutdown_requested = False
    
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
            
            logger.info(f"Game reset complete. Grid regenerated with {len(self.game_state.graph.vertices)} vertices")
            logger.info(f"Waiting for players to reconnect...")
            
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
            
            logger.info(f"Force starting game with {len(self.bot_connections)} players")
            
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
            
            logger.info(f"Bot {player_id} connected and assigned vertex {starting_vertex.id}. Total bots: {len(self.bot_connections)}")
            
            # Send connection confirmation
            await self.send_to_websocket(websocket, {
                "type": "connection_confirmed",
                "player_id": player_id,
                "starting_vertex": starting_vertex.id,
                "message": f"Successfully connected and assigned starting vertex {starting_vertex.id}"
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
        """
        Register a new viewer connection.
        
        Args:
            websocket: WebSocket connection
            
        Returns:
            True if connection was added successfully, False otherwise
        """
        try:
            self.viewer_connections.add(websocket)
            logger.info(f"Viewer connected. Total viewers: {len(self.viewer_connections)}")
            
            # Send current game state
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
            logger.info(f"Starting game with {len(self.bot_connections)} players")
            
            # Update game status and turn
            self.game_state.status = self.game_state.status.__class__.ACTIVE  # Set to ACTIVE
            self.game_state.current_turn = 1
            
            # Initialize game engine
            self.game_engine = GameEngine(self.game_state)
            
            # Start the turn timer
            self.turn_timer_task = asyncio.create_task(self.turn_timer_loop())
            
            # Broadcast initial game state
            await self.broadcast_game_state()
            
            logger.info(f"Game started on turn {self.game_state.current_turn} with {self.turn_duration_seconds}s turn duration")
            
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
                            "cost": self.game_engine._calculate_movement_cost(
                                command.from_vertex, command.to_vertex
                            )
                        })
            
            # Process the turn using the existing engine
            turn_result = self.game_engine.process_turn(self.turn_commands)
            
            # Capture unit generation data by comparing before/after states
            for vertex in self.game_state.graph.vertices.values():
                before = vertices_before[vertex.id]
                
                # Calculate expected units after movements but before generation
                expected_after_moves = before["units"]
                
                # Subtract units that moved out
                for move in move_animations:
                    if move["from_vertex"] == vertex.id and move["player_id"] == before["controller"]:
                        expected_after_moves -= (move["units"] + move["cost"])
                
                # Add units that moved in (after conflict resolution)
                # This is approximated since conflict resolution is complex
                current_controller = vertex.controller
                if current_controller is not None:
                    # If vertex changed hands or gained units beyond movements, record generation
                    units_generated = vertex.units - expected_after_moves
                    if units_generated > 0:
                        # This includes both successful attacks and unit generation
                        # For animation purposes, we'll show the net increase
                        unit_generation_data.append({
                            "vertex_id": vertex.id,
                            "controller": current_controller,
                            "units_added": vertex.weight if current_controller == before["controller"] else 0,
                            "units_from_combat": max(0, units_generated - vertex.weight) if current_controller == before["controller"] else vertex.units
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
                "winner": turn_result.get("winner")
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
                logger.info(f"Game over! Winners: {turn_result['winner']}")
                
                # Stop the turn timer
                if self.turn_timer_task and not self.turn_timer_task.done():
                    self.turn_timer_task.cancel()
                    
                await self.broadcast_to_bots({
                    "type": "game_over",
                    "winner": turn_result["winner"],
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


async def keyboard_input_handler(server: GameServer) -> None:
    """
    Handle keyboard input for server commands.
    
    Args:
        server: The game server instance
    """
    logger.info("Keyboard command handler started. Available commands:")
    logger.info("  'quit' - Stop the server")
    logger.info("  'start' - Force start game (requires at least 1 player)")
    logger.info("  'reset' - Reset game state and wait for new players")
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
                    
                elif command == "start":
                    if server.game_started:
                        logger.warning("Game is already running")
                    elif len(server.bot_connections) == 0:
                        logger.warning("Cannot start game - no players connected")
                    else:
                        logger.info(f"Force starting game with {len(server.bot_connections)} players")
                        success = server.force_start_game()
                        if success:
                            logger.info("Game force-started successfully")
                        else:
                            logger.warning("Failed to force start game")
                            
                elif command == "reset":
                    if server.game_started:
                        logger.info("Resetting active game...")
                    else:
                        logger.info("Resetting server state...")
                    server.reset_game()
                    logger.info("Game reset complete")
                    
                elif command == "status":
                    game_status = "Active" if server.game_started else "Waiting"
                    turn = server.game_state.current_turn if server.game_state else 0
                    logger.info(f"=== Server Status ===")
                    logger.info(f"Game Status: {game_status}")
                    logger.info(f"Current Turn: {turn}")
                    logger.info(f"Connected Bots: {len(server.bot_connections)}")
                    logger.info(f"Connected Viewers: {len(server.viewer_connections)}")
                    logger.info(f"Grid Size: {server.grid_width}x{server.grid_height}")
                    logger.info(f"Turn Duration: {server.turn_duration_seconds}s")
                    if server.bot_connections:
                        logger.info(f"Bot Players: {list(server.bot_connections.keys())}")
                    
                elif command == "help":
                    logger.info("=== Available Commands ===")
                    logger.info("  'quit' or 'exit' - Stop the server")
                    logger.info("  'start' - Force start game (requires at least 1 player)")
                    logger.info("  'reset' - Reset game state and wait for new players")
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
    Run the game server with keyboard command support.
    
    Args:
        host: Host address to bind to
        port: Port to listen on
        grid_width: Width of the game grid
        grid_height: Height of the game grid
        turn_duration: Duration of each turn in seconds
    """
    server = GameServer(grid_width=grid_width, grid_height=grid_height)
    server.turn_duration_seconds = turn_duration
    
    logger.info(f"Starting game server on {host}:{port}")
    logger.info(f"Grid size: {grid_width}x{grid_height}")
    logger.info(f"Turn duration: {turn_duration}s")
    logger.info(f"Grid generated with {len(server.game_state.graph.vertices)} vertices")
    logger.info(f"Waiting for at least {server.minimum_players} bot connections...")
    logger.info("")
    logger.info("Server is ready for connections and keyboard commands.")
    
    # Start the WebSocket server
    websocket_server = await websockets.serve(server.handle_client, host, port)
    
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
        
        # Cancel any running tasks
        if server.grace_period_task and not server.grace_period_task.done():
            server.grace_period_task.cancel()
        if server.turn_timer_task and not server.turn_timer_task.done():
            server.turn_timer_task.cancel()
        
        # Close WebSocket server
        websocket_server.close()
        await websocket_server.wait_closed()
        
        # Cancel keyboard task if still running
        if not keyboard_task.done():
            keyboard_task.cancel()
        
        # Notify any remaining clients
        await server.broadcast_to_bots({
            "type": "server_shutdown",
            "message": "Server is shutting down"
        })
        await server.broadcast_to_viewers({
            "type": "server_shutdown", 
            "message": "Server is shutting down"
        })
        
        logger.info("Server shutdown complete")


async def test_early_assignment_and_removal():
    """
    Test early vertex assignment and player removal during grace period.
    """
    async def bot_client(player_id: str, delay: float = 1, disconnect_after: float = None):
        """
        Simulate a bot client that connects and optionally disconnects.
        
        Args:
            player_id: Unique identifier for this bot
            delay: Delay before connecting
            disconnect_after: If set, disconnect after this many seconds
        """
        await asyncio.sleep(delay)
        
        try:
            uri = "ws://localhost:8765"
            logger.info(f"[{player_id}] Connecting to {uri}")
            
            async with websockets.connect(uri) as websocket:
                # Join as a bot
                join_message = {
                    "type": "join_as_bot",
                    "player_id": player_id
                }
                await websocket.send(json.dumps(join_message))
                logger.info(f"[{player_id}] Sent join request")
                
                # Wait for confirmation
                message = await websocket.recv()
                data = json.loads(message)
                
                if data.get("type") == "connection_confirmed":
                    starting_vertex = data.get("starting_vertex")
                    logger.info(f"[{player_id}] Confirmed, assigned vertex {starting_vertex}")
                
                # If set to disconnect, do so after specified time
                if disconnect_after:
                    await asyncio.sleep(disconnect_after)
                    logger.info(f"[{player_id}] Disconnecting as planned")
                    return
                
                # Otherwise wait for grace period or game start
                while True:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                        data = json.loads(message)
                        message_type = data.get("type")
                        
                        if message_type == "grace_period_started":
                            duration = data.get("duration_seconds", 0)
                            logger.info(f"[{player_id}] Grace period started: {duration}s")
                            
                        elif message_type == "grace_period_canceled":
                            reason = data.get("reason")
                            logger.info(f"[{player_id}] Grace period canceled: {reason}")
                            break
                            
                        elif message_type == "game_state":
                            game_status = data.get("game_status")
                            if game_status == "active":
                                logger.info(f"[{player_id}] Game started!")
                                break
                                
                        elif message_type == "error":
                            error_msg = data.get("message", "Unknown error")
                            logger.warning(f"[{player_id}] Error: {error_msg}")
                            
                    except asyncio.TimeoutError:
                        logger.warning(f"[{player_id}] Timeout waiting for server message")
                        break
                    except websockets.exceptions.ConnectionClosed:
                        logger.info(f"[{player_id}] Connection closed by server")
                        break
                
        except Exception as e:
            logger.error(f"[{player_id}] Client error: {e}")
    
    # Test scenario: 3 players connect, 1 disconnects, grace period should be canceled
    logger.info("=== Testing Early Assignment and Grace Period Cancellation ===")
    
    bot1_task = asyncio.create_task(bot_client("test_bot_1", delay=0.5))
    bot2_task = asyncio.create_task(bot_client("test_bot_2", delay=1.0))
    bot3_task = asyncio.create_task(bot_client("test_bot_3", delay=1.5, disconnect_after=3.0))  # Disconnects
    
    # Wait for all bots
    try:
        await asyncio.gather(bot1_task, bot2_task, bot3_task)
        logger.info("=== Test completed ===")
    except Exception as e:
        logger.error(f"Test error: {e}")


async def test_two_player_game():
    """
    Test a complete two-player game flow including connection, gameplay, and natural termination.
    """
    async def bot_client(player_id: str, delay: float = 1):
        """
        Simulate a bot client that connects, plays until game ends, and then disconnects.
        
        Args:
            player_id: Unique identifier for this bot
            delay: Delay before connecting (to simulate different connection times)
        """
        await asyncio.sleep(delay)
        
        try:
            uri = "ws://localhost:8765"
            logger.info(f"[{player_id}] Connecting to {uri}")
            
            async with websockets.connect(uri) as websocket:
                # Join as a bot
                join_message = {
                    "type": "join_as_bot",
                    "player_id": player_id
                }
                await websocket.send(json.dumps(join_message))
                logger.info(f"[{player_id}] Sent join request")
                
                game_over = False
                my_status = "active"
                
                while not game_over:
                    try:
                        # Wait for message from server with longer timeout for full game
                        message = await asyncio.wait_for(websocket.recv(), timeout=60.0)
                        data = json.loads(message)
                        message_type = data.get("type")
                        
                        if message_type == "connection_confirmed":
                            starting_vertex = data.get("starting_vertex")
                            logger.info(f"[{player_id}] Successfully joined, assigned vertex {starting_vertex}")
                            
                        elif message_type == "grace_period_started":
                            duration = data.get("duration_seconds", 0)
                            logger.info(f"[{player_id}] Grace period started: {duration}s")
                            
                        elif message_type == "game_state":
                            game_status = data.get("game_status")
                            current_turn = data.get("turn", 0)
                            
                            # Check my current status
                            players = data.get("players", [])
                            for player in players:
                                if player.get("id") == player_id:
                                    my_status = player.get("status", "active")
                                    break
                            
                            if game_status == "active":
                                logger.info(f"[{player_id}] Turn {current_turn}, my status: {my_status}")
                                
                                # Only send commands if I'm still active
                                if my_status == "active":
                                    await asyncio.sleep(delay)
                                    await send_strategic_commands(websocket, player_id, data)
                                else:
                                    logger.info(f"[{player_id}] Eliminated, not sending commands")
                                    
                            elif game_status == "ended":
                                winner = data.get("winner", [])
                                rankings = data.get("final_rankings", [])
                                logger.info(f"[{player_id}] Game ended!")
                                logger.info(f"[{player_id}] Final rankings: {rankings}")
                                logger.info(f"[{player_id}] Winners: {winner}")
                                game_over = True
                                
                        elif message_type == "game_over":
                            winner = data.get("winner", [])
                            rankings = data.get("final_rankings", [])
                            logger.info(f"[{player_id}] Game over message received!")
                            logger.info(f"[{player_id}] Final rankings: {rankings}")
                            logger.info(f"[{player_id}] Winners: {winner}")
                            game_over = True
                            
                        elif message_type == "error":
                            error_msg = data.get("message", "Unknown error")
                            logger.warning(f"[{player_id}] Error: {error_msg}")
                            
                    except asyncio.TimeoutError:
                        logger.warning(f"[{player_id}] Timeout waiting for server message")
                        break
                    except websockets.exceptions.ConnectionClosed:
                        logger.info(f"[{player_id}] Connection closed by server")
                        break
                
                logger.info(f"[{player_id}] Game completed, disconnecting gracefully")
                
        except Exception as e:
            logger.error(f"[{player_id}] Client error: {e}")
    
    async def send_strategic_commands(websocket, player_id: str, game_state: dict):
        """
        Send strategic movement commands based on current game state.
        
        Args:
            websocket: WebSocket connection
            player_id: ID of the player
            game_state: Current game state data
        """
        try:
            # Find vertices controlled by this player
            vertices = game_state.get("graph", {}).get("vertices", [])
            edges = game_state.get("graph", {}).get("edges", [])
            
            my_vertices = [v for v in vertices if v.get("controller") == player_id]
            enemy_vertices = [v for v in vertices if v.get("controller") not in [player_id, None]]
            neutral_vertices = [v for v in vertices if v.get("controller") is None]
            
            if not my_vertices:
                logger.warning(f"[{player_id}] No controlled vertices found")
                # Send empty command to advance turn
                move_message = {"type": "move_command", "commands": []}
                await websocket.send(json.dumps(move_message))
                return
            
            logger.info(f"[{player_id}] I control {len(my_vertices)} vertices, {len(enemy_vertices)} enemy, {len(neutral_vertices)} neutral")
            
            # Build adjacency map for quick lookups
            adjacency = {}
            for edge in edges:
                from_id = edge["from"]
                to_id = edge["to"]
                if from_id not in adjacency:
                    adjacency[from_id] = []
                adjacency[from_id].append(to_id)
            
            commands = []
            
            # Strategy: Prioritize attacking enemies, then expanding to neutrals
            for vertex in my_vertices:
                vertex_id = vertex["id"]
                units = vertex["units"]
                
                if units <= 1:  # Keep at least 1 unit for defense
                    continue
                
                adjacent_ids = adjacency.get(vertex_id, [])
                if not adjacent_ids:
                    continue
                
                # Find adjacent targets
                adjacent_enemies = []
                adjacent_neutrals = []
                
                for adj_id in adjacent_ids:
                    adj_vertex = next((v for v in vertices if v["id"] == adj_id), None)
                    if adj_vertex:
                        if adj_vertex.get("controller") not in [player_id, None]:
                            adjacent_enemies.append(adj_vertex)
                        elif adj_vertex.get("controller") is None:
                            adjacent_neutrals.append(adj_vertex)
                
                # Prioritize weak enemies we can defeat
                for enemy in adjacent_enemies:
                    enemy_units = enemy.get("units", 0)
                    if units > enemy_units + 1:  # Can defeat and have units left
                        units_to_send = min(enemy_units + 2, units - 1)  # Send enough to win + some extra
                        commands.append({
                            "from": vertex_id,
                            "to": enemy["id"],
                            "units": units_to_send
                        })
                        logger.info(f"[{player_id}] Attacking enemy vertex {enemy['id']} with {units_to_send} units")
                        units -= units_to_send
                        break
                
                # If no good enemy targets, expand to neutrals
                if units > 1 and not any(cmd["from"] == vertex_id for cmd in commands):
                    for neutral in adjacent_neutrals:
                        neutral_weight = neutral.get("weight", 1)
                        if units > neutral_weight + 1:  # Can afford the cost
                            units_to_send = min(2, units - 1)  # Send small expansion force
                            commands.append({
                                "from": vertex_id,
                                "to": neutral["id"],
                                "units": units_to_send
                            })
                            logger.info(f"[{player_id}] Expanding to neutral vertex {neutral['id']} with {units_to_send} units")
                            break
            
            # Send commands or empty list to advance turn
            move_message = {
                "type": "move_command",
                "commands": commands
            }
            await websocket.send(json.dumps(move_message))
            
            if commands:
                logger.info(f"[{player_id}] Sent {len(commands)} strategic commands")
            else:
                logger.info(f"[{player_id}] No viable moves, sent empty command list")
                
        except Exception as e:
            logger.error(f"[{player_id}] Error sending commands: {e}")
            # Send empty command to avoid blocking the game
            move_message = {"type": "move_command", "commands": []}
            await websocket.send(json.dumps(move_message))
    
    # Start both bot clients with slight delay
    logger.info("=== Starting Complete Two-Player Game Test ===")
    
    bot1_task = asyncio.create_task(bot_client("test_bot_1", delay=0.5))
    bot2_task = asyncio.create_task(bot_client("test_bot_2", delay=1.0))
    
    # Wait for both bots to complete
    try:
        await asyncio.gather(bot1_task, bot2_task)
        logger.info("=== Complete Two-Player Game Test Finished ===")
    except Exception as e:
        logger.error(f"Test error: {e}")


async def test_client():
    """
    Simple test client to verify server connectivity.
    """
    try:
        uri = "ws://localhost:8765"
        logger.info(f"Connecting test client to {uri}")
        
        async with websockets.connect(uri) as websocket:
            # Join as a bot
            join_message = {
                "type": "join_as_bot",
                "player_id": "test_bot_1"
            }
            await websocket.send(json.dumps(join_message))
            
            # Wait for confirmation
            response = await websocket.recv()
            data = json.loads(response)
            logger.info(f"Server response: {data}")
            
            # Send a test command (will fail since game isn't started, but tests message routing)
            test_command = {
                "type": "move_command",
                "commands": [
                    {"from": 0, "to": 1, "units": 1}
                ]
            }
            await websocket.send(json.dumps(test_command))
            
            # Wait for response
            response = await websocket.recv()
            data = json.loads(response)
            logger.info(f"Command response: {data}")
            
    except Exception as e:
        logger.error(f"Test client error: {e}")


async def test_periodic_turns():
    """
    Test the new periodic turn system with faster turns for demonstration.
    """
    async def bot_client(player_id: str, delay: float = 1):
        """
        Simulate a bot client with faster command submission for periodic turn testing.
        """
        await asyncio.sleep(delay)
        
        try:
            uri = "ws://localhost:8765"
            logger.info(f"[{player_id}] Connecting to {uri}")
            
            async with websockets.connect(uri) as websocket:
                # Join as a bot
                join_message = {
                    "type": "join_as_bot",
                    "player_id": player_id
                }
                await websocket.send(json.dumps(join_message))
                logger.info(f"[{player_id}] Sent join request")
                
                game_over = False
                turn_count = 0
                
                while not game_over and turn_count < 20:  # Limit for demo
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                        data = json.loads(message)
                        message_type = data.get("type")
                        
                        if message_type == "connection_confirmed":
                            starting_vertex = data.get("starting_vertex")
                            logger.info(f"[{player_id}] Confirmed, assigned vertex {starting_vertex}")
                            
                        elif message_type == "grace_period_started":
                            duration = data.get("duration_seconds", 0)
                            logger.info(f"[{player_id}] Grace period: {duration}s")
                            
                        elif message_type == "turn_processed":
                            turn = data.get("turn")
                            moves = data.get("move_animations", [])
                            generation = data.get("unit_generation", [])
                            logger.info(f"[{player_id}] Turn {turn} processed: {len(moves)} moves, {len(generation)} generations")
                            
                        elif message_type == "game_state":
                            game_status = data.get("game_status")
                            current_turn = data.get("turn", 0)
                            
                            if game_status == "active":
                                turn_count = current_turn
                                logger.info(f"[{player_id}] Turn {current_turn}")
                                
                                # Send commands quickly for periodic turn testing
                                await send_quick_commands(websocket, player_id, data)
                                    
                            elif game_status == "ended":
                                winner = data.get("winner", [])
                                logger.info(f"[{player_id}] Game ended! Winners: {winner}")
                                game_over = True
                                
                        elif message_type == "game_over":
                            winner = data.get("winner", [])
                            logger.info(f"[{player_id}] Game over! Winners: {winner}")
                            game_over = True
                            
                    except asyncio.TimeoutError:
                        logger.warning(f"[{player_id}] Timeout - continuing")
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        logger.info(f"[{player_id}] Connection closed")
                        break
                
                logger.info(f"[{player_id}] Test completed after {turn_count} turns")
                
        except Exception as e:
            logger.error(f"[{player_id}] Client error: {e}")
    
    async def send_quick_commands(websocket, player_id: str, game_state: dict):
        """Send simple commands quickly for testing periodic turns."""
        try:
            vertices = game_state.get("graph", {}).get("vertices", [])
            edges = game_state.get("graph", {}).get("edges", [])
            
            my_vertices = [v for v in vertices if v.get("controller") == player_id]
            
            if not my_vertices:
                # Send empty command
                move_message = {"type": "move_command", "commands": []}
                await websocket.send(json.dumps(move_message))
                return
            
            # Build adjacency map
            adjacency = {}
            for edge in edges:
                from_id = edge["from"]
                to_id = edge["to"]
                if from_id not in adjacency:
                    adjacency[from_id] = []
                adjacency[from_id].append(to_id)
            
            commands = []
            
            # Simple strategy: attack any adjacent enemy or expand to neutral
            for vertex in my_vertices[:2]:  # Limit to 2 vertices to keep it simple
                vertex_id = vertex["id"]
                units = vertex["units"]
                
                if units <= 1:
                    continue
                
                adjacent_ids = adjacency.get(vertex_id, [])
                if not adjacent_ids:
                    continue
                
                # Find a target
                for adj_id in adjacent_ids:
                    adj_vertex = next((v for v in vertices if v["id"] == adj_id), None)
                    if adj_vertex and adj_vertex.get("controller") != player_id:
                        # Attack or expand
                        units_to_send = min(2, units - 1)
                        if units_to_send > 0:
                            commands.append({
                                "from": vertex_id,
                                "to": adj_id,
                                "units": units_to_send
                            })
                            break
            
            # Send commands
            move_message = {
                "type": "move_command", 
                "commands": commands
            }
            await websocket.send(json.dumps(move_message))
            
            if commands:
                logger.info(f"[{player_id}] Sent {len(commands)} quick commands")
                
        except Exception as e:
            logger.error(f"[{player_id}] Error sending quick commands: {e}")
            # Send empty command to avoid blocking
            move_message = {"type": "move_command", "commands": []}
            await websocket.send(json.dumps(move_message))
    
    # Start test with faster turns
    logger.info("=== Testing Periodic Turn System (0.5s turns) ===")
    
    bot1_task = asyncio.create_task(bot_client("fast_bot_1", delay=0.3))
    bot2_task = asyncio.create_task(bot_client("fast_bot_2", delay=0.6))
    
    try:
        await asyncio.gather(bot1_task, bot2_task)
        logger.info("=== Periodic Turn Test Completed ===")
    except Exception as e:
        logger.error(f"Test error: {e}")


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
    elif len(sys.argv) > 1 and sys.argv[1] == "test_periodic":
        # Run periodic turn test
        asyncio.run(test_periodic_turns())
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
