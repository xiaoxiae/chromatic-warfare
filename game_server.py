"""
WebSocket Game Server for Turn-Based Strategy Game

This module provides the WebSocket server infrastructure for hosting the turn-based
strategy game, managing bot connections, viewer connections, and message routing.
"""

import asyncio
import json
import logging
import websockets
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
        
        # Game state
        self.game_state: Optional[GameState] = None
        self.game_engine: Optional[GameEngine] = None
        self.game_started = False
        
        # Grace period for additional connections
        self.grace_period_seconds = 15
        self.minimum_players = 2
        
        # Turn management
        self.turn_commands: Dict[str, list] = {}  # player_id -> list of commands
        self.turn_timeout_seconds = 30
    
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
            
            # Add connection
            self.bot_connections[player_id] = websocket
            self.connection_to_player[websocket] = player_id
            
            logger.info(f"Bot {player_id} connected. Total bots: {len(self.bot_connections)}")
            
            # Send connection confirmation
            await self.send_to_websocket(websocket, {
                "type": "connection_confirmed",
                "player_id": player_id,
                "message": "Successfully connected to game server"
            })
            
            # Check if we can start the game
            if len(self.bot_connections) >= self.minimum_players and not self.game_started:
                asyncio.create_task(self.start_grace_period())
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding bot connection for {player_id}: {e}")
            return False
    
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
            
            # Send current game state if game is active
            if self.game_state:
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
            
            # Remove from bots
            if websocket in self.connection_to_player:
                player_id = self.connection_to_player[websocket]
                del self.connection_to_player[websocket]
                
                if player_id in self.bot_connections:
                    del self.bot_connections[player_id]
                    logger.info(f"Bot {player_id} disconnected. Total bots: {len(self.bot_connections)}")
                    
                    # Update player status in game if game is active
                    if self.game_state and player_id in self.game_state.players:
                        self.game_state.players[player_id].status = PlayerStatus.DISCONNECTED
                        # Use asyncio.create_task to avoid blocking during iteration
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
            
            # Check if all active players have submitted commands
            active_players = [p.id for p in self.game_state.players.values() 
                            if p.status == PlayerStatus.ACTIVE]
            
            if all(pid in self.turn_commands for pid in active_players):
                await self.process_turn()
                
        except Exception as e:
            logger.error(f"Error handling move command: {e}")
            await self.send_error(websocket, "Error processing command")
    
    async def start_grace_period(self) -> None:
        """
        Start the grace period for additional players to join.
        """
        if self.game_started:
            return
        
        logger.info(f"Starting {self.grace_period_seconds} second grace period for additional players")
        
        await self.broadcast_to_bots({
            "type": "grace_period_started",
            "duration_seconds": self.grace_period_seconds,
            "current_players": len(self.bot_connections)
        })
        
        await asyncio.sleep(self.grace_period_seconds)
        
        if not self.game_started and len(self.bot_connections) >= self.minimum_players:
            await self.start_game()
    
    async def start_game(self) -> None:
        """
        Initialize and start the game.
        """
        try:
            if self.game_started:
                return
            
            self.game_started = True
            logger.info(f"Starting game with {len(self.bot_connections)} players")
            
            # Initialize game state
            self.game_state = GameState(max_turns=self.max_turns)
            self.game_state.graph.generate_grid_graph(self.grid_width, self.grid_height)
            
            # Add players
            for player_id in self.bot_connections.keys():
                self.game_state.add_player(player_id)
            
            # Start the game
            self.game_state.start_game(starting_units=self.starting_units)
            self.game_engine = GameEngine(self.game_state)
            
            # Broadcast initial game state
            await self.broadcast_game_state()
            
            logger.info(f"Game started on turn {self.game_state.current_turn}")
            
        except Exception as e:
            logger.error(f"Error starting game: {e}")
            self.game_started = False
    
    async def process_turn(self) -> None:
        """
        Process all commands for the current turn.
        """
        try:
            if not self.game_engine or not self.game_state:
                return
            
            logger.info(f"Processing turn {self.game_state.current_turn}")
            
            # Process the turn
            turn_result = self.game_engine.process_turn(self.turn_commands)
            
            # Clear commands for next turn
            self.turn_commands.clear()
            
            # Broadcast updated game state
            await self.broadcast_game_state()
            
            # Log turn results
            if turn_result["eliminations"]:
                logger.info(f"Players eliminated: {turn_result['eliminations']}")
            
            if turn_result["game_over"]:
                logger.info(f"Game over! Winners: {turn_result['winner']}")
                await self.broadcast_to_bots({
                    "type": "game_over",
                    "winner": turn_result["winner"],
                    "turn": self.game_state.current_turn
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


async def run_server(host: str = "localhost", port: int = 8765, 
                    grid_width: int = 5, grid_height: int = 5) -> None:
    """
    Run the game server.
    
    Args:
        host: Host address to bind to
        port: Port to listen on
        grid_width: Width of the game grid
        grid_height: Height of the game grid
    """
    server = GameServer(grid_width=grid_width, grid_height=grid_height)
    
    logger.info(f"Starting game server on {host}:{port}")
    logger.info(f"Grid size: {grid_width}x{grid_height}")
    logger.info(f"Waiting for at least {server.minimum_players} bot connections...")
    
    async with websockets.serve(server.handle_client, host, port):
        await asyncio.Future()  # Run forever


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
                            logger.info(f"[{player_id}] Successfully joined game")
                            
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


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run test client
        asyncio.run(test_client())
    elif len(sys.argv) > 1 and sys.argv[1] == "test_game":
        # Run two-player game test
        asyncio.run(test_two_player_game())
    else:
        # Run server
        try:
            asyncio.run(run_server())
        except KeyboardInterrupt:
            logger.info("Server shutting down...")
