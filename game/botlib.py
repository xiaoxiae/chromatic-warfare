"""
Bot Library for Turn-Based Strategy Game

This library provides both high-level and low-level APIs for creating bots
that play the turn-based strategy game. The main goal is to abstract away
all the WebSocket handling, connection management, and game state parsing,
allowing bot developers to focus on strategy.

Usage:
1. Inherit from GameBot
2. Implement the play_turn() method
3. Call bot.run() to start the bot

Example:
    class MyBot(GameBot):
        def play_turn(self, game_state):
            # Simple strategy: attack weakest neighbor
            for vertex in self.my_vertices:
                weak_enemy = self.find_weakest_enemy_neighbor(vertex)
                if weak_enemy and vertex.units > weak_enemy.units + 1:
                    return [self.attack(vertex, weak_enemy, vertex.units - 1)]
            return []
    
    bot = MyBot("my_bot_id")
    bot.run()
"""

import asyncio
import json
import logging
import websockets
import random
import string
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from config import BotConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Vertex:
    """Represents a vertex in the game graph."""
    id: int
    weight: int
    position: Tuple[float, float]
    controller: Optional[str]
    units: int
    
    @property
    def is_mine(self) -> bool:
        """Check if this vertex is controlled by me."""
        return hasattr(self, '_my_player_id') and self.controller == self._my_player_id
    
    @property
    def is_neutral(self) -> bool:
        """Check if this vertex is neutral (uncontrolled)."""
        return self.controller is None
    
    @property
    def is_enemy(self) -> bool:
        """Check if this vertex is controlled by an enemy."""
        return not self.is_mine and not self.is_neutral


@dataclass
class Edge:
    """Represents an edge between vertices."""
    from_vertex: int
    to_vertex: int


@dataclass
class Player:
    """Represents a player in the game."""
    id: str
    status: str
    total_units: int
    
    @property
    def is_active(self) -> bool:
        return self.status == "active"
    
    @property
    def is_eliminated(self) -> bool:
        return self.status == "eliminated"


@dataclass
class Command:
    """Represents a movement command."""
    from_vertex: int
    to_vertex: int
    units: int
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary format for JSON serialization."""
        return {
            "from": self.from_vertex,
            "to": self.to_vertex,
            "units": self.units
        }


class GameState:
    """
    Convenient wrapper around the raw game state that provides
    easy access to game information and relationships.
    """
    
    def __init__(self, raw_state: Dict[str, Any], my_player_id: str):
        self.raw = raw_state
        self.my_player_id = my_player_id
        self.turn = raw_state.get("turn", 0)
        self.game_status = raw_state.get("game_status", "waiting")
        
        # Parse vertices
        vertices_data = raw_state.get("graph", {}).get("vertices", [])
        self.vertices = {}
        for v_data in vertices_data:
            vertex = Vertex(
                id=v_data["id"],
                weight=v_data["weight"],
                position=tuple(v_data["position"]),
                controller=v_data.get("controller"),
                units=v_data["units"]
            )
            vertex._my_player_id = my_player_id  # For is_mine property
            self.vertices[vertex.id] = vertex
        
        # Parse edges and build adjacency map
        edges_data = raw_state.get("graph", {}).get("edges", [])
        self.edges = [Edge(e["from"], e["to"]) for e in edges_data]
        
        self._adjacency = {}
        for edge in self.edges:
            if edge.from_vertex not in self._adjacency:
                self._adjacency[edge.from_vertex] = []
            self._adjacency[edge.from_vertex].append(edge.to_vertex)
        
        # Parse players
        players_data = raw_state.get("players", [])
        self.players = {p["id"]: Player(p["id"], p["status"], p["total_units"]) for p in players_data}
        
        # Convenient access to different types of vertices
        self.my_vertices = [v for v in self.vertices.values() if v.is_mine]
        self.enemy_vertices = [v for v in self.vertices.values() if v.is_enemy]
        self.neutral_vertices = [v for v in self.vertices.values() if v.is_neutral]
    
    def get_vertex(self, vertex_id: int) -> Optional[Vertex]:
        """Get vertex by ID."""
        return self.vertices.get(vertex_id)
    
    def get_neighbors(self, vertex_id: int) -> List[Vertex]:
        """Get all neighboring vertices."""
        neighbor_ids = self._adjacency.get(vertex_id, [])
        return [self.vertices[vid] for vid in neighbor_ids if vid in self.vertices]
    
    def get_enemy_neighbors(self, vertex_id: int) -> List[Vertex]:
        """Get enemy neighboring vertices."""
        return [v for v in self.get_neighbors(vertex_id) if v.is_enemy]
    
    def get_neutral_neighbors(self, vertex_id: int) -> List[Vertex]:
        """Get neutral neighboring vertices."""
        return [v for v in self.get_neighbors(vertex_id) if v.is_neutral]
    
    def get_my_neighbors(self, vertex_id: int) -> List[Vertex]:
        """Get my neighboring vertices."""
        return [v for v in self.get_neighbors(vertex_id) if v.is_mine]
    
    def can_capture(self, from_vertex: int, to_vertex: int, units: int) -> bool:
        """Check if it's possible to capture the given vertex; returns False if own."""
        source = self.get_vertex(from_vertex)
        target = self.get_vertex(to_vertex)
        
        if not source or not target or not source.is_mine:
            return False

        # Check adjacency
        if to_vertex not in self._adjacency.get(from_vertex, []):
            return False
        
        # Calculate cost
        if target.is_enemy:
            return units > target.units
        elif target.is_mine:
            return False
        elif target.is_neutral:
            return units >= target.weight
    
    def calculate_movement_cost(self, from_vertex: int, to_vertex: int) -> int:
        """Calculate the cost of moving units."""
        source = self.get_vertex(from_vertex)
        target = self.get_vertex(to_vertex)
        
        if not source or not target:
            return float('inf')
        
        # No cost between own vertices
        if target.is_mine:
            return 0
        
        # Cost equals vertex weight for neutral vertices
        if target.is_neutral:
            return target.weight
        
        # Cost equals defending units for enemy vertices
        return target.units
    
    @property
    def my_player(self) -> Optional[Player]:
        """Get my player object."""
        return self.players.get(self.my_player_id)
    
    @property
    def is_game_active(self) -> bool:
        """Check if game is currently active."""
        return self.game_status == "active"
    
    @property
    def is_game_ended(self) -> bool:
        """Check if game has ended."""
        return self.game_status == "ended"


class GameBot(ABC):
    """
    Abstract base class for game bots. Handles all WebSocket communication
    and provides a clean interface for bot implementation.
    """
    
    def __init__(self, game_id: str = None, player_id: Optional[str] = None, server_url: str = "ws://localhost:8765",
                 request_bots: Optional[Tuple[int, str]] = None):
        """
        Initialize the bot.
        
        Args:
            player_id: Unique identifier for this bot
            game_id: ID of the game to join (if None, will join "default" game)
            server_url: WebSocket URL of the game server
            request_bots: Tuple of (num_bots, difficulty) to request when connecting
        """
        if not player_id:
            random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
            player_id = str(self.__class__.__name__) + " " + random_string

        self.player_id = player_id
        self.game_id = game_id if game_id is not None else "default"
        self.server_url = server_url
        self.websocket = None
        self.game_state = None
        self.running = False
        self.request_bots_config = request_bots
        
        # Callbacks
        self.on_connection_confirmed: Optional[Callable[[Dict], None]] = None
        self.on_game_started: Optional[Callable[[GameState], None]] = None
        self.on_game_ended: Optional[Callable[[Dict], None]] = None
        self.on_turn_processed: Optional[Callable[[Dict], None]] = None
        
        logger.info(f"Bot {self.player_id} initialized for game {self.game_id}")

    async def request_bots(self, num_bots: int = 1, difficulty: str = "easy") -> bool:
        """
        Request the server to add bots to the current game.
        
        Args:
            num_bots: Number of bots to add
            difficulty: Difficulty level ("easy", "medium", or "hard")
            
        Returns:
            True if request was sent successfully
        """
        if not self.websocket:
            logger.error(f"[{self.player_id}] Cannot request bots: not connected")
            return False
        
        if difficulty not in BotConfig.DIFFICULTIES:
            logger.error(f"[{self.player_id}] Invalid difficulty: {difficulty}")
            return False
        
        message = {
            "type": "request_bots",
            "num_bots": num_bots,
            "difficulty": difficulty
        }
        
        try:
            await self.websocket.send(json.dumps(message))
            logger.info(f"[{self.player_id}] Requested {num_bots} {difficulty} bots")
            return True
        except Exception as e:
            logger.error(f"[{self.player_id}] Failed to request bots: {e}")
            return False
    
    def request_bots_sync(self, num_bots: int = 1, difficulty: str = "easy") -> None:
        """
        Synchronous wrapper for requesting bots.
        Can be called from play_turn or other sync methods.
        """
        if self.websocket:
            asyncio.create_task(self.request_bots(num_bots, difficulty))
    
    @abstractmethod
    def play_turn(self, game_state: GameState) -> List[Command]:
        """
        Main bot logic - implement this method.
        
        Args:
            game_state: Current state of the game
            
        Returns:
            List of commands to execute this turn
        """
        pass
    
    # High-level convenience methods
    
    def move(self, from_vertex: Vertex, to_vertex: Vertex, units: int) -> Command:
        """Move units"""
        return Command(from_vertex.id, to_vertex.id, units)

    def move_all(self, from_vertex: Vertex, to_vertex: Vertex) -> Command:
        """Move all units."""
        return Command(from_vertex.id, to_vertex.id, from_vertex.units)
    
    def get_frontline_vertices(self) -> List[Vertex]:
        """Get vertices that have enemy or neutral neighbors."""
        if not self.game_state:
            return []
        
        frontline = []
        for vertex in self.game_state.my_vertices:
            neighbors = self.game_state.get_neighbors(vertex.id)
            if any(not n.is_mine for n in neighbors):
                frontline.append(vertex)
        
        return frontline
    
    def get_distance_to_frontline(self, vertex_id: int) -> int:
        """
        Calculate the shortest distance from a vertex to the nearest frontline vertex.
        
        A frontline vertex is one that has enemy or neutral neighbors.
        Returns 0 if the vertex itself is on the frontline.
        Returns -1 if the vertex is not mine or doesn't exist.
        
        Args:
            vertex_id: ID of the vertex to check
            
        Returns:
            Number of steps to nearest frontline vertex, or -1 if invalid
        """
        if not self.game_state:
            return -1
        
        vertex = self.game_state.get_vertex(vertex_id)
        if not vertex or not vertex.is_mine:
            return -1
        
        # BFS to find shortest distance to frontline
        from collections import deque
        
        # Get all frontline vertices
        frontline_ids = {v.id for v in self.get_frontline_vertices()}
        
        if not frontline_ids:
            # No frontline exists (all vertices are interior or we have no territory)
            return -1
        
        if vertex_id in frontline_ids:
            # This vertex is already on the frontline
            return 0
        
        # BFS from the given vertex to find closest frontline
        queue = deque([(vertex_id, 0)])
        visited = {vertex_id}
        
        while queue:
            current_id, distance = queue.popleft()
            
            # Check all neighbors
            for neighbor in self.game_state.get_neighbors(current_id):
                if neighbor.id in visited:
                    continue
                
                # Only traverse through my own vertices
                if not neighbor.is_mine:
                    continue
                
                visited.add(neighbor.id)
                
                # If this neighbor is on the frontline, we found our answer
                if neighbor.id in frontline_ids:
                    return distance + 1
                
                # Otherwise, continue searching
                queue.append((neighbor.id, distance + 1))
        
        # No path found (shouldn't happen if graph is connected)
        return -1
    
    def build_command(self, from_id: int, to_id: int, units: int) -> Command:
        """Build a command with raw vertex IDs."""
        return Command(from_id, to_id, units)
    
    def build_commands_batch(self, moves: List[Tuple[int, int, int]]) -> List[Command]:
        """Build multiple commands from a list of (from, to, units) tuples."""
        return [Command(from_id, to_id, units) for from_id, to_id, units in moves]
    
    # Connection and game loop management
    
    async def connect(self) -> bool:
        """Connect to the game server."""
        try:
            logger.info(f"[{self.player_id}] Connecting to {self.server_url}")
            self.websocket = await websockets.connect(self.server_url)
            
            # Join as bot with game ID
            join_message = {
                "type": "join_as_bot",
                "player_id": self.player_id,
                "game_id": self.game_id
            }
            await self.websocket.send(json.dumps(join_message))
            
            logger.info(f"[{self.player_id}] Sent join request")
            return True
            
        except Exception as e:
            logger.error(f"[{self.player_id}] Connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from the server."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            logger.info(f"[{self.player_id}] Disconnected")
    
    async def send_commands(self, commands: List[Command]) -> None:
        """Send movement commands to the server."""
        if not self.websocket:
            return
        
        message = {
            "type": "move_command",
            "commands": [cmd.to_dict() for cmd in commands]
        }
        
        try:
            await self.websocket.send(json.dumps(message))
            logger.debug(f"[{self.player_id}] Sent {len(commands)} commands")
        except Exception as e:
            logger.error(f"[{self.player_id}] Failed to send commands: {e}")
    
    async def handle_message(self, message: Dict[str, Any]) -> None:
        """Handle incoming message from server."""
        message_type = message.get("type")
        
        if message_type == "connection_confirmed":
            game_id = message.get("game_id")
            starting_vertex = message.get("starting_vertex")
            logger.info(f"[{self.player_id}] Connection confirmed for game {game_id}, assigned vertex {starting_vertex}")
            
            # Request bots if configured to do so
            if self.request_bots_config:
                num_bots, difficulty = self.request_bots_config
                await self.request_bots(num_bots, difficulty)
            
            if self.on_connection_confirmed:
                self.on_connection_confirmed(message)
                
        elif message_type == "game_state":
            # Update game state
            self.game_state = GameState(message, self.player_id)
            
            # Check if game just started
            if self.game_state.is_game_active and self.on_game_started:
                self.on_game_started(self.game_state)
            
            # If it's my turn and game is active, play
            if (self.game_state.is_game_active and 
                self.game_state.my_player and 
                self.game_state.my_player.is_active):
                
                try:
                    commands = self.play_turn(self.game_state)
                    if commands is None:
                        commands = []
                    await self.send_commands(commands)
                except Exception as e:
                    logger.error(f"[{self.player_id}] Error in play_turn: {e}")
                    # Send empty commands to not block the game
                    await self.send_commands([])
            
            # Check if game ended
            if self.game_state.is_game_ended and self.on_game_ended:
                self.on_game_ended(message)
                
        elif message_type == "game_over":
            logger.info(f"[{self.player_id}] Game over!")
            rankings = message.get("final_rankings", [])
            logger.info(f"[{self.player_id}] Final rankings: {rankings}")
            if self.on_game_ended:
                self.on_game_ended(message)
            self.running = False
            
        elif message_type == "turn_processed":
            if self.on_turn_processed:
                self.on_turn_processed(message)
                
        elif message_type == "game_reset":
            logger.info(f"[{self.player_id}] Game reset")
            self.game_state = None
            
        elif message_type == "error":
            error_msg = message.get("message", "Unknown error")
            logger.warning(f"[{self.player_id}] Server error: {error_msg}")
            
        else:
            logger.debug(f"[{self.player_id}] Unknown message type: {message_type}")
    
    async def game_loop(self) -> None:
        """Main game loop - handles messages from server."""
        try:
            while self.running:
                try:
                    # Wait for message with timeout
                    message_str = await asyncio.wait_for(
                        self.websocket.recv(), 
                        timeout=60.0
                    )
                    
                    message = json.loads(message_str)
                    await self.handle_message(message)
                    
                except asyncio.TimeoutError:
                    logger.warning(f"[{self.player_id}] Timeout waiting for server message")
                    break
                    
                except websockets.exceptions.ConnectionClosed:
                    logger.info(f"[{self.player_id}] Connection closed by server")
                    break
                    
        except Exception as e:
            logger.error(f"[{self.player_id}] Game loop error: {e}")
        finally:
            self.running = False
    
    def run(self, request_bots: Optional[Tuple[int, str]] = None) -> None:
        """
        Run the bot (blocking call).
        
        Args:
            request_bots: Optional tuple of (num_bots, difficulty) to request AI opponents
        """
        if request_bots:
            self.request_bots_config = request_bots
        asyncio.run(self._run_async())
    
    async def _run_async(self) -> None:
        """Async version of run."""
        self.running = True
        
        try:
            # Connect to server
            if not await self.connect():
                return
            
            # Start game loop
            await self.game_loop()
            
        finally:
            await self.disconnect()
            logger.info(f"[{self.player_id}] Bot terminated")
