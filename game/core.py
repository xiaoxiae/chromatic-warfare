"""
Turn-Based Strategy Game - Core Data Structures

This module contains the foundational data structures for a turn-based strategy game
played on a directed planar graph.
"""

import math
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import random
from collections import defaultdict

from config import MapConfig


class PlayerStatus(Enum):
    """Enumeration for player status states."""
    ACTIVE = "active"
    ELIMINATED = "eliminated"
    DISCONNECTED = "disconnected"


class GameStatus(Enum):
    """Enumeration for game status states."""
    WAITING = "waiting"
    ACTIVE = "active"
    ENDED = "ended"


@dataclass
class Command:
    """
    Represents a movement command from a player.
    
    Attributes:
        player_id: ID of the player issuing the command
        from_vertex: Source vertex ID
        to_vertex: Destination vertex ID
        units: Number of units to move
    """
    player_id: str
    from_vertex: int
    to_vertex: int
    units: int
    
    def __post_init__(self):
        """Validate command data after initialization."""
        if self.units <= 0:
            raise ValueError("Unit count must be positive")
        if self.from_vertex == self.to_vertex:
            raise ValueError("Cannot move units to the same vertex")


@dataclass
class Vertex:
    """
    Represents a vertex in the game graph.
    
    Attributes:
        id: Unique identifier for the vertex
        weight: Weight of the vertex (affects unit generation)
        position: (x, y) coordinates for visualization
        controller: Player ID who controls this vertex, or None if uncontrolled
        units: Number of units currently on this vertex
    """
    id: int
    weight: int
    position: Tuple[float, float]
    controller: Optional[str] = None
    units: int = 0
    
    def __post_init__(self):
        """Validate vertex data after initialization."""
        if self.weight <= 0:
            raise ValueError("Vertex weight must be positive")
        if self.units < 0:
            raise ValueError("Unit count cannot be negative")


@dataclass(frozen=True)
class Edge:
    """
    Represents a directed edge between two vertices.
    
    Attributes:
        from_vertex: Source vertex ID
        to_vertex: Destination vertex ID
    """
    from_vertex: int
    to_vertex: int
    
    def __post_init__(self):
        """Validate edge data after initialization."""
        if self.from_vertex == self.to_vertex:
            raise ValueError("Self-loops are not allowed")


class Graph:
    """
    Manages the game graph consisting of vertices and directed edges.
    """
    
    def __init__(self):
        """Initialize an empty graph."""
        self.vertices: Dict[int, Vertex] = {}
        self.edges: Set[Edge] = set()
        # Adjacency list for efficient neighbor lookup
        self._adjacency: Dict[int, Set[int]] = {}
    
    def add_vertex(self, vertex: Vertex) -> None:
        """
        Add a vertex to the graph.
        
        Args:
            vertex: Vertex object to add
            
        Raises:
            ValueError: If vertex ID already exists
        """
        if vertex.id in self.vertices:
            raise ValueError(f"Vertex with ID {vertex.id} already exists")
        
        self.vertices[vertex.id] = vertex
        self._adjacency[vertex.id] = set()
    
    def remove_vertex(self, vertex_id: int) -> None:
        """
        Remove a vertex and all its edges from the graph.
        
        Args:
            vertex_id: ID of vertex to remove
            
        Raises:
            ValueError: If vertex ID doesn't exist
        """
        if vertex_id not in self.vertices:
            raise ValueError(f"Vertex with ID {vertex_id} doesn't exist")
        
        # Remove all edges involving this vertex
        edges_to_remove = [edge for edge in self.edges 
                          if edge.from_vertex == vertex_id or edge.to_vertex == vertex_id]
        for edge in edges_to_remove:
            self.remove_edge(edge.from_vertex, edge.to_vertex)
        
        del self.vertices[vertex_id]
        del self._adjacency[vertex_id]
    
    def add_edge(self, from_vertex: int, to_vertex: int) -> None:
        """
        Add a directed edge between two vertices.
        
        Args:
            from_vertex: Source vertex ID
            to_vertex: Destination vertex ID
            
        Raises:
            ValueError: If either vertex doesn't exist or edge already exists
        """
        if from_vertex not in self.vertices:
            raise ValueError(f"Source vertex {from_vertex} doesn't exist")
        if to_vertex not in self.vertices:
            raise ValueError(f"Destination vertex {to_vertex} doesn't exist")
        
        edge = Edge(from_vertex, to_vertex)
        if edge in self.edges:
            raise ValueError(f"Edge from {from_vertex} to {to_vertex} already exists")
        
        self.edges.add(edge)
        self._adjacency[from_vertex].add(to_vertex)
    
    def remove_edge(self, from_vertex: int, to_vertex: int) -> None:
        """
        Remove a directed edge from the graph.
        
        Args:
            from_vertex: Source vertex ID
            to_vertex: Destination vertex ID
            
        Raises:
            ValueError: If edge doesn't exist
        """
        edge = Edge(from_vertex, to_vertex)
        if edge not in self.edges:
            raise ValueError(f"Edge from {from_vertex} to {to_vertex} doesn't exist")
        
        self.edges.remove(edge)
        self._adjacency[from_vertex].discard(to_vertex)
    
    def get_adjacent_vertices(self, vertex_id: int) -> Set[int]:
        """
        Get all vertices adjacent to the given vertex.
        
        Args:
            vertex_id: ID of the vertex to get neighbors for
            
        Returns:
            Set of adjacent vertex IDs
            
        Raises:
            ValueError: If vertex doesn't exist
        """
        if vertex_id not in self.vertices:
            raise ValueError(f"Vertex {vertex_id} doesn't exist")
        
        return self._adjacency[vertex_id].copy()
    
    def get_vertices_controlled_by_player(self, player_id: str) -> List[Vertex]:
        """
        Get all vertices controlled by a specific player.
        
        Args:
            player_id: ID of the player
            
        Returns:
            List of vertices controlled by the player
        """
        return [vertex for vertex in self.vertices.values() 
                if vertex.controller == player_id]
    
    def get_uncontrolled_vertices(self) -> List[Vertex]:
        """
        Get all vertices that are not controlled by any player.
        
        Returns:
            List of uncontrolled vertices
        """
        return [vertex for vertex in self.vertices.values() 
                if vertex.controller is None]
    
    def generate_grid_graph(self, width: int, height: int, vertex_weight_range: Optional[Tuple[int, int]] = None, vertex_remove_probability: Optional[float] = None) -> None:
        """
        Generate a grid graph with bidirectional connections between adjacent cells.
        
        Args:
            width: Width of the grid
            height: Height of the grid
            vertex_weight_range: Optional tuple of (min_weight, max_weight) to sample vertex weights from.
                               If None, all vertices will have weight 1.
            vertex_remove_probability: Optional float between 0 and 1 representing the probability
                                     of attempting to remove each vertex. Vertices are only removed
                                     if the graph remains connected after removal.
            
        Raises:
            ValueError: If dimensions are not positive, weight range is invalid, or 
                       remove probability is not between 0 and 1
        """
        if not (MapConfig.MIN_WIDTH <= width <= MapConfig.MAX_WIDTH):
            raise ValueError(f"Grid width must be between {MapConfig.MIN_WIDTH} and {MapConfig.MAX_WIDTH}")
        if not (MapConfig.MIN_HEIGHT <= height <= MapConfig.MAX_HEIGHT):
            raise ValueError(f"Grid height must be between {MapConfig.MIN_HEIGHT} and {MapConfig.MAX_HEIGHT}")
        
        if vertex_weight_range is not None:
            min_weight, max_weight = vertex_weight_range
            if not (MapConfig.MIN_VERTEX_WEIGHT <= min_weight <= MapConfig.MAX_VERTEX_WEIGHT):
                raise ValueError(f"Minimum weight must be between {MapConfig.MIN_VERTEX_WEIGHT} and {MapConfig.MAX_VERTEX_WEIGHT}")
            if not (MapConfig.MIN_VERTEX_WEIGHT <= max_weight <= MapConfig.MAX_VERTEX_WEIGHT):
                raise ValueError(f"Maximum weight must be between {MapConfig.MIN_VERTEX_WEIGHT} and {MapConfig.MAX_VERTEX_WEIGHT}")
        
        if vertex_remove_probability is not None:
            if not (MapConfig.MIN_REMOVE_PROBABILITY <= vertex_remove_probability <= MapConfig.MAX_REMOVE_PROBABILITY):
                raise ValueError(f"Vertex remove probability must be between {MapConfig.MIN_REMOVE_PROBABILITY} and {MapConfig.MAX_REMOVE_PROBABILITY}")
        
        # Clear existing graph
        self.vertices.clear()
        self.edges.clear()
        self._adjacency.clear()
        
        # Create vertices
        for y in range(height):
            for x in range(width):
                vertex_id = y * width + x
                position = (float(x), float(y))
                
                # Determine vertex weight
                if vertex_weight_range is not None:
                    min_weight, max_weight = vertex_weight_range
                    weight = random.randint(min_weight, max_weight)
                else:
                    weight = 1
                
                vertex = Vertex(vertex_id, weight, position)
                self.add_vertex(vertex)
        
        # Create edges (bidirectional connections between adjacent cells)
        for y in range(height):
            for x in range(width):
                current_id = y * width + x
                
                # Connect to right neighbor
                if x < width - 1:
                    right_id = y * width + (x + 1)
                    self.add_edge(current_id, right_id)
                    self.add_edge(right_id, current_id)
                
                # Connect to bottom neighbor
                if y < height - 1:
                    bottom_id = (y + 1) * width + x
                    self.add_edge(current_id, bottom_id)
                    self.add_edge(bottom_id, current_id)
        
        # Remove vertices randomly while maintaining connectivity
        if vertex_remove_probability is not None:
            self._remove_vertices_randomly(vertex_remove_probability)
    
    def _remove_vertices_randomly(self, remove_probability: float) -> None:
        """
        Remove vertices randomly while maintaining graph connectivity.
        
        Args:
            remove_probability: Probability of attempting to remove each vertex
        """
        # Get all vertex IDs and shuffle them for random removal order
        vertex_ids = list(self.vertices.keys())
        random.shuffle(vertex_ids)
        
        for vertex_id in vertex_ids:
            # Skip if vertex was already removed
            if vertex_id not in self.vertices:
                continue
                
            # Check if we should attempt to remove this vertex
            if random.random() < remove_probability:
                # Check if removing this vertex would disconnect the graph
                if self._can_remove_vertex_safely(vertex_id):
                    self.remove_vertex(vertex_id)
    
    def _can_remove_vertex_safely(self, vertex_id: int) -> bool:
        """
        Check if removing a vertex would keep the graph connected.
        
        Args:
            vertex_id: ID of vertex to check for removal
            
        Returns:
            True if the vertex can be removed without disconnecting the graph
        """
        # If this is the only vertex, we can't remove it
        if len(self.vertices) <= 1:
            return False
        
        # Get neighbors of the vertex to be removed
        neighbors = self.get_adjacent_vertices(vertex_id)
        
        # If the vertex has no neighbors, it's safe to remove (isolated vertex)
        if not neighbors:
            return True
        
        # Temporarily remove the vertex and check if remaining graph is connected
        # Store the vertex and its edges for restoration
        vertex = self.vertices[vertex_id]
        edges_to_restore = []
        
        # Find all edges involving this vertex
        for edge in list(self.edges):
            if edge.from_vertex == vertex_id or edge.to_vertex == vertex_id:
                edges_to_restore.append(edge)
        
        # Temporarily remove the vertex
        self.remove_vertex(vertex_id)
        
        # Check if remaining graph is connected
        is_connected = self._is_graph_connected()
        
        # Restore the vertex and its edges
        self.add_vertex(vertex)
        for edge in edges_to_restore:
            self.add_edge(edge.from_vertex, edge.to_vertex)
        
        return is_connected
    
    def _is_graph_connected(self) -> bool:
        """
        Check if the graph is connected using BFS.
        
        Returns:
            True if the graph is connected, False otherwise
        """
        if not self.vertices:
            return True  # Empty graph is considered connected
        
        # Start BFS from any vertex
        start_vertex = next(iter(self.vertices.keys()))
        visited = set()
        queue = [start_vertex]
        visited.add(start_vertex)
        
        while queue:
            current = queue.pop(0)
            for neighbor in self._adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # Graph is connected if all vertices were visited
        return len(visited) == len(self.vertices)

    def generate_hex_graph(self, width: int, height: int, vertex_weight_range: Optional[Tuple[int, int]] = None, vertex_remove_probability: Optional[float] = None) -> None:
        """
        Generate a hexagonal grid graph where each vertex connects to 6 neighbors.
        
        Args:
            width: Width of the hexagonal grid (number of columns)
            height: Height of the hexagonal grid (number of rows)
            vertex_weight_range: Optional tuple of (min_weight, max_weight) to sample vertex weights from.
                               If None, all vertices will have weight 1.
            vertex_remove_probability: Optional float between 0 and 1 representing the probability
                                     of attempting to remove each vertex. Vertices are only removed
                                     if the graph remains connected after removal.
            
        Raises:
            ValueError: If dimensions are not positive, weight range is invalid, or 
                       remove probability is not between 0 and 1
        """
        if not (MapConfig.MIN_WIDTH <= width <= MapConfig.MAX_WIDTH):
            raise ValueError(f"Grid width must be between {MapConfig.MIN_WIDTH} and {MapConfig.MAX_WIDTH}")
        if not (MapConfig.MIN_HEIGHT <= height <= MapConfig.MAX_HEIGHT):
            raise ValueError(f"Grid height must be between {MapConfig.MIN_HEIGHT} and {MapConfig.MAX_HEIGHT}")
        
        if vertex_weight_range is not None:
            min_weight, max_weight = vertex_weight_range
            if not (MapConfig.MIN_VERTEX_WEIGHT <= min_weight <= MapConfig.MAX_VERTEX_WEIGHT):
                raise ValueError(f"Minimum weight must be between {MapConfig.MIN_VERTEX_WEIGHT} and {MapConfig.MAX_VERTEX_WEIGHT}")
            if not (MapConfig.MIN_VERTEX_WEIGHT <= max_weight <= MapConfig.MAX_VERTEX_WEIGHT):
                raise ValueError(f"Maximum weight must be between {MapConfig.MIN_VERTEX_WEIGHT} and {MapConfig.MAX_VERTEX_WEIGHT}")
        
        if vertex_remove_probability is not None:
            if not (MapConfig.MIN_REMOVE_PROBABILITY <= vertex_remove_probability <= MapConfig.MAX_REMOVE_PROBABILITY):
                raise ValueError(f"Vertex remove probability must be between {MapConfig.MIN_REMOVE_PROBABILITY} and {MapConfig.MAX_REMOVE_PROBABILITY}")
        
        # Clear existing graph
        self.vertices.clear()
        self.edges.clear()
        self._adjacency.clear()
        
        # Create vertices with hexagonal positioning
        for row in range(height):
            for col in range(width):
                vertex_id = row * width + col
                
                # Calculate hexagonal grid position
                # Odd rows are offset by 0.5 in x direction
                x = col + (0.5 if row % 2 == 1 else 0.0)
                y = row * (math.sqrt(3) / 2)  # Vertical spacing for hexagons
                position = (x, y)
                
                # Determine vertex weight
                if vertex_weight_range is not None:
                    min_weight, max_weight = vertex_weight_range
                    weight = random.randint(min_weight, max_weight)
                else:
                    weight = 1
                
                vertex = Vertex(vertex_id, weight, position)
                self.add_vertex(vertex)
        
        # Create hexagonal connections (each vertex connects to 6 neighbors)
        for row in range(height):
            for col in range(width):
                current_id = row * width + col
                
                # Get all 6 potential neighbors for hexagonal grid
                neighbors = self._get_hex_neighbors(col, row, width, height)
                
                # Add edges to all valid neighbors
                for neighbor_col, neighbor_row in neighbors:
                    neighbor_id = neighbor_row * width + neighbor_col
                    # Add bidirectional edge if it doesn't already exist
                    if current_id < neighbor_id:  # Avoid duplicate edges
                        self.add_edge(current_id, neighbor_id)
                        self.add_edge(neighbor_id, current_id)
        
        # Remove vertices randomly while maintaining connectivity
        if vertex_remove_probability is not None:
            self._remove_vertices_randomly(vertex_remove_probability)

    def _get_hex_neighbors(self, col: int, row: int, width: int, height: int) -> list:
        """
        Get the valid hexagonal neighbors for a given position.
        
        Args:
            col: Column of the current hex
            row: Row of the current hex
            width: Width of the grid
            height: Height of the grid
            
        Returns:
            List of (col, row) tuples representing valid neighbor positions
        """
        neighbors = []
        
        if row % 2 == 0:  # Even row
            # Potential neighbors for even rows
            potential = [
                (col - 1, row - 1),  # Northwest
                (col, row - 1),      # Northeast
                (col - 1, row),      # West
                (col + 1, row),      # East
                (col - 1, row + 1),  # Southwest
                (col, row + 1)       # Southeast
            ]
        else:  # Odd row
            # Potential neighbors for odd rows (offset)
            potential = [
                (col, row - 1),      # Northwest
                (col + 1, row - 1),  # Northeast
                (col - 1, row),      # West
                (col + 1, row),      # East
                (col, row + 1),      # Southwest
                (col + 1, row + 1)   # Southeast
            ]
        
        # Filter out invalid positions (outside grid boundaries)
        for neighbor_col, neighbor_row in potential:
            if (0 <= neighbor_col < width and 0 <= neighbor_row < height):
                neighbors.append((neighbor_col, neighbor_row))
        
        return neighbors


class Player:
    """
    Represents a player in the game.
    
    Attributes:
        id: Unique identifier for the player
        status: Current status of the player
        total_units: Total number of units the player has across all vertices
    """
    
    def __init__(self, player_id: str, status: PlayerStatus = PlayerStatus.ACTIVE):
        """
        Initialize a new player.
        
        Args:
            player_id: Unique identifier for the player
            status: Initial status of the player
        """
        self.id = player_id
        self.status = status
        self.total_units = 0
    
    def update_total_units(self, graph: Graph) -> None:
        """
        Update the player's total unit count based on controlled vertices.
        
        Args:
            graph: Game graph to calculate units from
        """
        controlled_vertices = graph.get_vertices_controlled_by_player(self.id)
        self.total_units = sum(vertex.units for vertex in controlled_vertices)
    
    def to_dict(self) -> Dict:
        """
        Convert player to dictionary format for JSON serialization.
        
        Returns:
            Dictionary representation of the player
        """
        return {
            "id": self.id,
            "status": self.status.value,
            "total_units": self.total_units
        }


class GameState:
    """
    Represents the complete state of the game.
    
    Attributes:
        graph: The game graph
        players: Dictionary of players by ID
        current_turn: Current turn number
        status: Current game status
        elimination_order: List of (turn, [player_ids]) tracking elimination order
        final_rankings: Final rankings of all players
        max_turns: Maximum number of turns before game ends
    """
    
    def __init__(self, max_turns: int = 100):
        """
        Initialize a new game state.
        
        Args:
            max_turns: Maximum number of turns before game ends
        """
        self.graph = Graph()
        self.players: Dict[str, Player] = {}
        self.current_turn = 0
        self.status = GameStatus.WAITING
        self.elimination_order: List[Tuple[int, List[str]]] = []  # (turn, [eliminated_players])
        self.final_rankings: List[List[str]] = []  # List of lists, each inner list is a tie group
        self.max_turns = max_turns
    
    def add_player(self, player_id: str) -> None:
        """
        Add a player to the game.
        
        Args:
            player_id: Unique identifier for the player
            
        Raises:
            ValueError: If player already exists or game has started
        """
        if self.status != GameStatus.WAITING:
            raise ValueError("Cannot add players after game has started")
        if player_id in self.players:
            raise ValueError(f"Player {player_id} already exists")
        
        self.players[player_id] = Player(player_id)
    
    def start_game(self, starting_units: int = 1) -> None:
        """
        Start the game by assigning random starting positions to players.
        
        Args:
            starting_units: Number of units each player starts with
            
        Raises:
            ValueError: If less than 2 players or game already started
        """
        if len(self.players) < 2:
            raise ValueError("Need at least 2 players to start game")
        if self.status != GameStatus.WAITING:
            raise ValueError("Game has already started")
        
        # Assign random starting positions
        uncontrolled_vertices = self.graph.get_uncontrolled_vertices()
        if len(uncontrolled_vertices) < len(self.players):
            raise ValueError("Not enough vertices for all players")
        
        starting_vertices = random.sample(uncontrolled_vertices, len(self.players))
        
        for player, vertex in zip(self.players.values(), starting_vertices):
            vertex.controller = player.id
            vertex.units = starting_units
            player.update_total_units(self.graph)
        
        self.status = GameStatus.ACTIVE
        self.current_turn = 1
    
    def update_player_totals(self) -> None:
        """Update total unit counts for all players."""
        for player in self.players.values():
            player.update_total_units(self.graph)
    
    def check_game_end(self) -> bool:
        """
        Check if the game should end and update game status accordingly.
        
        Returns:
            True if game has ended, False otherwise
        """
        active_players = [p for p in self.players.values() 
                         if p.status == PlayerStatus.ACTIVE and p.total_units > 0]
        
        # Game ends if only one or no active players remain
        if len(active_players) <= 1:
            self.status = GameStatus.ENDED
            self._calculate_final_rankings()
            return True
        
        # Game ends if max turns reached
        if self.current_turn >= self.max_turns:
            self.status = GameStatus.ENDED
            self._calculate_final_rankings()
            return True
        
        return False
    
    def _calculate_final_rankings(self) -> None:
        """
        Calculate final rankings based on elimination order and final scores.
        """
        self.final_rankings = []
        
        # Start with players still active (winners)
        active_players = [p for p in self.players.values() 
                         if p.status == PlayerStatus.ACTIVE and p.total_units > 0]
        
        if active_players:
            if len(active_players) == 1:
                # Single winner
                self.final_rankings.append([active_players[0].id])
            else:
                # Multiple survivors - rank by unit count
                active_players.sort(key=lambda p: p.total_units, reverse=True)
                
                # Group players with same unit count
                current_group = []
                current_units = None
                
                for player in active_players:
                    if current_units is None or player.total_units == current_units:
                        current_group.append(player.id)
                        current_units = player.total_units
                    else:
                        # Start new group
                        self.final_rankings.append(current_group)
                        current_group = [player.id]
                        current_units = player.total_units
                
                # Add final group
                if current_group:
                    self.final_rankings.append(current_group)
        
        # Add eliminated players in reverse order (last eliminated = higher rank)
        for turn, eliminated_players in reversed(self.elimination_order):
            if eliminated_players:
                self.final_rankings.append(eliminated_players)
    
    def to_dict(self) -> Dict:
        """
        Convert game state to dictionary format for JSON serialization.
        
        Returns:
            Dictionary representation of the game state
        """
        return {
            "type": "game_state",
            "turn": self.current_turn,
            "graph": {
                "vertices": [
                    {
                        "id": v.id,
                        "weight": v.weight,
                        "position": list(v.position),
                        "controller": v.controller,
                        "units": v.units
                    }
                    for v in self.graph.vertices.values()
                ],
                "edges": [
                    {"from": e.from_vertex, "to": e.to_vertex}
                    for e in self.graph.edges
                ]
            },
            "players": [p.to_dict() for p in self.players.values()],
            "game_status": self.status.value,
            "elimination_order": self.elimination_order,
            "final_rankings": self.final_rankings,
        }

    def get_starting_vertex(self) -> Optional[Vertex]:
        """
        Select a fair starting vertex for a new player.
        
        Args:
            graph: The game graph containing vertices and edges
            
        Returns:
            The ID of the selected starting vertex, or None if no vertices exist
            
        Logic:
            - If no players exist (no controlled vertices), pick the vertex that maximizes
              the sum of squared distances to all other vertices
            - If players exist, pick the vertex that maximizes the sum of squared
              distances to all controlled vertices
        """
        if not self.graph.vertices:
            return None
        
        # Get all controlled vertices (vertices with players)
        controlled_vertices = [v for v in self.graph.vertices.values() if v.controller is not None]
        
        # Get all uncontrolled vertices (potential starting positions)
        uncontrolled_vertices = [v for v in self.graph.vertices.values() if v.controller is None]
        
        if not uncontrolled_vertices:
            return None  # No available starting positions
        
        best_vertex_id = None
        best_score = float('-inf')
        
        for candidate in uncontrolled_vertices:
            if not controlled_vertices:
                # No players yet - maximize distance to all other vertices
                score = self._calculate_distance_score_to_all(candidate, self.graph.vertices)
            else:
                # Players exist - maximize distance to controlled vertices
                score = self._calculate_distance_score_to_controlled(candidate, controlled_vertices)
            
            if score > best_score:
                best_score = score
                best_vertex_id = candidate.id
        
        return self.graph.vertices[best_vertex_id]


    def _calculate_distance_score_to_all(self, candidate: Vertex, all_vertices: Dict[int, Vertex]) -> float:
        """
        Calculate the sum of squared distances from candidate to all other vertices.
        
        Args:
            candidate: The vertex to calculate distances from
            all_vertices: Dictionary of all vertices in the graph
            
        Returns:
            Sum of squared Euclidean distances to all other vertices
        """
        total_score = 0.0
        candidate_pos = candidate.position
        
        for vertex in all_vertices.values():
            if vertex.id != candidate.id:
                distance_squared = self._squared_euclidean_distance(candidate_pos, vertex.position)
                total_score += distance_squared
        
        return total_score


    def _calculate_distance_score_to_controlled(self, candidate: Vertex, controlled_vertices: List[Vertex]) -> float:
        """
        Calculate the sum of squared distances from candidate to all controlled vertices.
        
        Args:
            candidate: The vertex to calculate distances from
            controlled_vertices: List of vertices controlled by existing players
            
        Returns:
            Sum of squared Euclidean distances to all controlled vertices
        """
        total_score = 0.0
        candidate_pos = candidate.position
        
        for controlled_vertex in controlled_vertices:
            distance_squared = self._squared_euclidean_distance(candidate_pos, controlled_vertex.position)
            total_score += distance_squared
        
        return total_score


    def _squared_euclidean_distance(self, pos1: tuple, pos2: tuple) -> float:
        """
        Calculate the squared Euclidean distance between two positions.
        
        Args:
            pos1: (x, y) coordinates of first position
            pos2: (x, y) coordinates of second position
            
        Returns:
            Squared Euclidean distance
        """
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return dx * dx + dy * dy


class GameEngine:
    """
    Handles core game mechanics including movement, combat, and turn resolution.
    """
    
    def __init__(self, game_state: GameState):
        """
        Initialize the game engine with a game state.
        
        Args:
            game_state: The game state to operate on
        """
        self.game_state = game_state
    
    def process_movement_command(self, player_id: str, from_vertex: int, 
                               to_vertex: int, units: int) -> bool:
        """
        Validate a single movement command.
        
        Args:
            player_id: ID of the player issuing the command
            from_vertex: Source vertex ID
            to_vertex: Destination vertex ID
            units: Number of units to move
            
        Returns:
            True if command is valid, False otherwise
        """
        try:
            # Basic validation
            if player_id not in self.game_state.players:
                return False
            
            if self.game_state.players[player_id].status != PlayerStatus.ACTIVE:
                return False
            
            # Check if vertices exist
            if (from_vertex not in self.game_state.graph.vertices or 
                to_vertex not in self.game_state.graph.vertices):
                return False
            
            # Check adjacency
            adjacent = self.game_state.graph.get_adjacent_vertices(from_vertex)
            if to_vertex not in adjacent:
                return False
            
            # Check if player controls source vertex
            source = self.game_state.graph.vertices[from_vertex]
            if source.controller != player_id:
                return False
            
            # Check if player has enough units (no movement costs now)
            if source.units < units:
                return False
            
            return True
            
        except Exception:
            return False
    
    def resolve_all_movements(self, movements_dict: Dict[str, List[Command]]) -> None:
        """
        Resolve all movement commands for the current turn.
        
        Args:
            movements_dict: Dictionary mapping player IDs to their list of commands
        """
        # Step 1: Validate all commands and track unit requirements per vertex
        valid_movements = []
        unit_requirements = defaultdict(int)
        
        for player_id, commands in movements_dict.items():
            for command in commands:
                if self.process_movement_command(command.player_id, command.from_vertex, 
                                               command.to_vertex, command.units):
                    # Check if source vertex can afford this command along with others
                    source_vertex = self.game_state.graph.vertices[command.from_vertex]
                    if unit_requirements[command.from_vertex] + command.units <= source_vertex.units:
                        unit_requirements[command.from_vertex] += command.units
                        valid_movements.append(command)
        
        # Step 2: Deduct units from source vertices
        for command in valid_movements:
            source = self.game_state.graph.vertices[command.from_vertex]
            source.units -= command.units
        
        # Step 3: Process mutual cancellations
        movements_after_cancellation = self._process_mutual_cancellations(valid_movements)
        
        # Step 4: Group remaining movements by destination and resolve
        movements_by_destination = defaultdict(list)
        for command in movements_after_cancellation:
            movements_by_destination[command.to_vertex].append(command)
        
        for dest_vertex_id, movements in movements_by_destination.items():
            self._resolve_vertex_conflict(dest_vertex_id, movements)
    
    def _process_mutual_cancellations(self, movements: List[Command]) -> List[Command]:
        """
        Process mutual cancellations between opposing movements.
        
        Args:
            movements: List of all valid movement commands
            
        Returns:
            List of movements after cancellations
        """
        # Create a mapping of (from, to) -> total units
        movement_map = defaultdict(int)
        movement_players = {}  # Track which player sent units for each route
        
        for command in movements:
            route = (command.from_vertex, command.to_vertex)
            movement_map[route] += command.units
            movement_players[route] = command.player_id
        
        # Process cancellations for opposing routes
        remaining_movements = []
        processed_routes = set()
        
        for (from_v, to_v), units in movement_map.items():
            reverse_route = (to_v, from_v)
            
            if reverse_route in movement_map and (from_v, to_v) not in processed_routes:
                # Mutual cancellation
                reverse_units = movement_map[reverse_route]
                
                if units > reverse_units:
                    # Forward movement survives
                    remaining_units = units - reverse_units
                    remaining_movements.append(Command(
                        movement_players[(from_v, to_v)], from_v, to_v, remaining_units
                    ))
                elif reverse_units > units:
                    # Reverse movement survives
                    remaining_units = reverse_units - units
                    remaining_movements.append(Command(
                        movement_players[reverse_route], to_v, from_v, remaining_units
                    ))
                # If equal, both cancel out completely (no remaining movement)
                
                # Mark both routes as processed
                processed_routes.add((from_v, to_v))
                processed_routes.add(reverse_route)
            
            elif (from_v, to_v) not in processed_routes:
                # No opposing movement, keep as is
                remaining_movements.append(Command(
                    movement_players[(from_v, to_v)], from_v, to_v, units
                ))
                processed_routes.add((from_v, to_v))
        
        return remaining_movements
    
    def _resolve_vertex_conflict(self, vertex_id: int, movements: List[Command]) -> None:
        """
        Resolve conflicts at a single vertex.
        
        Args:
            vertex_id: ID of the destination vertex
            movements: List of movement commands targeting this vertex
        """
        if not movements:
            return
        
        vertex = self.game_state.graph.vertices[vertex_id]
        
        # Group movements by attacking player
        attacks_by_player = defaultdict(int)
        for movement in movements:
            attacks_by_player[movement.player_id] += movement.units
        
        # Handle different vertex states
        if vertex.controller is None:
            # Neutral vertex - require weight cost, then probabilistic resolution
            self._resolve_neutral_vertex_conflict(vertex, attacks_by_player)
        else:
            # Controlled vertex - check if any attacker is the controller
            controller_reinforcement = attacks_by_player.pop(vertex.controller, 0)
            vertex.units += controller_reinforcement  # Add friendly reinforcements
            
            if attacks_by_player:
                # There are actual attacks
                self._resolve_controlled_vertex_conflict(vertex, attacks_by_player)
    
    def _resolve_neutral_vertex_conflict(self, vertex: Vertex, 
                                       attacks_by_player: Dict[str, int]) -> None:
        """
        Resolve conflict at a neutral vertex.
        
        Args:
            vertex: The neutral vertex being attacked
            attacks_by_player: Dictionary mapping player IDs to total attacking units
        """
        if not attacks_by_player:
            return
        
        total_attacking_units = sum(attacks_by_player.values())
        
        # Pay the weight cost
        if total_attacking_units < vertex.weight:
            return
        
        effective_units = total_attacking_units - vertex.weight
    
        if effective_units < 0:
            return
        
        # Calculate probabilities based on proportion of attacking units
        probabilities = {player_id: units / total_attacking_units 
                        for player_id, units in attacks_by_player.items()}
        
        # Randomly select winner
        rand = random.random()
        cumulative_prob = 0
        winner = None
        
        for player_id, prob in probabilities.items():
            cumulative_prob += prob
            if rand <= cumulative_prob:
                winner = player_id
                break
        
        if winner:
            vertex.controller = winner
            vertex.units = effective_units
    
    def _resolve_controlled_vertex_conflict(self, vertex: Vertex, 
                                          attacks_by_player: Dict[str, int]) -> None:
        """
        Resolve conflict at a controlled vertex.
        
        Args:
            vertex: The controlled vertex being attacked
            attacks_by_player: Dictionary mapping player IDs to total attacking units
        """
        if not attacks_by_player:
            return
        
        total_attacking_units = sum(attacks_by_player.values())
        defending_units = vertex.units
        
        if total_attacking_units < defending_units:
            # Defenders win, reduce defender count
            vertex.units = defending_units - total_attacking_units
        elif total_attacking_units == defending_units:
            # Tie - vertex becomes neutral
            vertex.controller = None
            vertex.units = 0
        else:
            # Attackers win
            remaining_attackers = total_attacking_units - defending_units
            
            if len(attacks_by_player) == 1:
                # Single attacker takes control
                winner = next(iter(attacks_by_player))
                vertex.controller = winner
                vertex.units = remaining_attackers
            else:
                # Multiple attackers - probabilistic resolution
                probabilities = {player_id: units / total_attacking_units 
                               for player_id, units in attacks_by_player.items()}
                
                rand = random.random()
                cumulative_prob = 0
                winner = None
                
                for player_id, prob in probabilities.items():
                    cumulative_prob += prob
                    if rand <= cumulative_prob:
                        winner = player_id
                        break
                
                if winner:
                    vertex.controller = winner
                    vertex.units = remaining_attackers
    
    def generate_units(self) -> None:
        """
        Generate new units at all controlled vertices based on their weight.
        """
        for vertex in self.game_state.graph.vertices.values():
            if vertex.controller is not None:
                vertex.units += vertex.weight
    
    def check_eliminations(self) -> List[str]:
        """
        Check for player eliminations and update their status.
        
        Returns:
            List of player IDs that were eliminated this turn
        """
        eliminated_players = []
        
        for player in self.game_state.players.values():
            if player.status == PlayerStatus.ACTIVE:
                # Update total units
                player.update_total_units(self.game_state.graph)
                
                # Check if player should be eliminated
                controlled_vertices = self.game_state.graph.get_vertices_controlled_by_player(player.id)
                if len(controlled_vertices) == 0 or player.total_units == 0:
                    eliminated_players.append(player.id)
        
        # Record elimination order if any players were eliminated
        if eliminated_players:
            self.game_state.elimination_order.append((self.game_state.current_turn, eliminated_players))
        
        return eliminated_players
    
    def is_game_over(self) -> bool:
        """
        Check if the game is over and update game status accordingly.
        
        Returns:
            True if the game is over, False otherwise
        """
        return self.game_state.check_game_end()
    
    def process_turn(self, movements_dict: Dict[str, List[Command]]) -> Dict:
        """
        Process a complete game turn including movement, combat, and unit generation.
        
        Args:
            movements_dict: Dictionary mapping player IDs to their movement commands
            
        Returns:
            Dictionary with turn results including eliminations and game status
        """
        turn_results = {
            "turn": self.game_state.current_turn,
            "eliminations": [],
            "game_over": False,
        }
        
        # Phase 1: Resolve all movements and combat
        self.resolve_all_movements(movements_dict)
        
        # Phase 2: Generate new units
        self.generate_units()
        
        # Phase 3: Check eliminations
        turn_results["eliminations"] = self.check_eliminations()
        
        # Phase 4: Check if game is over
        if self.is_game_over():
            turn_results["game_over"] = True
        else:
            # Advance to next turn
            self.game_state.current_turn += 1
        
        return turn_results


def test_new_combat_system():
    """
    Test the new combat system with mutual cancellations and simplified costs.
    """
    print("Testing new combat system...")
    
    # Test 1: Mutual cancellation
    print("\n=== Test 1: Mutual Cancellation ===")
    game = GameState()
    game.graph.generate_grid_graph(2, 2)
    game.add_player("player1")
    game.add_player("player2")
    game.start_game(starting_units=0)
    
    # Set up manual scenario
    vertex_a = game.graph.vertices[0]  # Player 1 controlled
    vertex_b = game.graph.vertices[1]  # Player 2 controlled
    
    vertex_a.controller = "player1"
    vertex_a.units = 10
    vertex_b.controller = "player2"
    vertex_b.units = 8
    
    engine = GameEngine(game)
    
    # Both players attack each other
    movements = {
        "player1": [Command("player1", 0, 1, 6)],  # Send 6 units A->B
        "player2": [Command("player2", 1, 0, 4)]   # Send 4 units B->A
    }
    
    print(f"Before: A has {vertex_a.units} units, B has {vertex_b.units} units")
    engine.resolve_all_movements(movements)
    print(f"After: A has {vertex_a.units} units, B has {vertex_b.units} units")
    print(f"A controlled by {vertex_a.controller}, B controlled by {vertex_b.controller}")
    
    # Expected: A=4 (10-6), B=2 (8-4-2 from remaining attack)
    assert vertex_a.units == 4, f"A should have 4 units, got {vertex_a.units}"
    assert vertex_b.units == 2, f"B should have 2 units, got {vertex_b.units}"
    assert vertex_a.controller == "player1", "A should still be controlled by player1"
    assert vertex_b.controller == "player2", "B should still be controlled by player2"
    
    # Test 2: Attack on neutral vertex with multiple attackers
    print("\n=== Test 2: Multiple Attacks on Neutral Vertex ===")
    game2 = GameState()
    game2.graph.generate_grid_graph(3, 3)
    game2.add_player("player1")
    game2.add_player("player2")
    game2.start_game(starting_units=0)
    
    # Set up scenario
    vertex_a = game2.graph.vertices[0]  # Player 1
    vertex_b = game2.graph.vertices[2]  # Player 2
    vertex_c = game2.graph.vertices[1]  # Neutral target (adjacent to both)
    
    vertex_a.controller = "player1"
    vertex_a.units = 10
    vertex_b.controller = "player2"
    vertex_b.units = 10
    vertex_c.controller = None
    vertex_c.units = 0
    vertex_c.weight = 2
    
    engine2 = GameEngine(game2)
    
    # Both attack the neutral vertex
    movements2 = {
        "player1": [Command("player1", 0, 1, 5)],
        "player2": [Command("player2", 2, 1, 3)]
    }
    
    print(f"Before: A={vertex_a.units}, B={vertex_b.units}, C={vertex_c.units} (neutral, weight={vertex_c.weight})")
    
    # Run multiple times to see probabilistic outcomes
    player1_wins = 0
    player2_wins = 0
    trials = 1000
    
    for trial in range(trials):
        # Reset state
        vertex_a.units = 10
        vertex_b.units = 10
        vertex_c.controller = None
        vertex_c.units = 0
        
        engine2.resolve_all_movements(movements2)
        
        if vertex_c.controller == "player1":
            player1_wins += 1
        elif vertex_c.controller == "player2":
            player2_wins += 1
    
    print(f"After {trials} trials: Player1 won {player1_wins} times, Player2 won {player2_wins} times")
    print(f"Expected ratio ~5:3, actual ratio ~{player1_wins}:{player2_wins}")
    
    # Test 3: Attack exceeding defender count
    print("\n=== Test 3: Overwhelming Attack ===")
    game3 = GameState()
    game3.graph.generate_grid_graph(2, 2)
    game3.add_player("attacker")
    game3.add_player("defender")
    game3.start_game(starting_units=0)
    
    attacker_vertex = game3.graph.vertices[0]
    defender_vertex = game3.graph.vertices[1]
    
    attacker_vertex.controller = "attacker"
    attacker_vertex.units = 15
    defender_vertex.controller = "defender"
    defender_vertex.units = 5
    
    engine3 = GameEngine(game3)
    
    movements3 = {
        "attacker": [Command("attacker", 0, 1, 10)]
    }
    
    print(f"Before: Attacker has {attacker_vertex.units}, Defender has {defender_vertex.units}")
    engine3.resolve_all_movements(movements3)
    print(f"After: Attacker vertex has {attacker_vertex.units}, Target controlled by {defender_vertex.controller} with {defender_vertex.units} units")
    
    # Expected: Attacker=5 (15-10), Target controlled by attacker with 5 units (10-5)
    assert attacker_vertex.units == 5, f"Attacker should have 5 units, got {attacker_vertex.units}"
    assert defender_vertex.controller == "attacker", f"Target should be controlled by attacker, controlled by {defender_vertex.controller}"
    assert defender_vertex.units == 5, f"Target should have 5 units, got {defender_vertex.units}"
    
    # Test 4: Equal forces tie
    print("\n=== Test 4: Equal Forces Tie ===")
    game4 = GameState()
    game4.graph.generate_grid_graph(2, 2)
    game4.add_player("attacker")
    game4.add_player("defender")
    game4.start_game(starting_units=0)
    
    attacker_vertex = game4.graph.vertices[0]
    defender_vertex = game4.graph.vertices[1]
    
    attacker_vertex.controller = "attacker"
    attacker_vertex.units = 10
    defender_vertex.controller = "defender"
    defender_vertex.units = 5
    
    engine4 = GameEngine(game4)
    
    movements4 = {
        "attacker": [Command("attacker", 0, 1, 5)]  # Equal to defender units
    }
    
    print(f"Before: Attacker has {attacker_vertex.units}, Defender has {defender_vertex.units}")
    engine4.resolve_all_movements(movements4)
    print(f"After: Attacker vertex has {attacker_vertex.units}, Target controlled by {defender_vertex.controller} with {defender_vertex.units} units")
    
    # Expected: Attacker=5 (10-5), Target becomes neutral with 0 units
    assert attacker_vertex.units == 5, f"Attacker should have 5 units, got {attacker_vertex.units}"
    assert defender_vertex.controller is None, f"Target should be neutral, controlled by {defender_vertex.controller}"
    assert defender_vertex.units == 0, f"Target should have 0 units, got {defender_vertex.units}"
    
    # Test 5: Insufficient attack
    print("\n=== Test 5: Insufficient Attack ===")
    game5 = GameState()
    game5.graph.generate_grid_graph(2, 2)
    game5.add_player("attacker")
    game5.add_player("defender")
    game5.start_game(starting_units=0)
    
    attacker_vertex = game5.graph.vertices[0]
    defender_vertex = game5.graph.vertices[1]
    
    attacker_vertex.controller = "attacker"
    attacker_vertex.units = 10
    defender_vertex.controller = "defender"
    defender_vertex.units = 8
    
    engine5 = GameEngine(game5)
    
    movements5 = {
        "attacker": [Command("attacker", 0, 1, 3)]  # Less than defender units
    }
    
    print(f"Before: Attacker has {attacker_vertex.units}, Defender has {defender_vertex.units}")
    engine5.resolve_all_movements(movements5)
    print(f"After: Attacker vertex has {attacker_vertex.units}, Target controlled by {defender_vertex.controller} with {defender_vertex.units} units")
    
    # Expected: Attacker=7 (10-3), Defender=5 (8-3), defender keeps control
    assert attacker_vertex.units == 7, f"Attacker should have 7 units, got {attacker_vertex.units}"
    assert defender_vertex.controller == "defender", f"Target should still be controlled by defender, controlled by {defender_vertex.controller}"
    assert defender_vertex.units == 5, f"Target should have 5 units, got {defender_vertex.units}"
    
    print("\nAll new combat system tests passed!")


def test_game_core():
    """
    Test function to verify the data structures work correctly.
    Creates a 3x3 grid with 2 players and tests basic functionality.
    """
    print("Testing game core data structures...")
    
    # Create game state
    game = GameState(max_turns=50)
    
    # Generate 3x3 grid
    game.graph.generate_grid_graph(3, 3)
    print(f"Created 3x3 grid with {len(game.graph.vertices)} vertices and {len(game.graph.edges)} edges")
    
    # Add players
    game.add_player("player1")
    game.add_player("player2")
    print(f"Added {len(game.players)} players")
    
    # Start game
    game.start_game(starting_units=5)
    print(f"Game started, turn {game.current_turn}")
    
    # Verify player starting positions
    for player in game.players.values():
        controlled = game.graph.get_vertices_controlled_by_player(player.id)
        print(f"{player.id}: controls {len(controlled)} vertices with {player.total_units} total units")
        
        # Test adjacency
        if controlled:
            vertex = controlled[0]
            adjacent = game.graph.get_adjacent_vertices(vertex.id)
            print(f"  Vertex {vertex.id} at {vertex.position} has {len(adjacent)} adjacent vertices: {adjacent}")
    
    # Test serialization
    state_dict = game.to_dict()
    print(f"Game state serialized to dict with {len(state_dict)} top-level keys")
    
    # Verify some graph properties
    assert len(game.graph.vertices) == 9, "Should have 9 vertices in 3x3 grid"
    assert len(game.graph.edges) == 24, "Should have 24 directed edges in 3x3 grid"  # 12 undirected = 24 directed
    
    # Test corner vertex (should have 2 neighbors)
    corner_neighbors = game.graph.get_adjacent_vertices(0)  # Top-left corner
    assert len(corner_neighbors) == 2, f"Corner vertex should have 2 neighbors, got {len(corner_neighbors)}"
    
    # Test center vertex (should have 4 neighbors)
    center_neighbors = game.graph.get_adjacent_vertices(4)  # Center vertex
    assert len(center_neighbors) == 4, f"Center vertex should have 4 neighbors, got {len(center_neighbors)}"
    
    print("All tests passed!")


if __name__ == "__main__":
    test_game_core()
    print()
    test_new_combat_system()
