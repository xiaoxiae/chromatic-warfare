"""
Turn-Based Strategy Game - Optimized Core Data Structures

This module contains performance-optimized data structures for a turn-based strategy game
played on a directed planar graph.
"""

import math
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import random

from config import MapConfig


class PlayerStatus(Enum):
    """Enumeration for player status states."""
    ACTIVE = "active"
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

        self._player_vertices_cache: Dict[str, List[int]] = {}
        self._cache_dirty = True

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
        self._cache_dirty = True

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
        self._cache_dirty = True

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

    def mark_cache_dirty(self):
        """Mark the player vertices cache as needing update."""
        self._cache_dirty = True

    def _rebuild_player_cache(self):
        """Rebuild the cache of player-controlled vertices."""
        self._player_vertices_cache.clear()
        for vertex_id, vertex in self.vertices.items():
            if vertex.controller:
                if vertex.controller not in self._player_vertices_cache:
                    self._player_vertices_cache[vertex.controller] = []
                self._player_vertices_cache[vertex.controller].append(vertex_id)
        self._cache_dirty = False

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

        return self._adjacency[vertex_id]  # Return reference, not copy

    def get_vertices_controlled_by_player(self, player_id: str) -> List[Vertex]:
        """
        Get all vertices controlled by a specific player.

        Args:
            player_id: ID of the player

        Returns:
            List of vertices controlled by the player
        """
        if self._cache_dirty:
            self._rebuild_player_cache()

        vertex_ids = self._player_vertices_cache.get(player_id, [])
        return [self.vertices[vid] for vid in vertex_ids]

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
        self._player_vertices_cache.clear()
        self._cache_dirty = True

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
        self._player_vertices_cache.clear()
        self._cache_dirty = True

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

    def clear_players(self):
        """
        Clear players from the game.
        """
        self.players.clear()

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

        self.graph.mark_cache_dirty()
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
        ranked_players = set()  # Track which players have been ranked

        # Start with players still active (winners)
        active_players = [p for p in self.players.values()
                        if p.status == PlayerStatus.ACTIVE and p.total_units > 0]

        if active_players:
            if len(active_players) == 1:
                # Single winner
                self.final_rankings.append([active_players[0].id])
                ranked_players.add(active_players[0].id)
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
                        ranked_players.update(current_group)
                        current_group = [player.id]
                        current_units = player.total_units

                # Add final group
                if current_group:
                    self.final_rankings.append(current_group)
                    ranked_players.update(current_group)

        # Add eliminated players in reverse order (last eliminated = higher rank)
        for turn, eliminated_players in reversed(self.elimination_order):
            # Filter out any players we've already ranked
            unranked_eliminated = [p for p in eliminated_players if p not in ranked_players]
            if unranked_eliminated:
                self.final_rankings.append(unranked_eliminated)
                ranked_players.update(unranked_eliminated)

        # Finally, add any players who were never eliminated but also have 0 units
        # (e.g., disconnected players or edge cases)
        remaining_players = []
        for player_id in self.players:
            if player_id not in ranked_players:
                remaining_players.append(player_id)

        if remaining_players:
            self.final_rankings.append(remaining_players)

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

        if best_vertex_id is None:
            raise ValueError("No valid starting position found!")

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
    OPTIMIZED: Batch processing, caching, and efficient data structures.
    """

    def __init__(self, game_state: GameState):
        """
        Initialize the game engine with a game state.

        Args:
            game_state: The game state to operate on
        """
        self.game_state = game_state
        self._adjacency_cache = {}
        self._last_cache_turn = -1

    def _update_cache_if_needed(self):
        """Update adjacency cache if graph changed."""
        if self._last_cache_turn != self.game_state.current_turn:
            self._adjacency_cache.clear()
            self._last_cache_turn = self.game_state.current_turn

    def process_movement_command(self, player_id: str, from_vertex: int,
                               to_vertex: int, units: int) -> bool:
        """
        Validate a single movement command.
        OPTIMIZED: Cached adjacency checks.

        Args:
            player_id: ID of the player issuing the command
            from_vertex: Source vertex ID
            to_vertex: Destination vertex ID
            units: Number of units to move

        Returns:
            True if command is valid, False otherwise
        """
        # Quick rejection checks first
        if units <= 0 or from_vertex == to_vertex:
            return False

        if player_id not in self.game_state.players:
            return False

        if self.game_state.players[player_id].status != PlayerStatus.ACTIVE:
            return False

        # Check if vertices exist
        graph = self.game_state.graph
        if from_vertex not in graph.vertices or to_vertex not in graph.vertices:
            return False

        # Check if player controls source vertex
        source = graph.vertices[from_vertex]
        if source.controller != player_id:
            return False

        # Check if player has enough units
        if source.units < units:
            return False

        cache_key = (from_vertex, to_vertex)
        if cache_key not in self._adjacency_cache:
            self._adjacency_cache[cache_key] = to_vertex in graph._adjacency[from_vertex]

        return self._adjacency_cache[cache_key]

    def resolve_all_movements(self, movements_dict: Dict[str, List[Command]]) -> None:
        """
        Resolve all movement commands for the current turn.
        OPTIMIZED: Single-pass validation and batch processing.

        Args:
            movements_dict: Dictionary mapping player IDs to their list of commands
        """
        self._update_cache_if_needed()

        valid_movements = []
        unit_requirements = {}
        graph = self.game_state.graph

        # Step 1: Batch validate all commands
        for player_id, commands in movements_dict.items():
            # Skip inactive players entirely
            if player_id not in self.game_state.players:
                continue
            if self.game_state.players[player_id].status != PlayerStatus.ACTIVE:
                continue

            for command in commands:
                # Quick validation without method call overhead
                if (command.units > 0 and
                    command.from_vertex != command.to_vertex and
                    command.from_vertex in graph.vertices and
                    command.to_vertex in graph.vertices):

                    source_vertex = graph.vertices[command.from_vertex]

                    # Check control and adjacency
                    if (source_vertex.controller == player_id and
                        command.to_vertex in graph._adjacency[command.from_vertex]):

                        # Check unit availability
                        current_requirement = unit_requirements.get(command.from_vertex, 0)
                        if current_requirement + command.units <= source_vertex.units:
                            unit_requirements[command.from_vertex] = current_requirement + command.units
                            valid_movements.append(command)

        # Step 2: Batch deduct units from source vertices
        for from_vertex, units in unit_requirements.items():
            graph.vertices[from_vertex].units -= units

        # Step 3: Process mutual cancellations
        movements_after_cancellation = self._process_mutual_cancellations_optimized(valid_movements)

        # Step 4: Batch resolve conflicts at destinations
        movements_by_destination = {}
        for command in movements_after_cancellation:
            if command.to_vertex not in movements_by_destination:
                movements_by_destination[command.to_vertex] = []
            movements_by_destination[command.to_vertex].append(command)

        # Mark cache as dirty since we're about to change controllers
        graph.mark_cache_dirty()

        for dest_vertex_id, movements in movements_by_destination.items():
            self._resolve_vertex_conflict_optimized(dest_vertex_id, movements)

    def _process_mutual_cancellations_optimized(self, movements: List[Command]) -> List[Command]:
        """
        Process mutual cancellations between opposing movements.
        OPTIMIZED: Single-pass processing with efficient lookups.

        Args:
            movements: List of all valid movement commands

        Returns:
            List of movements after cancellations
        """
        movement_map = {}
        movement_players = {}

        for command in movements:
            route = (command.from_vertex, command.to_vertex)
            movement_map[route] = movement_map.get(route, 0) + command.units
            movement_players[route] = command.player_id

        # Process cancellations
        remaining_movements = []
        processed = set()

        for route, units in movement_map.items():
            if route in processed:
                continue

            from_v, to_v = route
            reverse_route = (to_v, from_v)

            if reverse_route in movement_map:
                reverse_units = movement_map[reverse_route]

                if units > reverse_units:
                    remaining_movements.append(Command(
                        movement_players[route], from_v, to_v, units - reverse_units
                    ))
                elif reverse_units > units:
                    remaining_movements.append(Command(
                        movement_players[reverse_route], to_v, from_v, reverse_units - units
                    ))

                processed.add(route)
                processed.add(reverse_route)
            else:
                remaining_movements.append(Command(
                    movement_players[route], from_v, to_v, units
                ))
                processed.add(route)

        return remaining_movements

    def _resolve_vertex_conflict_optimized(self, vertex_id: int, movements: List[Command]) -> None:
        """
        Resolve conflicts at a single vertex.
        OPTIMIZED: Direct vertex access and efficient probability calculation.

        Args:
            vertex_id: ID of the destination vertex
            movements: List of movement commands targeting this vertex
        """
        if not movements:
            return

        vertex = self.game_state.graph.vertices[vertex_id]

        attacks_by_player = {}
        for movement in movements:
            player_id = movement.player_id
            attacks_by_player[player_id] = attacks_by_player.get(player_id, 0) + movement.units

        # Handle different vertex states
        if vertex.controller is None:
            # Neutral vertex
            self._resolve_neutral_vertex_conflict_optimized(vertex, attacks_by_player)
        else:
            # Controlled vertex - check if controller is reinforcing
            controller_reinforcement = attacks_by_player.pop(vertex.controller, 0)
            vertex.units += controller_reinforcement

            if attacks_by_player:
                self._resolve_controlled_vertex_conflict_optimized(vertex, attacks_by_player)

    def _resolve_neutral_vertex_conflict_optimized(self, vertex: Vertex,
                                                  attacks_by_player: Dict[str, int]) -> None:
        """
        Resolve conflict at a neutral vertex.
        OPTIMIZED: Efficient probability calculation.
        """
        if not attacks_by_player:
            return

        total_attacking_units = sum(attacks_by_player.values())

        # Pay the weight cost
        if total_attacking_units < vertex.weight:
            return

        effective_units = total_attacking_units - vertex.weight

        rand_val = random.random() * total_attacking_units
        cumulative = 0

        for player_id, units in attacks_by_player.items():
            cumulative += units
            if rand_val <= cumulative:
                vertex.controller = player_id
                vertex.units = effective_units
                break

    def _resolve_controlled_vertex_conflict_optimized(self, vertex: Vertex,
                                                     attacks_by_player: Dict[str, int]) -> None:
        """
        Resolve conflict at a controlled vertex.
        OPTIMIZED: Streamlined combat resolution.
        """
        if not attacks_by_player:
            return

        total_attacking_units = sum(attacks_by_player.values())
        defending_units = vertex.units

        if total_attacking_units < defending_units:
            vertex.units = defending_units - total_attacking_units
        elif total_attacking_units == defending_units:
            vertex.controller = None
            vertex.units = 0
        else:
            remaining_attackers = total_attacking_units - defending_units

            if len(attacks_by_player) == 1:
                winner = next(iter(attacks_by_player))
                vertex.controller = winner
                vertex.units = remaining_attackers
            else:
                rand_val = random.random() * total_attacking_units
                cumulative = 0

                for player_id, units in attacks_by_player.items():
                    cumulative += units
                    if rand_val <= cumulative:
                        vertex.controller = player_id
                        vertex.units = remaining_attackers
                        break

    def generate_units(self) -> None:
        """
        Generate new units at all controlled vertices based on their weight.
        OPTIMIZED: Single pass with no function calls.
        """
        for vertex in self.game_state.graph.vertices.values():
            if vertex.controller is not None:
                vertex.units += vertex.weight

    def check_eliminations(self) -> List[str]:
        """
        Check for player eliminations and update their status.
        OPTIMIZED: Batch processing with cached lookups.

        Returns:
            List of player IDs that were eliminated this turn
        """
        eliminated_players = []
        graph = self.game_state.graph

        player_units = {}
        player_vertices = {}

        for vertex in graph.vertices.values():
            if vertex.controller is not None:
                if vertex.controller not in player_units:
                    player_units[vertex.controller] = 0
                    player_vertices[vertex.controller] = 0
                player_units[vertex.controller] += vertex.units
                player_vertices[vertex.controller] += 1

        # Check each active player
        for player_id, player in self.game_state.players.items():
            if player.status == PlayerStatus.ACTIVE:
                total_units = player_units.get(player_id, 0)
                vertex_count = player_vertices.get(player_id, 0)

                player.total_units = total_units

                if vertex_count == 0 or total_units == 0:
                    eliminated_players.append(player_id)

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
        OPTIMIZED: Batch processing throughout.

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
