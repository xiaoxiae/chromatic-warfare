"""
Turn-Based Strategy Game - Core Data Structures

This module contains the foundational data structures for a turn-based strategy game
played on a directed planar graph.
"""

from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import random
from collections import defaultdict


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
    
    def generate_grid_graph(self, width: int, height: int, base_weight: int = 1) -> None:
        """
        Generate a grid graph with bidirectional connections between adjacent cells.
        
        Args:
            width: Width of the grid
            height: Height of the grid
            base_weight: Weight to assign to all vertices
            
        Raises:
            ValueError: If dimensions are not positive
        """
        if width <= 0 or height <= 0:
            raise ValueError("Grid dimensions must be positive")
        
        # Clear existing graph
        self.vertices.clear()
        self.edges.clear()
        self._adjacency.clear()
        
        # Create vertices
        for y in range(height):
            for x in range(width):
                vertex_id = y * width + x
                position = (float(x), float(y))
                vertex = Vertex(vertex_id, base_weight, position)
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
        
        # Update status based on unit count
        if self.total_units == 0 and len(controlled_vertices) == 0:
            self.status = PlayerStatus.ELIMINATED
    
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
    
    def get_winner(self) -> Optional[List[str]]:
        """
        Get the winner(s) of the game.
        
        Returns:
            List of winning player IDs, or None if game not ended
        """
        if self.status == GameStatus.ENDED and self.final_rankings:
            return self.final_rankings[0]  # First group is the winners
        return None
    
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
            "winner": self.get_winner()
        }


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
        Validate and process a single movement command.
        
        Args:
            player_id: ID of the player issuing the command
            from_vertex: Source vertex ID
            to_vertex: Destination vertex ID
            units: Number of units to move
            
        Returns:
            True if command is valid and can be executed, False otherwise
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
            
            # Calculate movement cost and check if player has enough units
            cost = self._calculate_movement_cost(from_vertex, to_vertex)
            if source.units < units + cost:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _calculate_movement_cost(self, from_vertex: int, to_vertex: int) -> int:
        """
        Calculate the cost to move units from source to destination vertex.
        
        Args:
            from_vertex: Source vertex ID
            to_vertex: Destination vertex ID
            
        Returns:
            Movement cost in units
        """
        source = self.game_state.graph.vertices[from_vertex]
        destination = self.game_state.graph.vertices[to_vertex]
        
        # No cost between own controlled vertices
        if destination.controller is not None and destination.controller == source.controller:
            return 0
        
        # Cost equals vertex weight for neutral vertices
        if destination.controller is None:
            return destination.weight
        
        # Cost equals defending units for enemy vertices
        return destination.units
    
    def resolve_all_movements(self, movements_dict: Dict[str, List[Command]]) -> None:
        """
        Resolve all movement commands for the current turn.
        
        Args:
            movements_dict: Dictionary mapping player IDs to their list of commands
        """
        # Group movements by destination vertex
        movements_by_destination: Dict[int, List[Command]] = defaultdict(list)
        
        # Validate and collect all valid movements
        valid_movements = []
        for player_id, commands in movements_dict.items():
            for command in commands:
                if self.process_movement_command(command.player_id, command.from_vertex, 
                                               command.to_vertex, command.units):
                    valid_movements.append(command)
                    movements_by_destination[command.to_vertex].append(command)
        
        # Deduct units and movement costs from source vertices
        for command in valid_movements:
            source = self.game_state.graph.vertices[command.from_vertex]
            cost = self._calculate_movement_cost(command.from_vertex, command.to_vertex)
            source.units -= (command.units + cost)
        
        # Resolve conflicts for each destination vertex
        for dest_vertex_id, movements in movements_by_destination.items():
            self._resolve_vertex_conflict(dest_vertex_id, movements)
    
    def _resolve_vertex_conflict(self, vertex_id: int, movements: List[Command]) -> None:
        """
        Resolve conflicts at a single vertex with potentially multiple attackers.
        
        Args:
            vertex_id: ID of the destination vertex
            movements: List of movement commands targeting this vertex
        """
        if not movements:
            return
        
        vertex = self.game_state.graph.vertices[vertex_id]
        
        # If only one movement and vertex is controlled by same player, just add units
        if len(movements) == 1 and vertex.controller == movements[0].player_id:
            vertex.units += movements[0].units
            return
        
        # Group movements by attacking player
        attacks_by_player = defaultdict(int)
        for movement in movements:
            attacks_by_player[movement.player_id] += movement.units
        
        # Handle different conflict scenarios
        if vertex.controller is None:
            # Neutral vertex - probabilistic resolution among attackers
            self._resolve_neutral_vertex_conflict(vertex, attacks_by_player)
        else:
            # Enemy-controlled vertex
            self._resolve_enemy_vertex_conflict(vertex, attacks_by_player)
    
    def _resolve_neutral_vertex_conflict(self, vertex: Vertex, 
                                       attacks_by_player: Dict[str, int]) -> None:
        """
        Resolve conflict at a neutral vertex using probabilistic resolution.
        
        Args:
            vertex: The neutral vertex being attacked
            attacks_by_player: Dictionary mapping player IDs to total attacking units
        """
        if not attacks_by_player:
            return
        
        total_attacking_units = sum(attacks_by_player.values())
        
        # Calculate probabilities based on unit ratios
        probabilities = {player_id: units / total_attacking_units 
                        for player_id, units in attacks_by_player.items()}
        
        # Randomly select winner based on probabilities
        rand = random.random()
        cumulative_prob = 0
        winner = None
        
        for player_id, prob in probabilities.items():
            cumulative_prob += prob
            if rand <= cumulative_prob:
                winner = player_id
                break
        
        if winner:
            # Winner takes the vertex with their attacking units
            vertex.controller = winner
            vertex.units = attacks_by_player[winner]
            
            # Other players' units return home (handled implicitly as they were already deducted)
    
    def _resolve_enemy_vertex_conflict(self, vertex: Vertex, 
                                     attacks_by_player: Dict[str, int]) -> None:
        """
        Resolve conflict at an enemy-controlled vertex.
        
        Args:
            vertex: The enemy-controlled vertex being attacked
            attacks_by_player: Dictionary mapping player IDs to total attacking units
        """
        if not attacks_by_player:
            return
        
        original_controller = vertex.controller
        original_units = vertex.units
        
        # Remove original controller from attackers if present (friendly fire not allowed)
        attacks_by_player.pop(original_controller, None)
        
        if not attacks_by_player:
            return
        
        # If multiple attackers, resolve probabilistically among them first
        if len(attacks_by_player) > 1:
            total_attacking_units = sum(attacks_by_player.values())
            probabilities = {player_id: units / total_attacking_units 
                           for player_id, units in attacks_by_player.items()}
            
            # Select primary attacker
            rand = random.random()
            cumulative_prob = 0
            primary_attacker = None
            
            for player_id, prob in probabilities.items():
                cumulative_prob += prob
                if rand <= cumulative_prob:
                    primary_attacker = player_id
                    break
            
            if primary_attacker:
                attacking_units = attacks_by_player[primary_attacker]
            else:
                return
        else:
            # Single attacker
            primary_attacker = next(iter(attacks_by_player))
            attacking_units = attacks_by_player[primary_attacker]
        
        # Resolve combat: 1-to-1 unit destruction
        if attacking_units > original_units:
            # Attacker wins
            vertex.controller = primary_attacker
            vertex.units = attacking_units - original_units
        elif attacking_units == original_units:
            # Tie - vertex becomes neutral
            vertex.controller = None
            vertex.units = 0
        else:
            # Defender wins
            vertex.units = original_units - attacking_units
    
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
                    player.status = PlayerStatus.ELIMINATED
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
            "winner": None
        }
        
        # Phase 1: Resolve all movements and combat
        self.resolve_all_movements(movements_dict)
        
        # Phase 2: Generate new units
        self.generate_units()
        
        # Phase 3: Check eliminations
        eliminated = self.check_eliminations()
        turn_results["eliminations"] = eliminated
        
        # Phase 4: Check if game is over
        if self.is_game_over():
            turn_results["game_over"] = True
            turn_results["winner"] = self.game_state.winner
        else:
            # Advance to next turn
            self.game_state.current_turn += 1
        
        return turn_results


def test_game_engine():
    """
    Test the game engine functionality including combat resolution and movement costs.
    """
    print("Testing game engine mechanics...")
    
    # Set up game
    game = GameState(max_turns=10)
    game.graph.generate_grid_graph(3, 3, base_weight=1)
    game.add_player("player1")
    game.add_player("player2")
    game.start_game(starting_units=5)
    
    engine = GameEngine(game)
    
    # Get player starting positions
    p1_vertices = game.graph.get_vertices_controlled_by_player("player1")
    p2_vertices = game.graph.get_vertices_controlled_by_player("player2")
    p1_start = p1_vertices[0].id
    p2_start = p2_vertices[0].id
    
    print(f"Player 1 starts at vertex {p1_start}, Player 2 at vertex {p2_start}")
    
    # Test 1: Valid movement command validation
    adjacent_to_p1 = list(game.graph.get_adjacent_vertices(p1_start))
    target_vertex = adjacent_to_p1[0] if adjacent_to_p1 else None
    
    if target_vertex is not None:
        valid = engine.process_movement_command("player1", p1_start, target_vertex, 2)
        print(f"Valid movement command: {valid}")
        assert valid, "Valid movement should return True"
        
        # Test movement cost calculation
        cost = engine._calculate_movement_cost(p1_start, target_vertex)
        target = game.graph.vertices[target_vertex]
        if target.controller is None:
            expected_cost = target.weight
        elif target.controller == "player1":
            expected_cost = 0
        else:
            expected_cost = target.units
        print(f"Movement cost: {cost}, expected: {expected_cost}")
        assert cost == expected_cost, f"Movement cost should be {expected_cost}, got {cost}"
    
    # Test 2: Invalid movement (not enough units)
    p1_vertex = game.graph.vertices[p1_start]
    invalid = engine.process_movement_command("player1", p1_start, target_vertex, p1_vertex.units + 10)
    print(f"Invalid movement (too many units): {not invalid}")
    assert not invalid, "Invalid movement should return False"
    
    # Test 3: Process a turn with movements
    movements = {
        "player1": [Command("player1", p1_start, target_vertex, 2)],
        "player2": []
    }
    
    initial_units = {p.id: p.total_units for p in game.players.values()}
    turn_result = engine.process_turn(movements)
    
    print(f"Turn result: {turn_result}")
    print(f"Game turn advanced to: {game.current_turn}")
    
    # Verify units were generated
    for player in game.players.values():
        player.update_total_units(game.graph)
        print(f"{player.id}: {initial_units[player.id]} -> {player.total_units} units")
    
    # Test 4: Combat resolution setup
    print("\nTesting combat resolution...")
    
    # Create a simple combat scenario
    game2 = GameState()
    game2.graph.generate_grid_graph(2, 2)
    game2.add_player("attacker")
    game2.add_player("defender")
    game2.start_game(starting_units=0)
    
    # Set up combat scenario manually
    attacker_vertex = game2.graph.vertices[0]
    defender_vertex = game2.graph.vertices[1]
    
    attacker_vertex.controller = "attacker"
    attacker_vertex.units = 10
    defender_vertex.controller = "defender"
    defender_vertex.units = 5
    
    engine2 = GameEngine(game2)
    
    # Verify the vertices are adjacent
    adjacent = game2.graph.get_adjacent_vertices(0)
    print(f"Vertex 0 is adjacent to: {adjacent}")
    assert 1 in adjacent, "Vertices 0 and 1 should be adjacent"
    
    # Test movement cost calculation
    cost = engine2._calculate_movement_cost(0, 1)
    print(f"Movement cost from 0 to 1: {cost} (should be {defender_vertex.units})")
    assert cost == defender_vertex.units, f"Cost should be {defender_vertex.units}, got {cost}"
    
    # Attack with 7 units (cost is 5, so need 12 total units, but we have 10)
    # Let's give the attacker enough units
    attacker_vertex.units = 15  # 7 to send + 5 cost = 12, plus extra
    
    combat_movements = {
        "attacker": [Command("attacker", 0, 1, 7)]
    }
    
    print(f"Before combat: Attacker has {attacker_vertex.units}, Defender has {defender_vertex.units}")
    print(f"Attacking with 7 units, cost is {cost}")
    
    # Validate the command first
    valid = engine2.process_movement_command("attacker", 0, 1, 7)
    print(f"Attack command valid: {valid}")
    assert valid, "Attack command should be valid"
    
    engine2.resolve_all_movements(combat_movements)
    
    print(f"After combat: Vertex 0 has {attacker_vertex.units}, Vertex 1 controlled by {defender_vertex.controller} with {defender_vertex.units} units")
    
    # Attacker should win (7 > 5), leaving 2 units
    assert defender_vertex.controller == "attacker", f"Attacker should control the vertex, but it's controlled by {defender_vertex.controller}"
    assert defender_vertex.units == 2, f"Should have 2 units remaining, got {defender_vertex.units}"
    
    print("All engine tests passed!")


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
    test_game_engine()
