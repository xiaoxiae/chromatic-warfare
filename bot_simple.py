#!/usr/bin/env python3
"""
Simple Bot - Expands to neutral adjacent vertices only, never attacks.

This bot implements a peaceful expansion strategy:
- Moves units from interior vertices toward the frontline
- Only expands to neutral (uncontrolled) vertices
- Never attacks enemy vertices
- Prioritizes the weakest neutral neighbors to minimize cost
- Maintains unit flow from back to front for sustained expansion
"""

from bot_lib import GameBot, GameState, Command
from typing import List


class SimpleBot(GameBot):
    """A simple bot that only expands to neutral vertices."""
    
    def play_turn(self, game_state: GameState) -> List[Command]:
        """
        Play a turn by expanding to neutral adjacent vertices and moving interior units forward.
        
        Strategy:
        1. Move units from interior vertices to frontline vertices
        2. Find all frontline vertices that have neutral neighbors
        3. For each such vertex, find the weakest neutral neighbor
        4. If we can afford to capture it, do so
        """
        commands = []
        
        # First, move units from interior to frontline
        interior_to_frontline_commands = self._move_interior_to_frontline()
        commands.extend(interior_to_frontline_commands)
        
        # Then, expand to neutral vertices from frontline positions
        expansion_commands = self._expand_to_neutrals()
        commands.extend(expansion_commands)
        
        return commands
    
    def _move_interior_to_frontline(self) -> List[Command]:
        """Move units from more interior vertices evenly to all closer neighbors."""
        commands = []
        
        # Go through all my vertices
        for vertex in self.game_state.my_vertices:
            if vertex.units <= 1:
                continue
            
            my_distance = self.get_distance_to_frontline(vertex.id)
            if my_distance <= 0:
                continue  # Already on frontline or invalid
            
            # Find all neighbors that are closer to frontline
            closer_neighbors = []
            for neighbor in self.game_state.get_my_neighbors(vertex.id):
                neighbor_distance = self.get_distance_to_frontline(neighbor.id)
                if neighbor_distance >= 0 and neighbor_distance < my_distance:
                    closer_neighbors.append(neighbor)
            
            if closer_neighbors:
                # Move ALL units evenly to closer neighbors (don't leave single units)
                units_per_neighbor = vertex.units // len(closer_neighbors)
                remaining_units = vertex.units % len(closer_neighbors)
                
                for i, neighbor in enumerate(closer_neighbors):
                    units_to_move = units_per_neighbor
                    if i < remaining_units:  # Distribute remainder
                        units_to_move += 1
                    
                    if units_to_move > 0:
                        command = self.reinforce(vertex, neighbor, units_to_move)
                        commands.append(command)
        
        return commands
    
    def _expand_to_neutrals(self) -> List[Command]:
        """Expand to neutral vertices from frontline positions."""
        commands = []
        
        # Get all frontline vertices
        frontline_vertices = self.get_frontline_vertices()
        
        for vertex in frontline_vertices:
            # Find the weakest neutral neighbor
            weakest_neutral = self.find_weakest_neutral_neighbor(vertex)
            
            if weakest_neutral is None:
                continue  # No neutral neighbors
            
            # Calculate cost to capture this neutral vertex
            cost = self.game_state.calculate_movement_cost(vertex.id, weakest_neutral.id)
            
            # We need at least cost + 1 units to capture (cost to pay + 1 to occupy)
            # Plus we want to keep 1 unit on the source vertex
            required_units = cost + 1  # cost + occupation + guard
            
            if vertex.units >= required_units:
                # We can afford this expansion
                units_to_send = vertex.units - 1  # Keep 1 unit to guard
                command = self.attack(vertex, weakest_neutral, units_to_send)
                commands.append(command)
        
        return commands


def main():
    """Run the simple bot."""
    import sys
    
    # Get bot ID and optional game ID from command line
    bot_id = sys.argv[1] if len(sys.argv) > 1 else "simple_bot"
    game_id = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Create and run the bot
    bot = SimpleBot(bot_id, game_id)
    print(f"Starting {bot_id} - peaceful expansion bot for game {bot.game_id}")
    bot.run()


if __name__ == "__main__":
    main()
