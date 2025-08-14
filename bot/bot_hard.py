#!/usr/bin/env python3
import sys
from botlib import GameBot, GameState, Command
from typing import List


class HardBot(GameBot):
    def play_turn(self, game_state: GameState) -> List[Command]:
        """
        Enhanced strategy that includes:
        - Attack enemy vertices when advantageous
        - Move units from interior vertices to frontline vertices
        - Capture neighbouring neutral vertices
        """
        # Priority order: attack enemies first, then consolidate, then expand
        return (self._attack_enemies() + 
                self._move_to_frontline() + 
                self._expand_to_neutrals())

    def _attack_enemies(self) -> List[Command]:
        """Attack enemy vertices from frontline positions when we have superiority."""
        commands = []
        
        for vertex in self.get_frontline_vertices():
            # Find all enemy neighbors we can capture
            enemy_neighbors = self.game_state.get_enemy_neighbors(vertex.id)
            
            if not enemy_neighbors:
                continue
            
            # Find the most valuable enemy target we can capture
            capturable_enemies = []
            for enemy in enemy_neighbors:
                if self.game_state.can_capture(vertex.id, enemy.id, vertex.units):
                    # Calculate value: units gained vs units spent
                    units_gained = enemy.units
                    units_spent = enemy.units + 1  # minimum needed to capture
                    value = units_gained - units_spent + enemy.weight  # include vertex weight as bonus
                    capturable_enemies.append((enemy, value, units_spent))
            
            if not capturable_enemies:
                continue
            
            # Sort by value (highest first), then by cost (lowest first)
            capturable_enemies.sort(key=lambda x: (-x[1], x[2]))
            best_target, _, min_units_needed = capturable_enemies[0]
            
            # Attack with optimal force (just enough to capture + small buffer for safety)
            units_to_send = min(vertex.units, min_units_needed + 1)
            
            if units_to_send > 0:
                command = self.build_command(vertex.id, best_target.id, units_to_send)
                commands.append(command)
        
        return commands

    def _move_to_frontline(self) -> List[Command]:
        """Move units from interior vertices to frontline vertices, prioritizing weaker frontline positions."""
        commands = []
        
        for vertex in self.game_state.my_vertices:
            distance_to_frontline = self.get_distance_to_frontline(vertex.id)
            
            # Skip if this vertex is already on frontline or has no path to frontline
            if distance_to_frontline <= 0:
                continue
            
            # Find all neighbors that are closer to frontline
            closer_neighbors = []
            for neighbor in self.game_state.get_my_neighbors(vertex.id):
                neighbor_distance = self.get_distance_to_frontline(neighbor.id)
                if neighbor_distance >= 0 and neighbor_distance < distance_to_frontline:
                    closer_neighbors.append(neighbor)
            
            if not closer_neighbors:
                continue

            # Prioritize sending to the weakest neighbor (needs reinforcement most)
            weakest_neighbor = min(closer_neighbors, key=lambda v: v.units)
            
            # Send all units to strengthen the frontline
            command = self.move_all(vertex, weakest_neighbor)
            commands.append(command)
        
        return commands
    
    def _expand_to_neutrals(self) -> List[Command]:
        """Expand to neutral vertices from frontline positions, targeting the most efficient captures."""
        commands = []
        
        for vertex in self.get_frontline_vertices():
            # Skip if we already used this vertex for attacking enemies
            if any(cmd.from_vertex == vertex.id for cmd in commands):
                continue
                
            neutral_neighbors = self.game_state.get_neutral_neighbors(vertex.id)
            
            if not neutral_neighbors:
                continue

            # Find the most efficient neutral target (lowest weight = easiest to capture)
            capturable_neutrals = []
            for neutral in neutral_neighbors:
                if self.game_state.can_capture(vertex.id, neutral.id, vertex.units):
                    # Efficiency = units gained per unit spent
                    efficiency = neutral.weight / max(1, neutral.weight)  # avoid division by zero
                    capturable_neutrals.append((neutral, efficiency))
            
            if not capturable_neutrals:
                continue
            
            # Sort by efficiency (higher is better for neutrals, we want easy captures)
            capturable_neutrals.sort(key=lambda x: x[1], reverse=True)
            best_neutral, _ = capturable_neutrals[0]
            
            # Capture with all available units
            command = self.move_all(vertex, best_neutral)
            commands.append(command)
        
        return commands


def main():
    game_id = sys.argv[1] if len(sys.argv) > 1 else None
    
    bot = HardBot("hard_bot", game_id)
    bot.run()


if __name__ == "__main__":
    main()
