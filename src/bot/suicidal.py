#!/usr/bin/env python3
import os
import sys
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from botlib import GameBot, GameState, Command


class SuicidalBot(GameBot):
    def play_turn(self, game_state: GameState) -> List[Command]:
        """
        Suicidal strategy that attacks enemies regardless of odds:
        - Attack any enemy neighbor with all available units
        - Prioritize attacking the strongest enemies first (more "suicidal")
        - No defensive or consolidation moves - pure aggression
        """
        return self._attack_everything()

    def _attack_everything(self) -> List[Command]:
        """Attack all enemy neighbors with maximum force, regardless of win probability."""
        commands = []
        
        # Get all vertices that can potentially attack
        attacking_vertices = [v for v in self.game_state.my_vertices if v.units > 0]
        
        for vertex in attacking_vertices:
            # Find all enemy neighbors
            enemy_neighbors = self.game_state.get_enemy_neighbors(vertex.id)
            
            if not enemy_neighbors:
                continue
            
            # Sort enemies by strength (strongest first for maximum "suicide" effect)
            # This ensures we attack the most dangerous enemies first
            enemy_neighbors.sort(key=lambda e: e.units, reverse=True)
            
            # Pick the strongest enemy to attack (most likely to result in our defeat)
            target_enemy = enemy_neighbors[0]
            
            # Attack with ALL units - no holding back, no strategic reserves
            units_to_send = vertex.units
            
            if units_to_send > 0:
                command = self.build_command(vertex.id, target_enemy.id, units_to_send)
                commands.append(command)
                
                # Optional: Add some "battle cry" logic by targeting multiple enemies if possible
                # But since we're sending all units to one target, this vertex is now empty
        
        # If no enemy attacks possible, at least try to attack neutrals aggressively
        if not commands:
            commands = self._attack_neutrals_suicidally()
        
        return commands

    def _attack_neutrals_suicidally(self) -> List[Command]:
        """If no enemies available, attack neutrals with reckless abandon."""
        commands = []
        
        for vertex in self.game_state.my_vertices:
            if vertex.units == 0:
                continue
                
            neutral_neighbors = self.game_state.get_neutral_neighbors(vertex.id)
            
            if not neutral_neighbors:
                continue
            
            # Attack the hardest neutral target (most "suicidal" choice)
            hardest_neutral = max(neutral_neighbors, key=lambda n: n.weight)
            
            # Send all units regardless of whether we can actually capture it
            command = self.build_command(vertex.id, hardest_neutral.id, vertex.units)
            commands.append(command)
        
        return commands


def main():
    game_id = sys.argv[1] if len(sys.argv) > 1 else None
    
    bot = SuicidalBot(game_id)
    bot.run()


if __name__ == "__main__":
    main()
