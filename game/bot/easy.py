#!/usr/bin/env python3
import os
import sys
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from botlib import GameBot, Game, Command


class EasyBot(GameBot):
    def play_turn(self, game: Game) -> List[Command]:
        """
        - Capture neighbouring neutral vertices
        """
        return self._expand_to_neutrals()

    def _expand_to_neutrals(self) -> List[Command]:
        """Expand to neutral vertices from frontline positions using all available units."""
        commands = []
        
        for vertex in self.game.get_frontline_vertices():
            # Find weakest neutral neighbour
            neutral_neighbors = self.game.get_neutral_neighbors(vertex.id)

            if not neutral_neighbors:
                continue

            weakest_neutral = min(neutral_neighbors, key=lambda v: v.weight)
            
            if weakest_neutral is None:
                continue
            
            # Capture with all units if we can
            if self.game.can_capture(vertex.id, weakest_neutral.id, vertex.units):
                command = self.move_all(vertex, weakest_neutral)
                commands.append(command)
        
        return commands


if __name__ == "__main__":
    game_id = sys.argv[1] if len(sys.argv) > 1 else None
    player_id = sys.argv[2] if len(sys.argv) > 2 else None

    bot = EasyBot(game_id, player_id)
    bot.run()
